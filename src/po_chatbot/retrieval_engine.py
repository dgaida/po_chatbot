"""Hybrid Retrieval Engine for the RAG system.

Combines dense vector search (ChromaDB) with sparse keyword search (BM25)
and applies a Cross-Encoder for re-ranking. Uses strict metadata filtering.
"""

import os
import json
import re
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class HybridRetrievalEngine:
    """Engine for performing hybrid retrieval using vector and keyword search."""

    def __init__(self) -> None:
        """Initializes the retrieval engine components: ChromaDB, BM25 and CrossEncoder."""
        self.db_path: str = os.getenv("DB_PATH", os.path.join("data", "chroma_db"))
        self.client: chromadb.PersistentClient = chromadb.PersistentClient(
            path=self.db_path
        )
        self.emb_model: str = os.getenv(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-large"
        )
        self.emb_func: embedding_functions.SentenceTransformerEmbeddingFunction = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.emb_model
            )
        )
        self.collection = self.client.get_collection(
            name="th_koeln_rules", embedding_function=self.emb_func
        )

        self.chunks_path: str = os.getenv("CHUNKS_PATH", os.path.join("data", "chunks.json"))
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks: List[Dict[str, Any]] = json.load(f)

        # BM25 runs on a text optimized for lexical matching (title/type + content)
        tokenized_corpus: List[List[str]] = [
            self._tokenize(doc.get("bm25_text", doc.get("content", "")))
            for doc in self.chunks
        ]
        self.bm25: BM25Okapi = BM25Okapi(tokenized_corpus)

        self.reranker_model: str = os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        self.reranker: CrossEncoder = CrossEncoder(self.reranker_model)

    def _normalize_study_program(self, s: str) -> str:
        """Normalizes a study program name for comparison.

        Args:
            s: The study program name to normalize.

        Returns:
            The normalized study program name.
        """
        if not s:
            return ""
        s = s.lower().strip()
        s = re.sub(r"\s*\(.*?\)\s*", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _meta_matches_study_program(
        self, meta: Dict[str, Any], study_program_filter: Optional[str]
    ) -> bool:
        """Checks if the metadata matches the study program filter.

        Args:
            meta: The metadata dictionary of a chunk.
            study_program_filter: The study program name to filter by.

        Returns:
            True if it matches, False otherwise.
        """
        if not study_program_filter:
            return True
        meta_sp = self._normalize_study_program((meta or {}).get("study_program", ""))
        wanted = self._normalize_study_program(study_program_filter)
        if not wanted:
            return True
        if meta_sp in ("", "allgemein"):
            return True
        # "Alle Studiengänge der F10", "Alle (BWL, BaF, ...)" etc.
        if meta_sp.startswith("alle"):
            return True
        return wanted in meta_sp or meta_sp in wanted

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25 processing.

        Args:
            text: The text to tokenize.

        Returns:
            A list of tokens.
        """
        return re.findall(r"\w+", text.lower())

    def _has_auslauf_intent(self, q_lower: str) -> bool:
        """Detects if the question aims at a phase-out regulation (Auslaufordnung).

        Args:
            q_lower: The lowercase question string.

        Returns:
            True if phase-out intent is detected, False otherwise.
        """
        # Direct phase-out keywords
        if any(
            w in q_lower
            for w in [
                "auslauf",
                "alte prüfungsordnung",
                "alten prüfungsordnung",
                "alte po",
                "alten po",
                "bis wann fertig",
                "abschließen",
            ]
        ):
            return True
        # PO context detection: 'prüfungsordnung' or 'po '
        has_po_context = (
            "prüfungsordnung" in q_lower
            or "neue po" in q_lower
            or "neuen po" in q_lower
            or "alten po" in q_lower
            or "alte po" in q_lower
        )
        # Recognition keywords
        recognition_kw = any(
            w in q_lower
            for w in [
                "anerkannt",
                "anerkennung",
                "anerkenn",
                "anrechn",
                "anrechnung",
                "übernommen",
                "übertrag",
                "übernehm",
                "äquivalenz",
                "gleichwertig",
                "gelten",
                "gültig",
            ]
        )
        # Switching keywords
        switch_kw = any(
            w in q_lower for w in ["wechsl", "wechsel", "umstieg", "umsteig"]
        )
        # Recognition/Switching + PO context = phase-out regulation
        if has_po_context and (recognition_kw or switch_kw):
            return True
        # Independent recognition with module context
        if recognition_kw and any(w in q_lower for w in ["modul", "leistung"]):
            return True
        return False

    def _apply_heuristic_boost(
        self, question: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Applies rule-based priority boosting to candidates.

        Args:
            question: The user question.
            candidates: List of retrieval candidates.

        Returns:
            The list of candidates with updated priorities.
        """
        q_lower = question.lower()

        # Intent recognition from the question
        form_intent = any(
            w in q_lower
            for w in ["antrag", "formular", "anmeldung", "bescheinigung", "zulassung"]
        )
        verlaengerung_intent = any(
            w in q_lower for w in ["verlänger", "abgabefrist", "fristverlängerung"]
        )
        po_intent = any(
            w in q_lower
            for w in ["prüfungsordnung", "po4", "po5", "po6", "po 4", "po 5"]
        )
        auslauf_intent = self._has_auslauf_intent(q_lower)

        boosted_candidates = []
        for cand in candidates:
            priority = 0
            title = cand["meta"].get("title", "").lower()
            doc_type = cand["meta"].get("doc_type", "").lower()

            # Boost phase-out, demote regular PO
            if auslauf_intent:
                if "auslauf" in title or "auslauf" in doc_type:
                    priority -= 250
                elif "prüfungsordnung" in doc_type:
                    priority += 30

            # Boost extension, demote admission/registration
            elif verlaengerung_intent:
                if "verlängerung" in title:
                    priority -= 200
                elif any(w in title for w in ["zulassung", "anmeldung"]):
                    priority += 50

            # Form/registration recognition
            elif form_intent:
                if any(w in doc_type for w in ["formular", "antrag"]) or any(
                    w in title for w in ["formular", "antrag", "anmeldung"]
                ):
                    priority -= 100

            # PO recognition: Boost PO texts, demote forms
            if po_intent and not auslauf_intent:
                if "prüfungsordnung" in doc_type or "prüfungsordnung" in title:
                    priority -= 150
                elif any(w in doc_type for w in ["formular", "antrag"]):
                    priority += 50

            cand["priority"] = priority
            boosted_candidates.append(cand)

        return boosted_candidates

    def _inject_keyword_matches(
        self,
        question: str,
        faculty_filter: str,
        study_program_filter: Optional[str],
        candidates_map: Dict[int, Dict[str, Any]],
    ) -> None:
        """Injects chunks whose titles contain explicitly mentioned terms.

        Args:
            question: The user question.
            faculty_filter: The faculty to filter by.
            study_program_filter: The study program to filter by.
            candidates_map: Map of candidate IDs to candidate data.
        """
        q_lower = question.lower()
        # Recognize explicit references in the question
        title_keywords = []
        if "po5" in q_lower or "po 5" in q_lower:
            title_keywords.append("po5")
        if "po4" in q_lower or "po 4" in q_lower:
            title_keywords.append("po4")
        if "po6" in q_lower or "po 6" in q_lower:
            title_keywords.append("po6")
        if any(w in q_lower for w in ["verlänger", "abgabefrist", "fristverlängerung"]):
            title_keywords.append("verlängerung")
        if any(
            w in q_lower for w in ["anmeldeformular", "anmeldungsformular", "anmeldung"]
        ):
            title_keywords.append("anmeldung")
        if self._has_auslauf_intent(q_lower):
            if "auslaufordnung" not in title_keywords:
                title_keywords.append("auslaufordnung")
        if "kolloquium" in q_lower:
            title_keywords.append("kolloquium")

        if not title_keywords:
            return

        for chunk in self.chunks:
            if chunk["chunk_id"] in candidates_map:
                continue
            meta = chunk["metadata"]
            if meta.get("faculty") != faculty_filter:
                continue
            if not self._meta_matches_study_program(meta, study_program_filter):
                continue
            title = meta.get("title", "").lower()
            doc_type = meta.get("doc_type", "").lower()
            searchable = title + " " + doc_type
            if any(kw in searchable for kw in title_keywords):
                candidates_map[chunk["chunk_id"]] = {
                    "doc": chunk.get("content", ""),
                    "meta": meta,
                    "origin": "keyword_inject",
                }

    def search(
        self,
        question: str,
        faculty_filter: str,
        top_k: int = 5,
        study_program_filter: Optional[str] = None,
    ) -> Dict[str, List[List[Any]]]:
        """Performs a hybrid search: Vector + BM25, followed by re-ranking.

        Args:
            question: The user question.
            faculty_filter: The faculty to filter by.
            top_k: Number of results to return.
            study_program_filter: The study program to filter by.

        Returns:
            A dictionary containing retrieved documents and their metadata.
        """
        candidates_map: Dict[int, Dict[str, Any]] = {}

        # 1. Semantic Vector Search
        vec_res = self.collection.query(
            query_texts=[question], n_results=30, where={"faculty": faculty_filter}
        )

        if vec_res["ids"] and vec_res["ids"][0]:
            for i, doc_id in enumerate(vec_res["ids"][0]):
                chunk_id = int(doc_id)
                meta = vec_res["metadatas"][0][i]
                if not self._meta_matches_study_program(meta, study_program_filter):
                    continue
                candidates_map[chunk_id] = {
                    "doc": vec_res["documents"][0][i],
                    "meta": meta,
                    "origin": "vector",
                }

        # 2. Keyword Search (BM25)
        tokenized_query = self._tokenize(question)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:30]

        for idx in top_indices:
            chunk = self.chunks[idx]
            if chunk["metadata"].get(
                "faculty"
            ) == faculty_filter and self._meta_matches_study_program(
                chunk["metadata"], study_program_filter
            ):
                if chunk["chunk_id"] not in candidates_map:
                    candidates_map[chunk["chunk_id"]] = {
                        "doc": chunk.get("content", ""),
                        "meta": chunk["metadata"],
                        "origin": "bm25",
                    }

        # 3. Keyword Injection
        self._inject_keyword_matches(
            question, faculty_filter, study_program_filter, candidates_map
        )

        candidates = list(candidates_map.values())
        if not candidates:
            return {"documents": [[]], "metadatas": [[]]}

        # 4. Rule-based priority boosting
        boosted = self._apply_heuristic_boost(question, candidates)
        boosted.sort(key=lambda x: x["priority"])

        # Limit candidates for efficient re-ranking
        to_rank = boosted[:15]

        # 5. CrossEncoder Re-Ranking
        docs_text = [c["doc"] for c in to_rank]
        pairs = [[question, doc] for doc in docs_text]
        scores = self.reranker.predict(pairs)

        # Integrate heuristic boost into final score
        final_ranked = []
        for score, cand in zip(scores, to_rank):
            adjusted_score = float(score) - cand["priority"]
            final_ranked.append((adjusted_score, cand))

        final_ranked.sort(key=lambda x: x[0], reverse=True)

        top_results = [item[1] for item in final_ranked[:top_k]]

        return {
            "documents": [[t["doc"] for t in top_results]],
            "metadatas": [[t["meta"] for t in top_results]],
        }


if __name__ == "__main__":
    engine = HybridRetrievalEngine()
    res = engine.search(
        "Wo finde ich den Antrag auf Zulassung zur Bachelorarbeit?", "F10"
    )

    print("Suchergebnisse (Top 3):")
    for i, meta in enumerate(res["metadatas"][0][:3]):
        titel = meta.get("title", meta["source"])
        print(f"{i+1}. {titel} (Typ: {meta.get('doc_type', 'k.A.')})")
