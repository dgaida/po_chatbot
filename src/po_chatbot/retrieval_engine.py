# Hybride Retrieval-Engine für das RAG-System.
# Kombiniert dichte Vektorsuche (ChromaDB) mit dünner Schlüsselwortsuche (BM25)
# und wendet einen Cross-Encoder für Re-Ranking an. Nutzt strikte Metadaten-Filterung.

import os
import json
import re
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi


class HybridRetrievalEngine:
    def __init__(self):
        # Initialisiert die Retrieval-Engine-Komponenten: ChromaDB, BM25 und CrossEncoder.
        self.db_path = os.path.join("data", "chroma_db")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="intfloat/multilingual-e5-large"
        )
        self.collection = self.client.get_collection(
            name="th_koeln_rules", embedding_function=self.emb_func
        )

        self.chunks_path = os.path.join("data", "chunks.json")
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # BM25 läuft auf einem für lexikalisches Matching optimierten Text (Titel/Typ + Inhalt)
        tokenized_corpus = [
            self._tokenize(doc.get("bm25_text", doc.get("content", "")))
            for doc in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def _normalize_study_program(self, s: str) -> str:
        if not s:
            return ""
        s = s.lower().strip()
        s = re.sub(r"\s*\(.*?\)\s*", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _meta_matches_study_program(
        self, meta: dict, study_program_filter: str | None
    ) -> bool:
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

    def _tokenize(self, text):
        # Einfacher Tokenizer für BM25-Verarbeitung.
        return re.findall(r"\w+", text.lower())

    def _has_auslauf_intent(self, q_lower):
        # Erkennt ob die Frage auf eine Auslaufordnung abzielt.
        # Deckt ab: explizites 'auslauf', PO-Wechsel, Modul-Anerkennung/Anrechnung.
        # Direkte Auslauf-Keywords
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
        # PO-Kontext erkennen: 'prüfungsordnung' oder 'po '
        has_po_context = (
            "prüfungsordnung" in q_lower
            or "neue po" in q_lower
            or "neuen po" in q_lower
            or "alten po" in q_lower
            or "alte po" in q_lower
        )
        # Anerkennungs-Keywords (Synonyme für Modulanerkennung)
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
        # Wechsel-Keywords
        switch_kw = any(
            w in q_lower for w in ["wechsl", "wechsel", "umstieg", "umsteig"]
        )
        # Anerkennung/Wechsel + PO-Kontext = Auslaufordnung
        if has_po_context and (recognition_kw or switch_kw):
            return True
        # Eigenständige Anerkennung mit Modul-Kontext (ohne explizite PO-Nennung)
        if recognition_kw and any(w in q_lower for w in ["modul", "leistung"]):
            return True
        return False

    def _apply_heuristic_boost(self, question, candidates):
        # Keyword-intent-basiertes Boosting. Löst das Retrieval-Problem bei
        # semantisch ähnlichen aber inhaltlich verschiedenen Dokumenten
        # (z.B. 'Verlängerung' vs 'Zulassung', 'Prüfungsordnung' vs 'Formular').
        q_lower = question.lower()

        # Absichtserkennung aus der Frage
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

            # Auslaufordnung verstärken, reguläre PO herabstufen
            if auslauf_intent:
                if "auslauf" in title or "auslauf" in doc_type:
                    priority -= 250  # Stärkste Verstärkung
                elif "prüfungsordnung" in doc_type:
                    priority += 30  # Herabstufen: reguläre PO ist nicht Auslaufordnung

            # Verlängerung verstärken, Zulassung/Anmeldung herabstufen
            elif verlaengerung_intent:
                if "verlängerung" in title:
                    priority -= 200  # Starke Verstärkung
                elif any(w in title for w in ["zulassung", "anmeldung"]):
                    priority += 50  # Herabstufen: Zulassung ist nicht Verlängerung

            # Formular/Anmeldung-Erkennung (nur wenn nicht Verlängerung)
            elif form_intent:
                if any(w in doc_type for w in ["formular", "antrag"]) or any(
                    w in title for w in ["formular", "antrag", "anmeldung"]
                ):
                    priority -= 100

            # PO-Erkennung: Prüfungsordnungs-Texte verstärken, Formulare herabstufen
            if po_intent and not auslauf_intent:
                if "prüfungsordnung" in doc_type or "prüfungsordnung" in title:
                    priority -= 150  # Starke Verstärkung für PO-Texte
                elif any(w in doc_type for w in ["formular", "antrag"]):
                    priority += 50  # Herabstufen: Formulare sind keine PO-Texte

            cand["priority"] = priority
            boosted_candidates.append(cand)

        return boosted_candidates

    def _inject_keyword_matches(
        self, question, faculty_filter, study_program_filter, candidates_map
    ):
        # Injiziert Chunks deren Titel explizit genannte Begriffe enthalten.
        # Löst das Problem, dass semantisch ähnliche aber inhaltlich verschiedene
        # Dokumente (z.B. PO4 vs PO5) vom Vektor/BM25-Retrieval nicht beide gefunden werden.
        q_lower = question.lower()
        # Erkenne explizite Referenzen in der Frage
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

    def search(self, question, faculty_filter, top_k=5, study_program_filter=None):
        # Führt eine hybride Suche durch: Vektor + BM25, gefolgt von Re-Ranking.
        candidates_map = {}

        # 1. Semantische Vektorsuche (breiterer Pool)
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

        # 2. Dünne Schlüsselwortsuche (BM25)
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

        # 3. Schlüsselwort-Einspeisung: Explizit referenzierte Dokumente erzwingen
        self._inject_keyword_matches(
            question, faculty_filter, study_program_filter, candidates_map
        )

        candidates = list(candidates_map.values())
        if not candidates:
            return {"documents": [[]], "metadatas": [[]]}

        # 4. Regelbasiertes Prioritäts-Boosting anwenden
        boosted = self._apply_heuristic_boost(question, candidates)
        boosted.sort(key=lambda x: x["priority"])

        # Kandidaten begrenzen für effizientes Re-Ranking
        to_rank = boosted[:15]

        # 5. CrossEncoder-Neugewichtung
        docs_text = [c["doc"] for c in to_rank]
        pairs = [[question, doc] for doc in docs_text]
        scores = self.reranker.predict(pairs)

        # Integration der heuristischen Verstärkung in den finalen Neugewichtungs-Score
        final_ranked = []
        for score, cand in zip(scores, to_rank):
            # Der Neugewichtungs-Score (meist -10 bis +10) wird mit der manuellen Verstärkung kombiniert.
            adjusted_score = score - cand["priority"]
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
        # Zeigt den sauberen Titel an (fällt auf die Quelle/URL zurück, falls der Titel fehlt)
        titel = meta.get("title", meta["source"])
        print(f"{i+1}. {titel} (Typ: {meta.get('doc_type', 'k.A.')})")
