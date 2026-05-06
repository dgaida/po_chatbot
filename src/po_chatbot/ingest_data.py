"""Ingestion pipeline for the RAG system.

Reads processed Markdown files, extracts YAML metadata, performs semantic chunking,
and populates both the ChromaDB vector store and a JSON file for BM25 search.
"""

import os
import re
from typing import Any, Dict, Tuple

import chromadb
import json
import yaml
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR: str = os.path.join("data", "text_extracted")
DB_PATH: str = os.path.join("data", "chroma_db")
CHUNKS_JSON_PATH: str = os.path.join("data", "chunks.json")
COLLECTION_NAME: str = "th_koeln_rules"
EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"


def parse_yaml_frontmatter(filepath: str) -> Tuple[Dict[str, Any], str]:
    """Extracts YAML metadata and main text from a Markdown file.

    Args:
        filepath: Path to the Markdown file.

    Returns:
        A tuple containing the metadata dictionary and the body text.
    """
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    match = re.search(
        r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL | re.MULTILINE
    )

    if match:
        yaml_text = match.group(1)
        body_text = match.group(2)
        try:
            metadata = yaml.safe_load(yaml_text)
            if not metadata:
                metadata = {}
        except Exception as e:
            print(f"Warning: YAML parsing error in {filepath}: {e}")
            metadata = {}
        return metadata, body_text
    else:
        print(f"Warning: No YAML frontmatter found in {filepath}")
        return {}, content


def ingest_data() -> None:
    """Processes documents, generates chunks with metadata, and populates the database."""
    if not os.path.exists(DOCS_DIR):
        print(f"Error: Directory {DOCS_DIR} does not exist.")
        return

    # Semantic splitter with priority on structural Markdown elements and paragraphs
    separators = ["\n# ", "\n## ", "\n### ", "\n§ ", "\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=250, separators=separators, keep_separator=True
    )

    all_chunks = []
    total_chunks = 0
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]
    print(f"Processing {len(files)} files...")

    for filename in files:
        filepath = os.path.join(DOCS_DIR, filename)
        metadata, body = parse_yaml_frontmatter(filepath)

        # Clean metadata for ChromaDB compatibility
        safe_meta = {"source": filename}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                safe_meta[k] = v
            elif isinstance(v, list):
                safe_meta[k] = ", ".join(v)  # type: ignore
            else:
                safe_meta[k] = str(v)

        fac = safe_meta.get("faculty", "N/A")
        print(f"Reading: {filename} (Faculty: {fac})")

        raw_chunks = splitter.create_documents([body])

        for chunk in raw_chunks:
            title = safe_meta.get("title", filename)
            doc_type = safe_meta.get("doc_type", "")

            content_text = chunk.page_content.strip()

            # For BM25 it's advantageous to include title/type
            bm25_text = "\n".join(
                [
                    p
                    for p in [str(title).strip(), str(doc_type).strip(), content_text]
                    if p
                ]
            )

            all_chunks.append(
                {
                    "chunk_id": total_chunks,
                    "content": content_text,
                    "bm25_text": bm25_text,
                    "metadata": safe_meta,
                }
            )
            total_chunks += 1

    # Save chunks for hybrid BM25 search
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"{total_chunks} chunks generated and saved to {CHUNKS_JSON_PATH}.")

    print("Initializing vectorization (ChromaDB)...")
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:  # nosec
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=emb_func
    )

    documents = [c["content"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [str(c["chunk_id"]) for c in all_chunks]

    batch_size = 100
    total_batches = (len(documents) // batch_size) + 1

    for i in range(0, len(documents), batch_size):
        end = i + batch_size
        collection.add(
            documents=documents[i:end], metadatas=metadatas[i:end], ids=ids[i:end]
        )
        if (i // batch_size + 1) <= total_batches:
            print(f"Batch {i // batch_size + 1}/{total_batches} inserted.")

    print("Database successfully built.")


if __name__ == "__main__":
    ingest_data()
