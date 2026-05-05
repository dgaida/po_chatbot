# Ingestion-Pipeline für das RAG-System.
# Liest aufbereitete Markdown-Dateien, extrahiert YAML-Metadaten, führt semantisches Chunking durch
# und befüllt sowohl den ChromaDB-Vektorspeicher als auch eine JSON-Datei für die BM25-Schlüsselwortsuche.

import os
import re
import yaml
import json
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = os.path.join("data", "text_extracted")
DB_PATH = os.path.join("data", "chroma_db")
CHUNKS_JSON_PATH = os.path.join("data", "chunks.json")
COLLECTION_NAME = "th_koeln_rules"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


def parse_yaml_frontmatter(filepath):
    # Extrahiert YAML-Metadaten und den Haupttext aus einer Markdown-Datei.
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
            print(f"Warnung: YAML-Parsing-Fehler in {filepath}: {e}")
            metadata = {}
        return metadata, body_text
    else:
        print(f"Warnung: Kein YAML-Frontmatter in {filepath} gefunden")
        return {}, content


def ingest_data():
    # Verarbeitet Dokumente, erzeugt Chunks mit Metadaten und befüllt die Vektordatenbank.
    if not os.path.exists(DOCS_DIR):
        print(f"Fehler: Verzeichnis {DOCS_DIR} existiert nicht.")
        return

    # Semantische Splitter mit Priorität auf strukturelle Markdown-Elemente und Absätze
    separators = ["\n# ", "\n## ", "\n### ", "\n§ ", "\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=250, separators=separators, keep_separator=True
    )

    all_chunks = []
    total_chunks = 0
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]
    print(f"Verarbeite {len(files)} Dateien...")

    for filename in files:
        filepath = os.path.join(DOCS_DIR, filename)
        metadata, body = parse_yaml_frontmatter(filepath)

        # Metadaten bereinigen für ChromaDB-Kompatibilität
        safe_meta = {"source": filename}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                safe_meta[k] = v
            elif isinstance(v, list):
                safe_meta[k] = ", ".join(v)
            else:
                safe_meta[k] = str(v)

        fac = safe_meta.get("faculty", "N/A")
        print(f"Lese: {filename} (Fakultät: {fac})")

        raw_chunks = splitter.create_documents([body])

        for chunk in raw_chunks:
            title = safe_meta.get("title", filename)
            safe_meta.get("study_program", "Allgemein")
            doc_type = safe_meta.get("doc_type", "")

            # Embeddings so sauber wie möglich halten (nur Inhalt),
            # Metadaten strukturiert belassen, Kontext erst beim Prompt-Bau formatieren.
            content_text = chunk.page_content.strip()

            # Für BM25 ist es vorteilhaft Titel/Typ einzubeziehen, aber strukturierte Labels nicht wiederholen.
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

    # Chunks für hybride BM25-Suche speichern
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"{total_chunks} Chunks erzeugt und gespeichert unter {CHUNKS_JSON_PATH}.")

    print("Initialisiere Vektorisierung (ChromaDB)...")
    client = chromadb.PersistentClient(path=DB_PATH)
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:  # nosec  # nosec
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=emb_func
    )

    # Embeddings werden auf sauberem Chunk-Inhalt berechnet.
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
            print(f"Batch {i // batch_size + 1}/{total_batches} eingefügt.")

    print("Datenbank erfolgreich aufgebaut.")


if __name__ == "__main__":
    ingest_data()
