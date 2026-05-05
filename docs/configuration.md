# Konfiguration

Die Konfiguration erfolgt hauptsächlich über Variablen in den Python-Skripten. Eine `.env`-Datei wird ebenfalls unterstützt.

## LLM-Konfiguration

In `src/po_chatbot/chatbot_student.py` und `evaluation/evaluate_rag.py`:

| Parameter | Standardwert | Beschreibung |
|---|---|---|
| `MODEL_NAME` | `qwen2.5:14b` | Name des Ollama-Modells |
| `TEMPERATURE` | `0.1` | Kreativität der Antwortgenerierung |
| `TOP_P` | `0.9` | Nucleus Sampling Parameter |
| `NUM_CTX` | `8192` | Kontextfenster-Größe |
| `REPEAT_PENALTY` | `1.1` | Bestrafung für Wortwiederholungen |

## Retrieval-Konfiguration

In `src/po_chatbot/retrieval_engine.py`:

| Parameter | Standardwert | Beschreibung |
|---|---|---|
| `TOP_K_RETRIEVAL` | `5` | Anzahl der zurückgegebenen Dokumente |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Modell für die Vektorisierung |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Modell für das Cross-Encoder Re-Ranking |

## Umgebungsvariablen

Falls vorhanden, werden folgende Variablen aus einer `.env` Datei geladen:

```env
OLLAMA_URL=http://localhost:11434/api/generate
```
