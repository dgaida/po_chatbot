# Configuration

Configuration is primarily handled via variables in the Python scripts. A `.env` file is also supported.

## LLM Configuration

In `src/po_chatbot/chatbot_student.py` and `evaluation/evaluate_rag.py`:

| Parameter | Default Value | Description |
|---|---|---|
| `MODEL_NAME` | `qwen2.5:14b` | Name of the Ollama model |
| `TEMPERATURE` | `0.1` | Creativity of response generation |
| `TOP_P` | `0.9` | Nucleus sampling parameter |
| `NUM_CTX` | `8192` | Context window size |
| `REPEAT_PENALTY` | `1.1` | Penalty for word repetition |

## Retrieval Configuration

In `src/po_chatbot/retrieval_engine.py`:

| Parameter | Default Value | Description |
|---|---|---|
| `TOP_K_RETRIEVAL` | `5` | Number of documents returned |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Model for vectorization |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Model for cross-encoder re-ranking |

## Environment Variables

If present, the following variables are loaded from a `.env` file:

```env
OLLAMA_URL=http://localhost:11434/api/generate
```
