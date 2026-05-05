# PO-Chatbot (TH Köln)

![Version](https://img.shields.io/badge/version-0.1.1-blue)
![Doc Coverage](assets/interrogate.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A RAG-based chatbot system for answering questions about exam regulations at TH Köln. Developed as part of a bachelor thesis (2026).

## Key Features

*   **Hybrid Search**: Combination of ChromaDB (vector search) and BM25 (keyword search) for maximum precision.
*   **Human-in-the-Loop**: Admin interface for validation and correction of AI responses.
*   **Local Execution**: Uses Ollama (qwen2.5:14b) for privacy-compliant processing.
*   **Multilingual Embeddings**: Utilizes `intfloat/multilingual-e5-large` for semantic understanding.

## Quickstart

```python
from po_chatbot.retrieval_engine import HybridRetrievalEngine

engine = HybridRetrievalEngine()
res = engine.search("How do I register my bachelor thesis?", "F10")
print(res['documents'][0][0])
```

## Project Structure

*   `src/po_chatbot/`: Core logic and UI implementation.
*   `evaluation/`: Framework for quality measurement.
*   `docs/`: This documentation.
