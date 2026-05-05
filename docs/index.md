# PO-Chatbot (TH Köln)

![Version](https://img.shields.io/badge/version-0.1.1-blue)
![Doc Coverage](assets/interrogate.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ein RAG-basiertes Chatbot-System zur Beantwortung von Fragen zum Prüfungsrecht an der TH Köln. Entwickelt im Rahmen einer Bachelorarbeit (2026).

## Hauptfunktionen

*   **Hybride Suche**: Kombination aus ChromaDB (Vektorsuche) und BM25 (Schlüsselwortsuche) für höchste Präzision.
*   **Human-in-the-Loop**: Admin-Interface zur Validierung und Korrektur von KI-Antworten.
*   **Lokale Ausführung**: Nutzt Ollama (qwen2.5:14b) für datenschutzkonforme Verarbeitung.
*   **Multilingual Embeddings**: Verwendet `intfloat/multilingual-e5-large` zur semantischen Erfassung.

## Schnelleinstieg

```python
from po_chatbot.retrieval_engine import HybridRetrievalEngine

engine = HybridRetrievalEngine()
res = engine.search("Wie melde ich meine Bachelorarbeit an?", "F10")
print(res['documents'][0][0])
```

## Projektstruktur

*   `src/po_chatbot/`: Kernlogik und UI-Implementierung.
*   `evaluation/`: Framework zur Qualitätsmessung.
*   `docs/`: Diese Dokumentation.
