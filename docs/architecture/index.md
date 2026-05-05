# Architektur

Der PO-Chatbot folgt einer klassischen Retrieval-Augmented Generation (RAG) Architektur, erweitert um eine hybride Suche und eine Human-in-the-Loop Komponente.

## Systemübersicht

Die folgende Grafik zeigt die Interaktion der Komponenten:

```mermaid
graph TD
    User((Student/Admin)) --> UI[Gradio UI]
    UI --> Engine[Hybrid Retrieval Engine]
    Engine --> Chroma[(ChromaDB)]
    Engine --> BM25[BM25 Index]
    Engine --> Reranker[Cross-Encoder Reranker]
    UI --> LLM[Ollama / LLM]
    LLM -.-> History[(Log Files)]
```

## Datenfluss

Der Datenfluss einer Anfrage sieht wie folgt aus:

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Gradio Interface
    participant E as Retrieval Engine
    participant DB as VectorDB / BM25
    participant L as LLM (Ollama)

    U->>UI: Stellt Frage (mit Filter)
    UI->>E: Suchanfrage
    E->>DB: Vektor- & Schlüsselwortsuche
    DB-->>E: Top-K Chunks
    E->>E: Re-Ranking & Boosting
    E-->>UI: Relevanter Kontext
    UI->>L: Prompt (Kontext + Frage)
    L-->>UI: Antworttext
    UI-->>U: Antwort + Quellenlinks
```

## Lifecycle-Prozess

Der Prozess der Datenaufbereitung und Validierung:

```mermaid
graph LR
    MD[Markdown Docs] --> Ingest[Ingestion Script]
    Ingest --> Chunks[JSON Chunks]
    Ingest --> DB[(ChromaDB)]

    U[User Query] --> RAG[RAG Pipeline]
    RAG --> Answer[AI Answer]
    Answer --> HiL{Admin Check}
    HiL -->|Korrekt| Valid[Validierte Antwort]
    HiL -->|Falsch| Correct[Manuelle Korrektur]
    Correct --> Valid
```
