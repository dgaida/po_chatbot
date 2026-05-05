# Architecture

The PO-Chatbot follows a classic Retrieval-Augmented Generation (RAG) architecture, enhanced with hybrid search and a human-in-the-loop component.

## System Overview

The following diagram shows the interaction of the components:

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

## Data Flow

The data flow of a request is as follows:

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Gradio Interface
    participant E as Retrieval Engine
    participant DB as VectorDB / BM25
    participant L as LLM (Ollama)

    U->>UI: Asks question (with filter)
    UI->>E: Search request
    E->>DB: Vector & keyword search
    DB-->>E: Top-K Chunks
    E->>E: Re-ranking & boosting
    E-->>UI: Relevant context
    UI->>L: Prompt (context + question)
    L-->>UI: Response text
    UI-->>U: Response + source links
```

## Lifecycle Process

The process of data preparation and validation:

```mermaid
graph LR
    MD[Markdown Docs] --> Ingest[Ingestion Script]
    Ingest --> Chunks[JSON Chunks]
    Ingest --> DB[(ChromaDB)]

    U[User Query] --> RAG[RAG Pipeline]
    RAG --> Answer[AI Answer]
    Answer --> HiL{Admin Check}
    HiL -->|Correct| Valid[Validated Answer]
    HiL -->|Incorrect| Correct[Manual Correction]
    Correct --> Valid
```
