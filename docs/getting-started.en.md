# Getting Started

This guide will walk you through the process of setting up and using the PO-Chatbot.

## Prerequisites

Ensure that the following software is installed:

*   **Python 3.11+**
*   **Ollama** (for local LLM execution)

## 1. Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## 2. LLM Setup

Download the default model using Ollama:

```bash
ollama pull qwen2.5:14b
```

## 3. Data Ingestion

Before using the chatbot, the documents must be indexed:

```bash
python src/po_chatbot/ingest_data.py
```
This creates a ChromaDB instance in `data/chroma_db`.

## 4. Start Chatbot

Start the student interface:

```bash
python src/po_chatbot/chatbot_student.py
```

Open the displayed URL (default http://127.0.0.1:7860) in your browser.
