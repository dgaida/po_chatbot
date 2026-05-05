# PO-Chatbot Agent Instructions

This repository contains a RAG-based chatbot for exam regulations (Prüfungsrecht) at TH Köln, originating from a 2026 Bachelor thesis by Nikita B.

## Tech Stack  
- **Python**: Core logic and UI.  
- **LangChain**: RAG pipeline orchestration.  
- **ChromaDB**: Semantic search / Vector database.  
- **BM25**: Keyword search.  
- **Sentence Transformers**: 'intfloat/multilingual-e5-large' for embeddings.  
- **Gradio**: Web interface for students and admins.  
- **Ollama**: Local LLM execution ('qwen2.5:14b').  

## Project Structure  
- `src/po_chatbot/`: Core chatbot implementation.  
  - `chatbot_student.py`: Student-facing UI.  
  - `chatbot_admin.py`: Admin-facing UI for HIL (Human-in-the-Loop) validation.  
  - `ingest_data.py`: Data ingestion pipeline (Markdown with YAML metadata).  
  - `retrieval_engine.py`: Hybrid retrieval logic (ChromaDB + BM25 + Re-ranking).  
- `evaluation/`: Performance and quality evaluation scripts.  

## Coding Guidelines  
- **LLM Client**: Always prefer using `dgaida/llm_client` as a unified interface for LLM calls (local or cloud).  
- **Versioning**: Integrated with `dgaida/auto-version-action` for automated version bumping and badge generation.  
- **Code Style**: Use `black` for formatting and `ruff` for linting.  

## Development Setup  
1. Install dependencies: `pip install -r requirements.txt`  
2. Ensure Ollama is running with `qwen2.5:14b`.  
3. Run `src/po_chatbot/ingest_data.py` to index documentation.  
4. Run `src/po_chatbot/chatbot_student.py` to start the UI.  
