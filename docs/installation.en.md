# Installation

The PO-Chatbot can be installed in several ways.

## Installation via pip

The easiest way is to install the dependencies directly into a virtual environment:

```bash
# Create virtual environment (example)
# python -m venv venv
# source venv/bin/activate
pip install -r requirements.txt
```

## Developer Installation

For contributing to the project, we recommend installing in editable mode:

```bash
pip install -e .
pip install black ruff pytest interrogate
```

## System Dependencies

### Ollama
The chatbot requires [Ollama](https://ollama.ai/) for local inference.  
1. Install Ollama for your operating system.  
2. Start the service: `ollama serve`.  
3. Download the model: `ollama pull qwen2.5:14b`.  

### Vector Database
ChromaDB is installed as a library and does not require a separate server installation unless you want to use a remote server.
