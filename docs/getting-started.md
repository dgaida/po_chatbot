# Erste Schritte

Diese Anleitung führt Sie durch den Prozess der Einrichtung und Nutzung des PO-Chatbots.

## Voraussetzungen

Stellen Sie sicher, dass die folgende Software installiert ist:

*   **Python 3.11+**
*   **Ollama** (für die lokale Ausführung des LLM)

## 1. Installation

Klonen Sie das Repository und installieren Sie die Abhängigkeiten:

```bash
pip install -r requirements.txt
```

## 2. LLM-Setup

Laden Sie das Standardmodell mit Ollama herunter:

```bash
ollama pull qwen2.5:14b
```

## 3. Daten-Ingestion

Bevor Sie den Chatbot nutzen können, müssen die Dokumente indexiert werden:

```bash
python src/po_chatbot/ingest_data.py
```
Dies erstellt eine ChromaDB-Instanz in `data/chroma_db`.

## 4. Chatbot starten

Starten Sie das Studenten-Interface:

```bash
python src/po_chatbot/chatbot_student.py
```

Öffnen Sie die angezeigte URL (standardmäßig http://127.0.0.1:7860) in Ihrem Browser.
