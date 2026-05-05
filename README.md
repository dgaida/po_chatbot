# Entwicklung eines RAG-basierten Chatbots zur Beantwortung prüfungsrechtlicher Fragen

Dieses Projekt implementiert ein Retrieval-Augmented Generation (RAG) System zur Beantwortung von Fragen rund um das Prüfungsrecht an der TH Köln. Es kombiniert semantische Suche (Vektordatenbank) mit Schlüsselwortsuche (BM25), um präzise Informationen aus Prüfungsordnungen zu extrahieren und verständlich aufzubereiten.

## Features

- **Hybrid Search Engine**: Kombiniert ChromaDB (Vektorsuche) und BM25 (Textsuche) für optimale Retrieval-Ergebnisse.
- **Studenten-Interface**: Eine Gradio-basierte Weboberfläche für Studierende zur Abfrage von Informationen.
- **Admin-Dashboard**: Eine Oberfläche für das Prüfungsamt zur Validierung und Freigabe von KI-generierten Antworten (Human-in-the-Loop).
- **Automatisierte Ingestion**: Pipeline zur Verarbeitung von Markdown-Dokumenten mit YAML-Metadaten.
- **Evaluation**: Umfangreiches Framework zur Bewertung der Retrieval-Qualität und Antwortgenerierung.

## Projektstruktur

- `src/po_chatbot/`: Kern-Logik des Chatbots.
  - `ingest_data.py`: Indexierung der Dokumente.
  - `retrieval_engine.py`: Hybrid-Search Logik.
  - `chatbot_student.py`: UI für Studierende.
  - `chatbot_admin.py`: Dashboard für das Prüfungsamt.
- `evaluation/`: Skripte zur Analyse und Messung der Performance.
- `data/`: (Nicht im Repo) Verzeichnis für Dokumente und Datenbanken.

## Installation & Setup

1. **Abhängigkeiten installieren**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Hinweis: Erstellen Sie ggf. eine requirements.txt basierend auf den Importen)*

2. **Ollama bereitstellen**:
   Das System nutzt standardmäßig `qwen2.5:14b` über Ollama.

3. **Daten indexieren**:
   ```bash
   python src/po_chatbot/ingest_data.py
   ```

4. **Chatbot starten**:
   ```bash
   python src/po_chatbot/chatbot_student.py
   ```

## Acknowledgments

Dieses Repository ist im Rahmen einer Bachelorarbeit im Jahr 2026 an der **Technischen Hochschule Köln (TH Köln)** entstanden.

**Titel:** Entwicklung eines RAG-basierten Chatbots zur Beantwortung prüfungsrechtlicher Fragen
**Autor:** Nikita B.
