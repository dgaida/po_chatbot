# Installation

Der PO-Chatbot kann auf verschiedene Arten installiert werden.

## Installation via pip

Der einfachste Weg ist die Installation der Abhängigkeiten direkt in eine virtuelle Umgebung:

```bash
# Virtuelle Umgebung erstellen (Beispiel)
# python -m venv venv
# source venv/bin/activate
pip install -r requirements.txt
```

## Entwickler-Installation

Für die Mitarbeit am Projekt empfehlen wir die Installation im Editier-Modus:

```bash
pip install -e .
pip install black ruff pytest interrogate
```

## Systemabhängigkeiten

### Ollama
Der Chatbot benötigt [Ollama](https://ollama.ai/) für die lokale Inferenz.
1. Installieren Sie Ollama für Ihr Betriebssystem.
2. Starten Sie den Dienst: `ollama serve`.
3. Laden Sie das Modell: `ollama pull qwen2.5:14b`.

### Vektordatenbank
ChromaDB wird als Bibliothek installiert und benötigt keine separate Server-Installation, es sei denn, Sie möchten einen Remote-Server nutzen.
