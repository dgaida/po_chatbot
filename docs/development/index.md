# Entwicklung

Dieses Dokument beschreibt die Richtlinien für die Mitarbeit am PO-Chatbot.

## Lokales Setup

1. Installieren Sie die Entwickler-Abhängigkeiten:
   ```bash
   pip install -e .
   pip install black ruff pytest interrogate
   ```
2. Aktivieren Sie die Git-Hooks (falls vorhanden) oder führen Sie die Checks manuell aus.

## Code-Stil

Wir verwenden **Black** für die Formatierung und **Ruff** für das Linting.
- Formatierung: `black src evaluation`
- Linting: `ruff check src evaluation`

## Dokumentation

- Verwenden Sie Google-Style Docstrings für alle öffentlichen Funktionen und Klassen.
- Die Dokumentation wird mit MkDocs erstellt. Führen Sie `mkdocs serve` aus, um eine Vorschau lokal anzuzeigen.

## Tests

Bisherige Tests befinden sich im `evaluation/` Verzeichnis. Neue Unit-Tests sollten im Verzeichnis `tests/` (noch zu erstellen) hinzugefügt werden.
