# Fehlerbehebung

Häufige Probleme und deren Lösungen.

## Ollama Verbindungsprobleme

**Fehler**: `Fehler: Ollama ist nicht erreichbar.`

*   **Lösung**: Stellen Sie sicher, dass Ollama im Hintergrund läuft (`ollama serve`). Prüfen Sie den Status mit `curl http://localhost:11434/api/tags`.
*   **Lösung**: Falls Ollama auf einem anderen Port oder Rechner läuft, passen Sie die `OLLAMA_URL` in den Skripten an.

## Keine Dokumente gefunden

**Fehler**: `Es konnten keine relevanten Dokumente gefunden werden.`

*   **Ursache**: Die Datenbank wurde noch nicht initialisiert.
*   **Lösung**: Führen Sie `python src/po_chatbot/ingest_data.py` aus. Prüfen Sie, ob in `data/text_extracted` Markdown-Dateien vorhanden sind.
*   **Ursache**: Der Fakultäts- oder Studiengangs-Filter ist zu restriktiv.
*   **Lösung**: Versuchen Sie es mit einer allgemeineren Einstellung.

## Halluzinationen in den Antworten

**Problem**: Der Chatbot erfindet Fakten oder nennt falsche Fristen.

*   **Lösung**: Erhöhen Sie die `REPEAT_PENALTY` oder senken Sie die `TEMPERATURE`.
*   **Lösung**: Überprüfen Sie die Qualität der extrahierten Texte in `data/chunks.json`. Nutzen Sie das Admin-Interface zur Korrektur.

## Langsame Antwortgenerierung

**Problem**: Die Antwort dauert mehrere Minuten.

*   **Lösung**: Der PO-Chatbot läuft lokal auf Ihrer CPU/GPU. Ein System mit mindestens 16GB RAM und einer dedizierten GPU wird für das 14B-Modell empfohlen. Verwenden Sie ggf. ein kleineres Modell (z.B. `qwen2.5:7b`).
