# Nutzung

Der PO-Chatbot bietet zwei Haupt-Interfaces für unterschiedliche Benutzergruppen.

## Studenten-Interface (`chatbot_student.py`)

Dieses Interface ist für die tägliche Nutzung durch Studierende gedacht.

*   **Frage stellen**: Geben Sie Ihre Frage in das Textfeld ein.
*   **Filter**: Wählen Sie Ihre Fakultät und Ihren Studiengang aus, um die Suche zu präzisieren.
*   **Antwort & Quellen**: Der Bot generiert eine Antwort und listet die verwendeten Dokumente mit Links auf.

## Admin-Interface (`chatbot_admin.py`)

Das Admin-Interface dient der Qualitätssicherung (Human-in-the-Loop).

1.  **Pending-Queue**: Hier werden alle Anfragen gelistet, die noch nicht validiert wurden.
2.  **Detailansicht**: Admins sehen die Frage, die KI-Antwort und den abgerufenen Kontext.
3.  **Aktionen**:
    *   **Approve**: Die Antwort ist korrekt.
    *   **Edit & Approve**: Korrigieren Sie die Antwort manuell vor der Freigabe.
    *   **Reject**: Die Antwort ist falsch oder irreführend.

## Evaluierungsskripte

Das Verzeichnis `evaluation/` enthält Skripte zur automatisierten Messung der Performance:

*   `evaluate_rag.py`: Führt Testfragen gegen verschiedene Modelle/Configs aus.
*   `analyze_all_phases.py`: Aggregiert die Ergebnisse der verschiedenen Testphasen.
