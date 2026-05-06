# Bereitstellung auf Render.com

Diese Anleitung beschreibt, wie Sie den PO-Chatbot auf [Render.com](https://render.com) bereitstellen können.

## Voraussetzungen

- Ein GitHub-Account mit dem Repository des Chatbots.  
- Ein Render.com Account.  
- Ein extern erreichbarer Ollama-Endpunkt (da Render keine GPUs für kostenlose Instanzen bietet und lokale LLMs dort zu langsam wären oder den RAM sprengen würden).  

## Blaupause (Blueprint)

Das Repository enthält eine `render.yaml`, die als Blaupause dient. Sie definiert zwei Web-Services:

1.  **po-chatbot-student**: Das Interface für Studierende.  
2.  **po-chatbot-admin**: Das Dashboard für das Prüfungsamt.  

## Schritte zur Bereitstellung

1.  Loggen Sie sich bei Render.com ein.  
2.  Klicken Sie auf **"New +"** und wählen Sie **"Blueprint"**.  
3.  Verbinden Sie Ihr GitHub-Repository.  
4.  Render erkennt die `render.yaml` automatisch und schlägt die Erstellung der Services vor.  
5.  Geben Sie die benötigten Umgebungsvariablen an (siehe unten).  
6.  Klicken Sie auf **"Apply"**.  

## Umgebungsvariablen

Folgende Variablen müssen oder können konfiguriert werden:

| Variable | Beschreibung | Standardwert |
| :--- | :--- | :--- |
| `OLLAMA_URL` | Die URL zu Ihrem Ollama-Endpunkt. | (Platzhalter) |
| `MODEL_NAME` | Das zu verwendende Modell. | `qwen2.5:14b` |
| `TEMPERATURE` | Kreativität der Generierung. | `0.0` |
| `MAX_CONCURRENT` | Maximale gleichzeitige Anfragen. | `1` |

## Datenpersistenz

Da Render-Webservices ein ephämeres Dateisystem haben, gehen Änderungen am `data/`-Verzeichnis (z.B. Logs oder neu indexierte Daten) nach einem Neustart verloren, sofern kein **Render Disk** eingebunden wird.

Für eine produktive Nutzung wird empfohlen:  
- Ein **Render Blueprint Disk** in der `render.yaml` hinzuzufügen.  
- Oder die Daten vor der Bereitstellung lokal zu indexieren und in das Repository zu committen (nicht empfohlen für große Datenbanken).  

## Ingestion auf Render

Wenn Sie Daten auf Render indexieren möchten, können Sie eine **Shell** in der Render-Konsole öffnen und ausführen:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/po_chatbot
python src/po_chatbot/ingest_data.py
```
