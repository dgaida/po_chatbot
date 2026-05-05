# Prüfungsamt-Dashboard:
# Zeigt wartende Anfragen aus der Student-App und erlaubt Freigabe/Ablehnung.

import os
import json
from datetime import datetime
import gradio as gr

# Dateipfade (geteilt mit chatbot_student.py)
PENDING_QUEUE = os.path.join("data", "evaluation_logs", "hil_pending.jsonl")
FEEDBACK_LOG = os.path.join("data", "evaluation_logs", "hil_feedback.jsonl")


def load_pending():
    # Lade alle wartenden Anfragen.
    if not os.path.exists(PENDING_QUEUE):
        return []
    entries = []
    with open(PENDING_QUEUE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if entry.get("status") == "pending":
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
    return entries


def get_queue_display():
    # Formatierte Übersicht aller wartenden Anfragen.
    entries = load_pending()
    if not entries:
        return "Keine wartenden Anfragen.", [], gr.update(choices=[]), ""

    choices = []
    overview_lines = []
    for i, e in enumerate(entries):
        ts = e.get("timestamp", "?")[:19].replace("T", " ")
        fac = e.get("faculty", "?")
        q = e.get("question", "?")[:80]
        e.get("id", str(i))
        label = f"[{ts}] {fac} — {q}"
        choices.append(label)
        overview_lines.append(f"**{i+1}.** {label}")

    overview = f"### {len(entries)} wartende Anfrage(n)\n\n" + "\n\n".join(overview_lines)
    return overview, entries, gr.update(choices=choices, value=choices[0] if choices else None), ""


def show_detail(selected, entries_state):
    # Zeige Details einer ausgewählten Anfrage.
    if not selected or not entries_state:
        return "", "", "", ""

    idx = None
    for i, e in enumerate(entries_state):
        ts = e.get("timestamp", "?")[:19].replace("T", " ")
        fac = e.get("faculty", "?")
        q = e.get("question", "?")[:80]
        label = f"[{ts}] {fac} — {q}"
        if label == selected:
            idx = i
            break

    if idx is None:
        return "", "", "", ""

    entry = entries_state[idx]
    question = f"**Fakultät:** {entry.get('faculty', '?')}\n\n**Frage:**\n{entry.get('question', '?')}"
    answer = entry.get("answer", "Keine Antwort")
    sources = "\n".join([f"- {s}" for s in entry.get("sources", [])])
    meta = (f"⏱ Retrieval: {entry.get('retrieval_time', '?')}s | "
            f"Generierung: {entry.get('generation_time', '?')}s | "
            f"ID: {entry.get('id', '?')}")

    return question, answer, sources, meta


def submit_feedback(selected, entries_state, action, correction_text):
    # Feedback speichern und Anfrage aus der Queue entfernen.
    if not selected or not entries_state:
        return "Keine Anfrage ausgewählt.", *get_queue_display()[:3], ""

    # Eintrag suchen
    target = None
    for e in entries_state:
        ts = e.get("timestamp", "?")[:19].replace("T", " ")
        fac = e.get("faculty", "?")
        q = e.get("question", "?")[:80]
        label = f"[{ts}] {fac} — {q}"
        if label == selected:
            target = e
            break

    if not target:
        return "Anfrage nicht gefunden.", *get_queue_display()[:3], ""

    # Feedback protokollieren
    feedback = {
        **target,
        "status": action,
        "feedback_time": datetime.now().isoformat(),
        "correction": correction_text if action == "rejected" else "",
    }
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback, ensure_ascii=False) + "\n")

    # Aus der Warteschlange entfernen
    remaining = []
    if os.path.exists(PENDING_QUEUE):
        with open(PENDING_QUEUE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("id") != target.get("id"):
                        remaining.append(line)
                except json.JSONDecodeError:
                    remaining.append(line)
        with open(PENDING_QUEUE, "w", encoding="utf-8") as f:
            f.write("\n".join(remaining) + ("\n" if remaining else ""))

    action_text = "freigegeben ✅" if action == "approved" else "abgelehnt ❌"
    status_msg = f"Anfrage {action_text}. Feedback gespeichert."

    # Warteschlange aktualisieren
    overview, entries, dropdown, _ = get_queue_display()
    return status_msg, overview, entries, dropdown


def approve(selected, entries_state, correction):
    return submit_feedback(selected, entries_state, "approved", correction)


def reject(selected, entries_state, correction):
    return submit_feedback(selected, entries_state, "rejected", correction)


# Gradio UI — Prüfungsamt-Dashboard
with gr.Blocks(title="Prüfungsamt — Human-in-the-Loop Dashboard") as demo:

    gr.HTML("""
    <div style="text-align:center; padding:20px 20px 10px;">
        <h1 style="color:#0b3d91;">Prüfungsamt — Qualitätskontrolle</h1>
        <p style="color:#666; font-size:14px;">
            Human-in-the-Loop Dashboard: Prüfen und geben Sie KI-generierte Antworten frei.
        </p>
    </div>
    """)

    entries_state = gr.State([])

    with gr.Row():
        refresh_btn = gr.Button("🔄 Queue aktualisieren", variant="secondary")

    queue_overview = gr.Markdown("Klicken Sie auf 'Queue aktualisieren'.")

    with gr.Row():
        with gr.Column(scale=1):
            request_dropdown = gr.Dropdown(label="Anfrage auswählen", choices=[], interactive=True)
        with gr.Column(scale=1):
            meta_display = gr.Textbox(label="Metadaten", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            question_display = gr.Markdown(label="Frage")
        with gr.Column(scale=2):
            answer_display = gr.Textbox(label="KI-generierte Antwort", lines=12, interactive=False)

    sources_display = gr.Textbox(label="Verwendete Quellen", lines=3, interactive=False)

    gr.HTML("<hr>")
    gr.HTML("<h3 style='color:#0b3d91;'>Bewertung</h3>")

    correction_input = gr.Textbox(
        label="Korrekturhinweis (optional, bei Ablehnung)",
        placeholder="z.B. 'Frist ist 4 Wochen, nicht 8 Wochen'",
        lines=2,
    )

    with gr.Row():
        approve_btn = gr.Button("✅ Antwort freigeben", variant="primary")
        reject_btn = gr.Button("❌ Antwort ablehnen", variant="stop")

    feedback_status = gr.Textbox(label="Status", interactive=False)

    # Ereignisse
    refresh_btn.click(
        fn=get_queue_display,
        outputs=[queue_overview, entries_state, request_dropdown, feedback_status],
    )

    request_dropdown.change(
        fn=show_detail,
        inputs=[request_dropdown, entries_state],
        outputs=[question_display, answer_display, sources_display, meta_display],
    )

    approve_btn.click(
        fn=approve,
        inputs=[request_dropdown, entries_state, correction_input],
        outputs=[feedback_status, queue_overview, entries_state, request_dropdown],
    )

    reject_btn.click(
        fn=reject,
        inputs=[request_dropdown, entries_state, correction_input],
        outputs=[feedback_status, queue_overview, entries_state, request_dropdown],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7861)
