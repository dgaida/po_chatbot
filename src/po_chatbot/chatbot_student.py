"""Student Chatbot: Gradio interface for examination law queries.

Provides a user interface for students to ask questions about examination regulations,
deadlines, and forms. Uses a RAG pipeline with Ollama and Hybrid Retrieval.
"""

import os
import json
import time
import threading
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
from dotenv import load_dotenv

from retrieval_engine import HybridRetrievalEngine

# Load environment variables
load_dotenv()

MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen2.5:14b")
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
REPEAT_PENALTY: float = float(os.getenv("REPEAT_PENALTY", "1.0"))
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
NUM_PREDICT: int = int(os.getenv("NUM_PREDICT", "1024"))
NUM_CTX: int = int(os.getenv("NUM_CTX", "4096"))
TOP_P: float = float(os.getenv("TOP_P", "0.85"))

# Logging and HIL paths
CHAT_LOG: str = os.getenv(
    "CHAT_LOG", os.path.join("data", "evaluation_logs", "chat_log.jsonl")
)
PENDING_QUEUE: str = os.getenv(
    "PENDING_QUEUE", os.path.join("data", "evaluation_logs", "hil_pending.jsonl")
)

# Study programs per faculty (from YAML metadata of documents)
STUDY_PROGRAMS: Dict[str, List[str]] = {
    "F10": [
        "Medieninformatik",
        "Code & Context",
        "Informatik",
        "IT-Management (Informatik)",
        "Wirtschaftsinformatik",
    ],
    "F04": [
        "Betriebswirtschaftslehre",
        "Wirtschaftsrecht",
        "International Business",
        "Banking and Finance",
        "Finance and Capital Markets",
        "Risk and Insurance",
    ],
    "F04 / F08": [
        "Logistik",
        "Supply Chain and Operations Management",
    ],
}

# Mapping: UI faculty -> ChromaDB faculty filter
FACULTY_FILTER_MAP: Dict[str, str] = {
    "F10": "F10",
    "F04": "F04",
    "F04 / F08": "F04, F08",
}

MAX_CONCURRENT: int = int(os.getenv("MAX_CONCURRENT", "1"))
_semaphore: threading.Semaphore = threading.Semaphore(MAX_CONCURRENT)
_queue_lock: threading.Lock = threading.Lock()
_queue_count: int = 0
_avg_gen_time: float = 10.0

SYSTEM_PROMPT: str = """Sie sind ein präziser Studienberater-Assistent der TH Köln.
Beantworten Sie die Fragen der Studierenden AUSSCHLIESSLICH basierend auf dem bereitgestellten Kontext. Erfinden Sie keine Fakten. Wenn keine Antwort im Kontext steht, sagen Sie: "Dazu liegen mir keine Informationen vor."

WICHTIG: Nennen Sie NIEMALS Dokument-Nummern wie "Dokument 1", "Quelle 3" oder "laut Dokument 5" im Fließtext. Nutzen Sie ausschließlich die Informationen aus den Texten selbst.
Erstellen Sie KEINE eigenen Links oder Markdown-Links wie [Text](). Verwenden Sie NUR die exakten URLs aus dem Kontext.

BEISPIEL FÜR EINE PERFEKTE ANTWORT:
Sie finden den Antrag auf Zulassung im entsprechenden Formular des Prüfungsservice. Reichen Sie dieses rechtzeitig ein.
Link zum Dokument: https://www.th-koeln.de/beispiel_link.pdf
"""

engine: Optional[HybridRetrievalEngine] = None


def init_engine() -> None:
    """Lazy initialization of the Retrieval Engine."""
    global engine
    if engine is None:
        engine = HybridRetrievalEngine()


def log_json(filepath: str, entry: Dict[str, Any]) -> None:
    """Logs a dictionary as a JSON line to a file.

    Args:
        filepath: Path to the log file.
        entry: Dictionary entry to log.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def generate_with_ollama(prompt: str) -> str:
    """Generates a response from the local Ollama instance.

    Args:
        prompt: The full prompt for the LLM.

    Returns:
        The generated response string or an error message.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "repeat_penalty": REPEAT_PENALTY,
            "top_p": TOP_P,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        if resp.status_code == 200:
            return str(resp.json().get("response", "")).strip()
        return f"Fehler: Ollama HTTP {resp.status_code}"
    except requests.ConnectionError:
        return (
            "Fehler: Ollama ist nicht erreichbar. Bitte stellen Sie sicher, "
            "dass Ollama lokal läuft (ollama serve)."
        )
    except Exception as e:
        return f"Fehler bei der Antwortgenerierung: {e}"


def update_study_programs(faculty: str) -> Dict[str, Any]:
    """Updates the study program choices based on the selected faculty.

    Args:
        faculty: The selected faculty.

    Returns:
        A Gradio update dictionary for the dropdown.
    """
    programs = STUDY_PROGRAMS.get(faculty, [])
    return gr.update(choices=programs, value=programs[0] if programs else None)


def answer_question(question: str, faculty: str, study_program: str) -> Tuple[str, str]:
    """RAG pipeline: Retrieve -> Generate -> Answer + Sources (with queue management).

    Args:
        question: User's question.
        faculty: User's faculty.
        study_program: User's study program.

    Returns:
        A tuple of (answer, sources_text).
    """
    global _queue_count
    if not question.strip():
        return "", ""

    # Queue management: increment waiting count
    with _queue_lock:
        _queue_count += 1

    # Wait for a free slot
    _semaphore.acquire()
    try:
        return _process_question(question, faculty, study_program)
    finally:
        _semaphore.release()
        with _queue_lock:
            _queue_count -= 1


def _process_question(
    question: str, faculty: str, study_program: str
) -> Tuple[str, str]:
    """Internal processing of the question after queue slot is granted.

    Args:
        question: User's question.
        faculty: User's faculty.
        study_program: User's study program.

    Returns:
        A tuple of (answer, sources_text).
    """
    global _avg_gen_time
    init_engine()
    if engine is None:
        return "Fehler: Retrieval-Engine konnte nicht initialisiert werden.", ""

    # Faculty filter mapping
    db_faculty = FACULTY_FILTER_MAP.get(faculty, faculty)

    # Retrieval
    start = time.time()
    results = engine.search(
        question, db_faculty, top_k=TOP_K_RETRIEVAL, study_program_filter=study_program
    )
    retrieval_time = time.time() - start

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return (
            "Es konnten keine relevanten Dokumente gefunden werden. "
            "Bitte formulieren Sie Ihre Frage um oder wählen Sie eine andere Fakultät."
        ), ""

    # Build context and collect unique sources
    context_str = ""
    seen_sources = set()
    sources_display = []
    for i, doc_text in enumerate(docs):
        meta = metas[i]
        title = meta.get("title", "Unbekanntes Dokument")
        link = meta.get("source", "")
        svp_url = meta.get("url_studienverlaufsplan", "")

        context_str += f"\n--- {title} ---\nLINK ZUM DOKUMENT: {link}\n"
        if svp_url:
            context_str += f"LINK ZUM STUDIENVERLAUFSPLAN: {svp_url}\n"
        context_str += f"INHALT:\n{doc_text}\n"

        source_key = link or title
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            entry = f"📄 **{title}**"
            if link:
                entry += f"  \n[Dokument öffnen]({link})"
            if svp_url:
                entry += f"  \n[Studienverlaufsplan]({svp_url})"
            sources_display.append(entry)

    # Generate answer
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\nMETADATEN DES STUDIERENDEN:\nFakultät: {faculty}\n"
        f"Studiengang: {study_program}\n\nKONTEXT AUS DATENBANK:\n{context_str}\n\n"
        f"FRAGE:\n{question}"
    )

    start = time.time()
    answer = generate_with_ollama(full_prompt)
    gen_time = time.time() - start

    _avg_gen_time = 0.7 * _avg_gen_time + 0.3 * gen_time

    sources_text = "\n\n".join(sources_display)

    # Log + HiL-Queue
    interaction = {
        "id": f"{int(time.time())}_{hash(question) % 10000:04d}",
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "faculty": faculty,
        "study_program": study_program,
        "answer": answer,
        "sources": [m.get("title", "") for m in metas],
        "retrieval_time": round(retrieval_time, 3),
        "generation_time": round(gen_time, 3),
        "status": "pending",
    }
    log_json(CHAT_LOG, interaction)
    log_json(PENDING_QUEUE, interaction)

    return answer, sources_text


# Gradio UI - Student Interface
with gr.Blocks(title="TH Köln Prüfungsamt-Assistent") as demo:

    gr.HTML("""
    <div style="text-align:center; padding:20px 20px 10px;">
        <h1 style="color:#0b3d91; margin-bottom:5px;">TH Köln Prüfungsamt-Assistent</h1>
        <p style="color:#666; font-size:14px;">
            Stellen Sie Ihre Frage zu Prüfungsordnungen, Fristen und Formularen.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Ihre Frage",
                placeholder="z.B. Gib mir den Studienverlaufsplan für Medieninformatik.",
                lines=3,
            )
        with gr.Column(scale=1):
            faculty_dropdown = gr.Dropdown(
                choices=["F10", "F04", "F04 / F08"],
                value="F10",
                label="Ihre Fakultät",
            )
            study_program_dropdown = gr.Dropdown(
                choices=STUDY_PROGRAMS["F10"],
                value="Medieninformatik",
                label="Ihr Studiengang",
            )
            submit_btn = gr.Button("Frage stellen", variant="primary")

    gr.HTML("""
    <div style="background:#fff3cd; border:1px solid #ffc107; border-radius:6px; padding:10px 15px; margin:10px 0;">
        <strong>Hinweis:</strong> Diese Antwort wurde automatisch generiert und ist
        noch nicht vom Prüfungsamt geprüft. Bei wichtigen Entscheidungen wenden Sie
        sich bitte direkt an Ihr Prüfungsamt.
    </div>
    """)
    answer_output = gr.Markdown(label="Antwort", elem_id="answer-box")
    gr.HTML("<hr style='margin:10px 0;'>")
    gr.HTML("<h3 style='color:#0b3d91; margin:5px 0;'>Verwendete Quellen</h3>")
    sources_output = gr.Markdown(value="*Noch keine Anfrage gestellt.*")

    # Queue status display
    queue_status = gr.HTML(value="", visible=False)

    gr.HTML("""
    <p style="color:#999; font-size:12px; text-align:center; margin-top:20px;">
        Dieser Assistent basiert auf offiziellen Dokumenten der TH Köln.
    </p>
    """)

    # Events
    faculty_dropdown.change(
        fn=update_study_programs,
        inputs=[faculty_dropdown],
        outputs=[study_program_dropdown],
    )

    def show_queue_status() -> Dict[str, Any]:
        """Shows the current queue status.

        Returns:
            A Gradio update dictionary for the HTML component.
        """
        with _queue_lock:
            waiting = max(0, _queue_count - MAX_CONCURRENT)
        if waiting > 0:
            est_wait = int(round(waiting * _avg_gen_time, 0))
            return gr.update(
                value=f'<div style="background:#e8f4fd; border:1px solid #0b3d91; '
                f'border-radius:6px; padding:10px 15px; margin:10px 0;">⏳ '
                f"<strong>{waiting} Anfrage(n) in der Warteschlange</strong> — "
                f"geschätzte Wartezeit: ~{est_wait}s</div>",
                visible=True,
            )
        return gr.update(value="", visible=False)

    timer = gr.Timer(2.0)
    timer.tick(show_queue_status, outputs=[queue_status])

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, faculty_dropdown, study_program_dropdown],
        outputs=[answer_output, sources_output],
    )
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, faculty_dropdown, study_program_dropdown],
        outputs=[answer_output, sources_output],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public", action="store_true", help="Make publicly accessible via ngrok"
    )
    args = parser.parse_args()

    print(
        f"Model: {MODEL_NAME} | temp={TEMPERATURE} | rp={REPEAT_PENALTY} | "
        f"top_k={TOP_K_RETRIEVAL}"
    )
    print(f"Max concurrent requests: {MAX_CONCURRENT}")
    demo.queue(default_concurrency_limit=MAX_CONCURRENT)

    if args.public:
        from pyngrok import ngrok

        public_url = ngrok.connect(7860)
        print(f"\n{'='*60}")
        print(f"  PUBLIC LINK: {public_url}")
        print(f"{'='*60}\n")
        demo.launch(server_name="0.0.0.0", server_port=7860)  # nosec B104
    else:
        demo.launch(share=False, server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
