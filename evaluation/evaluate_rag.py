"""Script for expanded grid search evaluation of the RAG system.

Compares local and cloud models with various parameter combinations.
Phases range from small model grid search to cloud model comparisons.
"""

import os
import sys
import json
import csv
import time
import re
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from retrieval_engine import HybridRetrievalEngine
from evaluation_metrics import RAGEvaluator
from llm_client import LLMClient

load_dotenv()

# Shared parameters
FIXED_TOP_P: float = 0.85
FIXED_NUM_CTX: int = 4096

# Grid for parameter search
TEMPERATURE_VALUES: List[float] = [0.0, 0.2, 0.4]
REPEAT_PENALTY_VALUES: List[float] = [1.0, 1.1, 1.2]
TOP_K_VALUES: List[int] = [3, 5, 7, 9]

# Phase configurations
PHASE1_MODELS: List[Dict[str, str]] = [
    {"provider": "local", "model": "gemma2"},
    {"provider": "local", "model": "qwen2.5"},
    {"provider": "local", "model": "mistral"},
    {"provider": "local", "model": "llama3.1"},
]
PHASE1_CSV: str = "data/evaluation_logs/phase1_small_models_grid.csv"

PHASE2_MODELS: List[Dict[str, str]] = [
    {"provider": "local", "model": "phi4"},
    {"provider": "local", "model": "qwen2.5:14b"},
]
PHASE2_CSV: str = "data/evaluation_logs/phase2_large_models_grid.csv"

PHASE3_BEST_CONFIGS: Dict[str, Dict[str, Union[float, int]]] = {
    "gemma2": {"temperature": 0.0, "repeat_penalty": 1.0, "top_k": 5},
    "qwen2.5": {"temperature": 0.4, "repeat_penalty": 1.0, "top_k": 3},
    "mistral": {"temperature": 0.0, "repeat_penalty": 1.2, "top_k": 5},
    "llama3.1": {"temperature": 0.2, "repeat_penalty": 1.2, "top_k": 5},
    "phi4": {"temperature": 0.4, "repeat_penalty": 1.0, "top_k": 5},
    "qwen2.5:14b": {"temperature": 0.0, "repeat_penalty": 1.0, "top_k": 5},
}
PHASE3_MODELS: List[Dict[str, str]] = [
    {"provider": "local", "model": "gemma2"},
    {"provider": "local", "model": "qwen2.5"},
    {"provider": "local", "model": "mistral"},
    {"provider": "local", "model": "llama3.1"},
    {"provider": "local", "model": "phi4"},
    {"provider": "local", "model": "qwen2.5:14b"},
]
PHASE3_CSV: str = "data/evaluation_logs/phase3_head_to_head.csv"

PHASE4_MODELS: List[Dict[str, str]] = [
    {"provider": "local", "model": "qwen2.5:14b"},
    {"provider": "local", "model": "phi4"},
    {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    {"provider": "gemini", "model": "gemini-2.5-flash"},
]
PHASE4_CLOUD_CONFIGS: Dict[str, Dict[str, float]] = {
    "llama-3.3-70b-versatile": {"temperature": 0.0},
    "gemini-2.5-flash": {"temperature": 0.0},
}
PHASE4_CSV: str = "data/evaluation_logs/phase4_local_vs_cloud.csv"

PHASE5_MODEL: Dict[str, str] = {"provider": "local", "model": "qwen2.5:14b"}
PHASE5_TOP_K_VALUES: List[int] = [6]
PHASE5_CSV: str = "data/evaluation_logs/phase5_topk_fine_grained.csv"

SYSTEM_PROMPT: str = """Sie sind ein präziser Studienberater-Assistent der TH Köln.
Ihre Aufgabe ist es, studentische Fragen AUSSCHLIESSLICH basierend auf dem bereitgestellten Datenbank-Kontext zu beantworten.

WICHTIG:
- Antworten Sie immer auf Deutsch.
- Geben Sie keine Zwischengedanken/Analysen/"chain-of-thought" aus (keine <think>-Blöcke o.ä.).

BEFOLGEN SIE ZWINGEND DIESE STRUKTURELLEN REGELN:

0. KEINE INTERNEN VERWEISE:
Nennen Sie NIEMALS Dokument-Nummern wie "Dokument 1", "Quelle 3" oder "laut Dokument 5" im Fließtext. Nutzen Sie ausschließlich die Informationen aus den Texten selbst, ohne auf die interne Nummerierung zu verweisen.

1. MEHRTEILIGE FRAGEN (MULTI-INTENT):
Prüfen Sie immer, ob der Student mehrere Fragen in einem Satz stellt (z.B. "Reicht das?" UND "Wie lange habe ich Zeit?" UND "Wo ist das Formular?"). Sie MÜSSEN jede einzelne Teilfrage beantworten.

2. QUELLENANGABE (STRIKTE FORMATIERUNG):
Sie DÜRFEN Quellenangaben oder Links NIEMALS in den Fließtext schreiben.
Erstellen Sie KEINE eigenen Links oder Markdown-Links wie [Text](). Verwenden Sie NUR die exakten URLs aus dem Kontext.
Sie MÜSSEN Ihre finale Antwort zwingend in zwei Blöcke aufteilen. Der zweite Block MUSS exakt so aussehen:
Quellen:
- [Link 1 einfügen]
- [Link 2 einfügen]

3. FEHLENDER KONTEXT (FALLBACK):
Wenn die exakte Antwort auf die Frage (oder eine der Teilfragen) nicht im Kontext steht, antworten Sie für diesen Teil EXAKT und AUSSCHLIESSLICH mit dem Satz: "Dazu liegen mir keine Informationen vor."
"""


def generate_response(
    provider: str,
    model: str,
    full_prompt: str,
    temp: float,
    extra_opts: Optional[Dict[str, Any]] = None,
) -> str:
    """Generates a response using either a local or cloud model.

    Args:
        provider: The model provider ('local', 'groq', 'gemini').
        model: The model name.
        full_prompt: The full prompt text.
        temp: Temperature setting.
        extra_opts: Extra options for the generator.

    Returns:
        The generated response string.
    """
    if provider == "local":
        return generate_with_ollama_direct(model, full_prompt, temp, extra_opts)
    else:
        return generate_with_cloud_model(provider, model, full_prompt, temp)


def generate_with_cloud_model(
    provider: str, model: str, full_prompt: str, temp: float
) -> str:
    """Generates a response from a cloud-based LLM.

    Args:
        provider: Cloud provider name.
        model: Model name.
        full_prompt: Full prompt text.
        temp: Temperature setting.

    Returns:
        The generated response string or error.
    """
    try:
        if provider == "gemini" and isinstance(model, str):
            if not model.startswith("models/"):
                model = f"models/{model}"

        max_tokens = 768 if provider == "groq" else 1536
        client = LLMClient(
            api_choice=provider, llm=model, temperature=temp, max_tokens=max_tokens
        )
        messages = [{"role": "user", "content": full_prompt}]

        last_err = None
        for attempt in range(3):
            try:
                response = client.chat_completion(messages)
                cleaned = (response or "").strip()
                if "<think>" in cleaned and "</think>" in cleaned:
                    cleaned = cleaned.split("</think>", 1)[1].strip()
                return cleaned
            except Exception as e:
                last_err = e
                err_str = str(e)
                if (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "RateLimit" in err_str
                ):
                    time.sleep(2 + attempt * 4)
                    continue
                break

        return f"FEHLER: Cloud-Modell {provider}/{model} - {last_err}"
    except Exception as e:
        return f"FEHLER: Cloud-Modell {provider}/{model} - {e}"


def generate_with_ollama_direct(
    model_name: str,
    full_prompt: str,
    temp: float,
    extra_opts: Optional[Dict[str, Any]] = None,
) -> str:
    """Generates a response directly from local Ollama API.

    Args:
        model_name: Model name.
        full_prompt: Full prompt text.
        temp: Temperature setting.
        extra_opts: Extra Ollama options.

    Returns:
        The generated response string or error.
    """
    url = "http://localhost:11434/api/generate"
    options = {"temperature": temp, "num_predict": 1024}
    if extra_opts:
        options.update(extra_opts)
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": False,
        "options": options,
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        if response.status_code == 200:
            return str(response.json().get("response", "")).strip()
        return f"FEHLER: Lokaler Ollama HTTP Status {response.status_code}"
    except Exception as e:
        return f"FEHLER: Lokaler Ollama Absturz - {e}"


def _needs_multiple_sources(question: str, item: Optional[Dict[str, Any]] = None) -> bool:
    """Checks if the question likely requires multiple source documents.

    Args:
        question: User question.
        item: Optional question metadata.

    Returns:
        True if multiple sources are likely needed.
    """
    q = (question or "").lower()
    if ("po" in q or "prüfungsordnung" in q) and any(
        w in q for w in ["unterschied", "zwischen", "po4", "po5", "po6"]
    ):
        return True
    if item and item.get("multi_intent_parts", 1) >= 2:
        return True
    if item and len(item.get("expected_sources", [])) >= 2:
        return True
    return False


def _pick_relevant_sources(
    metas: List[Dict[str, Any]], faculty: str, study_program: str, max_sources: int
) -> List[str]:
    """Filters and picks the most relevant source URLs from metadata.

    Args:
        metas: List of metadata dictionaries.
        faculty: Faculty filter.
        study_program: Study program filter.
        max_sources: Maximum number of sources to pick.

    Returns:
        A list of picked source URLs.
    """
    picked: List[str] = []
    seen = set()
    faculty_norm = (faculty or "").strip().lower()
    sp_norm = (study_program or "").strip().lower()

    for meta in metas or []:
        if len(picked) >= max_sources:
            break

        src = (meta or {}).get("source")
        if not src or src in seen:
            continue

        src_l = src.lower()
        if not (src_l.endswith(".pdf") or "/mam/downloads/" in src_l):
            continue

        meta_fac = str((meta or {}).get("faculty", "")).strip().lower()
        if meta_fac and faculty_norm and meta_fac != faculty_norm:
            continue

        meta_sp = str((meta or {}).get("study_program", "")).strip().lower()
        if (
            meta_sp
            and meta_sp not in ("allgemein", "")
            and "alle" not in meta_sp
            and sp_norm
            and sp_norm not in meta_sp
            and meta_sp not in sp_norm
        ):
            continue

        picked.append(src)
        seen.add(src)

    return picked


def enforce_source_policy(
    response_text: str,
    metas: List[Dict[str, Any]],
    faculty: str,
    study_program: str,
    question: str,
    item: Dict[str, Any],
) -> str:
    """Enforces strict source formatting and filters irrelevant sources.

    Args:
        response_text: Generated response text.
        metas: Retrieved metadata.
        faculty: Faculty filter.
        study_program: Study program filter.
        question: User question.
        item: Question metadata.

    Returns:
        The response with enforced source formatting.
    """
    text = (response_text or "").strip()
    if text.startswith("FEHLER:"):
        return text

    if "Dazu liegen mir keine Informationen vor." in text:
        return "Dazu liegen mir keine Informationen vor."

    max_sources = 3 if _needs_multiple_sources(question, item) else 1
    sources = _pick_relevant_sources(
        metas, faculty, study_program, max_sources=max_sources
    )

    body = re.split(
        r"\n\s*(Quellen:|Quelle:)\s*\n", text, maxsplit=1, flags=re.IGNORECASE
    )[0].strip()
    body = re.sub(r"https?://[^\s<>'\"\]\)]+", "", body).strip()
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    if not sources:
        return "Dazu liegen mir keine Informationen vor."

    q_lower = (question or "").lower()
    svp_keywords = [
        "studienverlauf",
        "verlaufsplan",
        "modulplan",
        "semester",
        "ects",
        "leistungspunkte",
        "regelstudienzeit",
        "prüfungsordnung",
        "praxissemester",
    ]
    if any(kw in q_lower for kw in svp_keywords):
        for meta in metas or []:
            svp_url = (meta or {}).get("url_studienverlaufsplan", "")
            if svp_url and svp_url not in sources:
                sources.append(svp_url)
                break

    sources_block = "\n\nQuellen:\n" + "\n".join(f"- {s}" for s in sources)
    return f"{body}{sources_block}".strip()


def load_questions(
    filepath: str = "data/improved_test_questions.json",
) -> List[Dict[str, Any]]:
    """Loads evaluation questions from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        A list of question dictionaries.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tasks(phase: int, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Builds the list of evaluation tasks based on the phase.

    Args:
        phase: Evaluation phase number.
        questions: List of questions.

    Returns:
        A list of task dictionaries.
    """
    tasks = []

    if phase in (1, 2):
        models = PHASE1_MODELS if phase == 1 else PHASE2_MODELS
        for model_cfg in models:
            for temp in TEMPERATURE_VALUES:
                for rp in REPEAT_PENALTY_VALUES:
                    for tk in TOP_K_VALUES:
                        for q in questions:
                            tasks.append(
                                {
                                    "provider": model_cfg["provider"],
                                    "model": model_cfg["model"],
                                    "k": tk,
                                    "temp": temp,
                                    "item": q,
                                    "extra_opts": {
                                        "repeat_penalty": rp,
                                        "top_p": FIXED_TOP_P,
                                        "num_ctx": FIXED_NUM_CTX,
                                    },
                                }
                            )

    elif phase == 3:
        for model_cfg in PHASE3_MODELS:
            model_name = model_cfg["model"]
            best = PHASE3_BEST_CONFIGS.get(model_name, {})
            temp = float(best.get("temperature", 0.0))
            rp = float(best.get("repeat_penalty", 1.0))
            tk = int(best.get("top_k", 7))
            for q in questions:
                tasks.append(
                    {
                        "provider": model_cfg["provider"],
                        "model": model_name,
                        "k": tk,
                        "temp": temp,
                        "item": q,
                        "extra_opts": {
                            "repeat_penalty": rp,
                            "top_p": FIXED_TOP_P,
                            "num_ctx": FIXED_NUM_CTX,
                        },
                    }
                )

    elif phase == 4:
        for model_cfg in PHASE4_MODELS:
            provider = model_cfg["provider"]
            model_name = model_cfg["model"]
            if provider == "local":
                best = PHASE3_BEST_CONFIGS.get(model_name, {})
                temp = float(best.get("temperature", 0.0))
                rp = float(best.get("repeat_penalty", 1.0))
                tk = int(best.get("top_k", 7))
                for q in questions:
                    tasks.append(
                        {
                            "provider": provider,
                            "model": model_name,
                            "k": tk,
                            "temp": temp,
                            "item": q,
                            "extra_opts": {
                                "repeat_penalty": rp,
                                "top_p": FIXED_TOP_P,
                                "num_ctx": FIXED_NUM_CTX,
                            },
                        }
                    )
            else:
                cloud_cfg = PHASE4_CLOUD_CONFIGS.get(model_name, {})
                temp = float(cloud_cfg.get("temperature", 0.0))
                for q in questions:
                    tasks.append(
                        {
                            "provider": provider,
                            "model": model_name,
                            "k": 5,
                            "temp": temp,
                            "item": q,
                            "extra_opts": {},
                        }
                    )

    elif phase == 5:
        model_cfg = PHASE5_MODEL
        for tk in PHASE5_TOP_K_VALUES:
            for temp in TEMPERATURE_VALUES:
                for rp in REPEAT_PENALTY_VALUES:
                    for q in questions:
                        tasks.append(
                            {
                                "provider": model_cfg["provider"],
                                "model": model_cfg["model"],
                                "k": tk,
                                "temp": temp,
                                "item": q,
                                "extra_opts": {
                                    "repeat_penalty": rp,
                                    "top_p": FIXED_TOP_P,
                                    "num_ctx": FIXED_NUM_CTX,
                                },
                            }
                        )

    return tasks


def get_csv_path(phase: int) -> str:
    """Returns the CSV file path for a given phase.

    Args:
        phase: Phase number.

    Returns:
        File path string.
    """
    return {1: PHASE1_CSV, 2: PHASE2_CSV, 3: PHASE3_CSV, 4: PHASE4_CSV, 5: PHASE5_CSV}[
        phase
    ]


def run_evaluation(phase: int) -> None:
    """Runs the full evaluation pipeline for a specific phase.

    Args:
        phase: Phase number.
    """
    phase_names = {
        1: "Phase 1: Kleine Modelle Grid Search (7B)",
        2: "Phase 2: Große Modelle Grid Search (14B)",
        3: "Phase 3: Head-to-Head (beste Configs)",
        4: "Phase 4: Lokale vs Cloud",
        5: "Phase 5: Fine-Grained Top-K (qwen2.5:14b)",
    }
    print(f"\n{'='*60}")
    print(f"  {phase_names[phase]}")
    print(f"{'='*60}\n")

    engine = HybridRetrievalEngine()
    evaluator = RAGEvaluator()
    questions = load_questions()
    tasks = build_tasks(phase, questions)

    print(f"Number of tasks: {len(tasks)}")
    print(f"Output: {get_csv_path(phase)}\n")

    results = []
    scored_results = []

    for task in tqdm(tasks, desc=phase_names[phase], unit="Eval"):
        provider = task["provider"]
        model = task["model"]
        k = task["k"]
        temp = task["temp"]
        item = task["item"]
        extra_opts = task.get("extra_opts", {})

        question = item["question"]
        faculty = item["faculty"]
        study_program = item.get("study_program", "Allgemein")

        try:
            res = engine.search(
                question, faculty, top_k=k, study_program_filter=study_program
            )
            docs = res["documents"][0]
            metas = res["metadatas"][0]
        except Exception:
            continue

        retrieved_sources = [m.get("source", "N/A") for m in metas]
        expected_src = item.get("expected_sources", [])
        retrieval_hit = (
            all(
                any(
                    exp.rstrip("/").split("/")[-1].lower() in rs.lower()
                    for rs in retrieved_sources
                )
                for exp in expected_src
            )
            if expected_src
            else True
        )

        context_str = ""
        for i, doc_text in enumerate(docs):
            meta = metas[i]
            context_str += f"\n--- {meta.get('title', 'Dokument')} ---\nLINK: {meta.get('source', 'N/A')}\nINHALT:\n{doc_text}\n"

        full_prompt = f"{SYSTEM_PROMPT}\n\nMETADATEN DES STUDIERENDEN:\nFakultät: {faculty}\nStudiengang: {study_program}\n\nKONTEXT AUS DATENBANK:\n{context_str}\n\nFRAGE:\n{question}"

        start_time = time.time()
        response_text = generate_response(
            provider, model, full_prompt, temp, extra_opts
        )
        response_text = enforce_source_policy(
            response_text, metas, faculty, study_program, question, item
        )
        duration = round(time.time() - start_time, 2)

        is_error = isinstance(response_text, str) and response_text.startswith(
            "FEHLER:"
        )

        if is_error:
            metrics = {
                "overall_score": None,
                "has_fallback": False,
                "has_sources": False,
                "source_count": 0,
                "multi_intent_complete": False,
                "hallucination_free": False,
                "source_format_correct": False,
                "instruction_following": False,
                "response_length": len(response_text) if isinstance(response_text, str) else 0,
                "context_usage_percent": 0.0,
            }
        else:
            metrics = evaluator.evaluate_response(
                question=item["question"],
                response=response_text,
                context=context_str,
                expected_fallback=item.get("expected_fallback", False),
                multi_intent_parts=item.get("multi_intent_parts", 1),
                question_data=item,
            )

        result = {
            "phase": phase,
            "provider": provider,
            "model": model,
            "temperature": temp,
            "top_k": k,
            "faculty": faculty,
            "study_program": study_program,
            "question": question,
            "generation_time_sec": duration,
            "response": response_text,
            "is_error": is_error,
            "overall_score": metrics["overall_score"],
            "has_fallback": metrics["has_fallback"],
            "has_sources": metrics["has_sources"],
            "source_count": metrics["source_count"],
            "multi_intent_complete": metrics["multi_intent_complete"],
            "hallucination_free": metrics["hallucination_free"],
            "source_format_correct": metrics["source_format_correct"],
            "instruction_following": metrics["instruction_following"],
            "response_length": metrics["response_length"],
            "context_usage_percent": metrics["context_usage_percent"],
            "answer_complete": metrics.get("answer_complete", True),
            "no_semantic_contradiction": metrics.get("no_semantic_contradiction", True),
            "has_expected_keywords": metrics.get("has_expected_keywords", True),
            "required_keywords_present": metrics.get("required_keywords_present", True),
            "link_validity": metrics.get("link_validity", True),
            "sources_match": metrics.get("sources_match", True),
            "is_trick_question": metrics.get("is_trick_question", False),
            "repeat_penalty": extra_opts.get("repeat_penalty", ""),
            "top_p": extra_opts.get("top_p", ""),
            "num_ctx": extra_opts.get("num_ctx", ""),
            "retrieved_sources": "; ".join(retrieved_sources),
            "retrieval_hit": retrieval_hit,
        }

        results.append(result)
        if not is_error:
            scored_results.append(result)

    os.makedirs("data/evaluation_logs", exist_ok=True)
    csv_file = get_csv_path(phase)
    if results:
        keys = results[0].keys()
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(results)

    summary = evaluator.generate_evaluation_summary(scored_results)
    summary["phase"] = phase
    summary["total_evaluations_including_errors"] = len(results)
    summary["error_rate"] = (len(results) - len(scored_results)) / max(1, len(results))
    summary_file = csv_file.replace(".csv", "_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nPhase {phase} finished.")
    print(f"Results: {csv_file}")
    print(f"Summary: {summary_file}")
    print(f"Average score: {summary.get('average_score', 0):.1f}/100")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("1", "2", "3", "4", "5"):
        print("Usage: python evaluate_rag.py <phase>")
        sys.exit(1)
    run_evaluation(int(sys.argv[1]))
