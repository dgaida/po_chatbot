"""Local evaluation metrics for the RAG system.

Provides the RAGEvaluator class to evaluate chatbot responses based on
various metrics like hallucination prevention, source presence, and multi-intent
completion.
"""

import re
from typing import Any, Dict, List, Optional, Set


class RAGEvaluator:
    """Evaluator for RAG system responses."""

    def __init__(self) -> None:
        """Initializes the evaluator with default fallback phrase."""
        self.fallback_phrase: str = "Dazu liegen mir keine Informationen vor."

    def evaluate_response(
        self,
        question: str,
        response: str,
        context: str,
        expected_fallback: bool = False,
        multi_intent_parts: int = 1,
        question_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Main evaluation function using all metrics.

        Args:
            question: The user question.
            response: The generated response.
            context: The retrieved context.
            expected_fallback: Whether a fallback was expected.
            multi_intent_parts: Number of expected intent parts.
            question_data: Additional data about the question.

        Returns:
            A dictionary containing all evaluated metrics and the overall score.
        """
        if question_data is None:
            question_data = {
                "question": question,
                "multi_intent_parts": multi_intent_parts,
            }

        metrics = {
            "question": question,
            "response": response,
            "response_length": len(response),
            "has_fallback": self.check_fallback_usage(response),
            "has_sources": self.check_sources_present(response),
            "source_count": self.count_sources(response),
            "multi_intent_complete": self.check_multi_intent_completion(
                question, response, multi_intent_parts, question_data
            ),
            "hallucination_free": self.check_hallucination_prevention(
                context, response
            ),
            "source_format_correct": self.check_source_format(response),
            "instruction_following": self.check_instruction_following(response),
            "context_usage_percent": self.check_context_usage(context, response),
            "answer_complete": self.check_answer_completeness(response, question_data),
            "no_semantic_contradiction": self.check_semantic_contradiction(
                response, question_data
            ),
            "has_expected_keywords": self.check_expected_keywords(
                response, question_data
            ),
            "required_keywords_present": self.check_required_keywords(
                response, question_data
            ),
            "link_validity": self.check_link_validity(response),
            "sources_match": self.check_expected_sources(response, question_data),
            "is_trick_question": question_data.get("is_trick_question", False),
            "overall_score": 0,
        }

        # Calculate overall score (0-100)
        metrics["overall_score"] = self.calculate_overall_score(metrics)

        return metrics

    def check_fallback_usage(self, response: str) -> bool:
        """Checks if the correct fallback phrase was used.

        Args:
            response: The generated response.

        Returns:
            True if fallback was used, False otherwise.
        """
        return self.fallback_phrase in response.strip()

    def check_sources_present(self, response: str) -> bool:
        """Checks if sources are present (except when fallback is used).

        Args:
            response: The generated response.

        Returns:
            True if sources are present or not needed, False otherwise.
        """
        if self.check_fallback_usage(response):
            return True  # Fallback does not require sources

        # Search for source links
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, response)
        return len(urls) > 0

    def count_sources(self, response: str) -> int:
        """Counts the number of source links.

        Args:
            response: The generated response.

        Returns:
            The number of source links found.
        """
        if self.check_fallback_usage(response):
            return 0

        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, response)
        return len(urls)

    def check_multi_intent_completion(
        self,
        question: str,
        response: str,
        expected_parts: int,
        question_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Checks if ALL parts of a multi-part question were answered.

        Args:
            question: The user question.
            response: The generated response.
            expected_parts: Number of expected intent parts.
            question_data: Additional question data.

        Returns:
            True if multi-intent is complete, False otherwise.
        """
        if expected_parts <= 1:
            return True

        if self.check_fallback_usage(response):
            return False

        question_lower = question.lower()
        response_lower = response.lower()

        # Strict sub-question detection
        sub_q_checks = []

        # "Wo finde ich...?" / "Wo...?" -> Needs link/form/source
        if "wo finde" in question_lower or (
            "wo" in question_lower and "formular" in question_lower
        ):
            has_location = any(
                x in response_lower
                for x in ["http", "formular", "antrag", "download", "link"]
            )
            sub_q_checks.append(has_location)

        # "Ab wann...?" -> Needs date/time reference
        if "ab wann" in question_lower or "wann gilt" in question_lower:
            has_date = bool(
                re.search(
                    r"\b(20\d{2}|ws|ss|semester|wintersemester|sommersemester)\b",
                    response_lower,
                )
            )
            sub_q_checks.append(has_date)

        # "Wie viele...?" / "Wie lange...?" -> Needs number
        if "wie viele" in question_lower or "wie lange" in question_lower:
            has_number = bool(re.search(r"\d+", response))
            sub_q_checks.append(has_number)

        # "Gibt es...? Wenn ja,..." -> Needs confirmation + details
        if "gibt es" in question_lower and "wenn ja" in question_lower:
            has_confirmation = any(
                x in response_lower
                for x in ["ja", "nein", "existiert", "gibt es", "vorhanden"]
            )
            sub_q_checks.append(has_confirmation)

        if sub_q_checks:
            return all(sub_q_checks)

        # Fallback: sentence count heuristic
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        return len(sentences) >= expected_parts

    def check_hallucination_prevention(self, context: str, response: str) -> bool:
        """Checks if the response only uses information from the context.

        Args:
            context: The retrieved context.
            response: The generated response.

        Returns:
            True if no hallucination detected, False otherwise.
        """

        def _num_variants(t: str) -> Set[str]:
            """Generates variants of a numeric string with different decimal marks."""
            if re.fullmatch(r"\d{1,4}([\.,]\d+)?", t):
                return {t, t.replace(",", "."), t.replace(".", ",")}
            return {t}

        if self.check_fallback_usage(response):
            return True

        context_l = (context or "").lower()
        response_s = response or ""
        response_s.lower()

        # Remove URLs before extracting facts
        response_no_urls = re.sub(r'https?://[^\s<>"]+', "", response_s)

        facts: Set[str] = set()

        # Paragraph references
        facts.update(
            m.group(0).strip().lower()
            for m in re.finditer(
                r"§\s*\d+[a-z]?", response_no_urls, flags=re.IGNORECASE
            )
        )

        # Date patterns
        facts.update(
            m.group(0).strip().lower()
            for m in re.finditer(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b", response_no_urls)
        )

        # Numbers
        for m in re.finditer(r"\b\d{1,4}([\.,]\d+)?\b", response_no_urls):
            tok = m.group(0).strip().lower()
            if tok in {"0", "1"}:
                continue
            facts.add(tok)

        if not facts:
            return True

        for fact in facts:
            if fact.startswith("§"):
                if fact not in context_l:
                    return False
                continue

            if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{2,4}", fact):
                if fact not in context_l:
                    return False
                continue

            variants = _num_variants(fact)
            if not any(v in context_l for v in variants):
                return False

        return True

    def check_source_format(self, response: str) -> bool:
        """Checks if sources are correctly formatted (not in-text).

        Args:
            response: The generated response.

        Returns:
            True if format is correct, False otherwise.
        """
        if self.check_fallback_usage(response):
            return True

        has_sources_block = "Quellen:" in response or "Quelle:" in response

        if has_sources_block:
            sources_section = (
                response.split("Quellen:")[-1]
                if "Quellen:" in response
                else response.split("Quelle:")[-1]
            )
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls_in_sources = re.findall(url_pattern, sources_section)

            total_urls = re.findall(url_pattern, response)
            return len(urls_in_sources) == len(total_urls)

        return False

    def check_instruction_following(self, response: str) -> bool:
        """Checks if structural instructions were followed.

        Args:
            response: The generated response.

        Returns:
            True if instructions were followed, False otherwise.
        """
        if self.check_fallback_usage(response):
            return True

        leak_patterns = [
            r"(?:in|laut|aus|gemäß)\s+dokument\s+\d",
            r"text[- ]?chunk\s+\d",
            r"dokument\s+\d\s+(?:steht|sagt|beschreibt|enthält|zeigt)",
            r"(?:im|aus dem)\s+(?:ersten|zweiten|dritten|vierten|fünften)\s+dokument",
        ]
        response_lower = response.lower()
        for pattern in leak_patterns:
            if re.search(pattern, response_lower):
                return False

        has_sources_block = "Quellen:" in response or "Quelle:" in response
        has_no_sources_in_text = self.check_source_format(response)

        return has_sources_block and has_no_sources_in_text

    def check_answer_completeness(
        self, response: str, question_data: Dict[str, Any]
    ) -> bool:
        """Checks if all parts of multi-intent questions were answered.

        Args:
            response: The generated response.
            question_data: Question metadata.

        Returns:
            True if answer is complete, False otherwise.
        """
        if self.check_fallback_usage(response):
            return True

        multi_intent_parts = question_data.get("multi_intent_parts", 1)
        if multi_intent_parts <= 1:
            return True

        question = question_data.get("question", "").lower()
        response_lower = response.lower()

        completeness_checks = []

        if "wo finde" in question or "wo gibt" in question:
            has_location = any(
                x in response_lower
                for x in ["http", "formular", "antrag", "link", "download"]
            )
            completeness_checks.append(has_location)

        if "wie viele" in question or "wie lange" in question:
            has_number = bool(re.search(r"\d+", response))
            completeness_checks.append(has_number)

        if "welche" in question and "unterschied" in question:
            has_enumeration = response.count("-") >= 2 or response.count("\n") >= 2
            completeness_checks.append(has_enumeration)

        if completeness_checks:
            return all(completeness_checks)

        return True

    def check_semantic_contradiction(
        self, response: str, question_data: Dict[str, Any]
    ) -> bool:
        """Checks for semantic contradictions in the response.

        Args:
            response: The generated response.
            question_data: Question metadata.

        Returns:
            True if no contradiction found, False otherwise.
        """
        if self.check_fallback_usage(response):
            return True

        question = question_data.get("question", "").lower()
        response_lower = response.lower()

        if any(
            word in question
            for word in ["unterschied", "vergleich", "differenz", "zwei version"]
        ):
            numbers = re.findall(
                r"\b(\d+)\s*(cp|ects|lp|leistungspunkte|wochen?|monate?|semester)\b",
                response_lower,
            )
            if len(numbers) >= 2:
                values = [int(n[0]) for n in numbers]
                if len(set(values)) == 1:
                    return False

            weeks = re.findall(r"\b(\d+)\s*wochen?\b", response_lower)
            if len(weeks) >= 2 and len(set(weeks)) == 1:
                return False

        if (
            "zwei version" in question
            or "welche version" in question
            or "verfügbar" in question
        ):
            po_mentions = re.findall(r"po\s*[0-9]+", response_lower)
            if len(po_mentions) >= 2 and len(set(po_mentions)) == 1:
                return False

        passive_without_subject = re.search(
            r"(dargelegt|nachgewiesen|eingereicht)\s+(sowie|und|oder)\s+\w+\s+werden[,.]",
            response_lower,
        )
        if passive_without_subject and not re.search(
            r"(müssen|muss|sollte|ist)\s+\w+\s+(dargelegt|nachgewiesen)", response_lower
        ):
            return False

        return True

    def check_expected_sources(
        self, response: str, question_data: Dict[str, Any]
    ) -> bool:
        """Checks if expected source URLs are cited in the response.

        Args:
            response: The generated response.
            question_data: Question metadata.

        Returns:
            True if sources match expectations, False otherwise.
        """
        expected = question_data.get("expected_sources", [])
        if not expected:
            return True

        if self.check_fallback_usage(response):
            return False

        response_lower = response.lower()
        found_urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', response_lower)

        multi_intent = question_data.get("multi_intent_parts", 1)
        require_all = multi_intent >= 2 and len(expected) >= 2

        if require_all:
            for exp_url in expected:
                exp_filename = exp_url.rstrip("/").split("/")[-1].lower()
                if not any(exp_filename in fu for fu in found_urls):
                    return False
            return True
        else:
            for exp_url in expected:
                exp_filename = exp_url.rstrip("/").split("/")[-1].lower()
                if any(exp_filename in fu for fu in found_urls):
                    return True
            return False

    def check_link_validity(self, response: str) -> bool:
        """Checks if links are correctly formatted and not empty.

        Args:
            response: The generated response.

        Returns:
            True if links are valid, False otherwise.
        """
        if self.check_fallback_usage(response):
            return True

        empty_links = re.findall(r"\[\]|\[Link\]|\[\]\(\)", response)
        if empty_links:
            return False

        empty_url_links = re.findall(r"\[[^\]]+\]\(\s*\)", response)
        if empty_url_links:
            return False

        response_lower = response.lower()
        phantom_patterns = [
            r"unter\s+folgend\w*\s+link",
            r"folgend\w*\s+links?\s*:",
            r"\[link\s*\d*\s*einfügen\]",
            r"hier\s+ist\s+der\s+link",
            r"link\s+zum\s+download\s*:",
        ]
        for pat in phantom_patterns:
            match = re.search(pat, response_lower)
            if match:
                after = response[match.end() : match.end() + 200]
                if not re.search(r"https?://", after):
                    return False

        return True

    def check_expected_keywords(
        self, response: str, question_data: Dict[str, Any]
    ) -> bool:
        """Checks if expected keywords are present (70% threshold).

        Args:
            response: The generated response.
            question_data: Question metadata.

        Returns:
            True if threshold is met, False otherwise.
        """
        expected = question_data.get("expected_answer_keywords", [])
        if not expected:
            return True

        if self.check_fallback_usage(response):
            return True

        response_lower = response.lower()
        found = sum(1 for kw in expected if kw.lower() in response_lower)
        threshold = max(1, int(len(expected) * 0.7))

        return found >= threshold

    def check_required_keywords(
        self, response: str, question_data: Dict[str, Any]
    ) -> bool:
        """Checks if all strictly required keywords are present.

        Args:
            response: The generated response.
            question_data: Question metadata.

        Returns:
            True if all required keywords are present, False otherwise.
        """
        required = question_data.get("required_keywords", [])
        if not required:
            return True

        if self.check_fallback_usage(response):
            return question_data.get("is_trick_question", False)

        response_lower = response.lower()
        return all(kw.lower() in response_lower for kw in required)

    def check_context_usage(self, context: str, response: str) -> float:
        """Measures the percentage of context used in the response.

        Args:
            context: The retrieved context.
            response: The generated response.

        Returns:
            Context usage percentage.
        """
        if self.check_fallback_usage(response):
            return 0.0

        context_words = set(
            w.lower() for w in re.findall(r"\b\w+\b", context) if len(w) > 3
        )
        response_words = set(
            w.lower() for w in re.findall(r"\b\w+\b", response) if len(w) > 3
        )

        if not response_words:
            return 0.0

        overlap = len(context_words & response_words)
        usage_percent = (overlap / len(response_words)) * 100

        return round(usage_percent, 1)

    def calculate_overall_score(self, metrics: Dict[str, Any]) -> int:
        """Calculates an overall score from 0 to 100.

        Args:
            metrics: Dictionary of evaluated metrics.

        Returns:
            The overall score.
        """
        score = 0
        is_trick = metrics.get("is_trick_question", False)

        if metrics["has_fallback"]:
            if is_trick:
                return 100
            else:
                score += 25
        elif is_trick:
            return 0
        else:
            if metrics["has_sources"]:
                score += 15
            if metrics["source_format_correct"]:
                score += 20
            if metrics["multi_intent_complete"]:
                score += 20
            if metrics["hallucination_free"]:
                score += 10
            if metrics.get("answer_complete", True):
                score += 5
            if metrics.get("no_semantic_contradiction", True):
                score += 5
            if metrics.get("has_expected_keywords", True):
                score += 5
            if metrics.get("link_validity", True):
                score += 5
            else:
                score -= 30
            if metrics.get("sources_match", True):
                score += 10
            else:
                score -= 5
            if metrics["instruction_following"]:
                score += 5
            if not metrics.get("required_keywords_present", True):
                score = min(score, 50)

        return min(max(score, 0), 100)

    def generate_evaluation_summary(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generates a summary of all evaluation results.

        Args:
            results: List of evaluation metrics dictionaries.

        Returns:
            Summary dictionary.
        """
        if not results:
            return {}

        total_evaluations = len(results)
        summary = {
            "total_evaluations": total_evaluations,
            "average_score": sum(r["overall_score"] for r in results)
            / total_evaluations,
            "fallback_rate": sum(1 for r in results if r["has_fallback"])
            / total_evaluations,
            "sources_present_rate": sum(1 for r in results if r["has_sources"])
            / total_evaluations,
            "multi_intent_completion_rate": sum(
                1 for r in results if r["multi_intent_complete"]
            )
            / total_evaluations,
            "hallucination_free_rate": sum(
                1 for r in results if r["hallucination_free"]
            )
            / total_evaluations,
            "source_format_correct_rate": sum(
                1 for r in results if r["source_format_correct"]
            )
            / total_evaluations,
            "instruction_following_rate": sum(
                1 for r in results if r["instruction_following"]
            )
            / total_evaluations,
            "average_context_usage": sum(r["context_usage_percent"] for r in results)
            / total_evaluations,
        }

        return summary
