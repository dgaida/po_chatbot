# Lokale Evaluierungs-Metriken für RAG-System

import re
import json
from typing import Dict, List, Any, Tuple

class RAGEvaluator:
    
    def __init__(self):
        self.fallback_phrase = "Dazu liegen mir keine Informationen vor."
        
    def evaluate_response(
        self,
        question: str,
        response: str,
        context: str,
        expected_fallback: bool = False,
        multi_intent_parts: int = 1,
        question_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        # Haupt-Evaluierungsfunktion mit allen Metriken.

        if question_data is None:
            question_data = {
                "question": question,
                "multi_intent_parts": multi_intent_parts
            }
        
        metrics = {
            "question": question,
            "response": response,
            "response_length": len(response),
            "has_fallback": self.check_fallback_usage(response),
            "has_sources": self.check_sources_present(response),
            "source_count": self.count_sources(response),
            "multi_intent_complete": self.check_multi_intent_completion(question, response, multi_intent_parts, question_data),
            "hallucination_free": self.check_hallucination_prevention(context, response),
            "source_format_correct": self.check_source_format(response),
            "instruction_following": self.check_instruction_following(response),
            "context_usage_percent": self.check_context_usage(context, response),
            "answer_complete": self.check_answer_completeness(response, question_data),
            "no_semantic_contradiction": self.check_semantic_contradiction(response, question_data),
            "has_expected_keywords": self.check_expected_keywords(response, question_data),
            "required_keywords_present": self.check_required_keywords(response, question_data),
            "link_validity": self.check_link_validity(response),
            "sources_match": self.check_expected_sources(response, question_data),
            "is_trick_question": question_data.get("is_trick_question", False),
            "overall_score": 0
        }
        
        # Berechne Gesamtscore (0-100)
        metrics["overall_score"] = self.calculate_overall_score(metrics)
        
        return metrics
    
    def check_fallback_usage(self, response: str) -> bool:
        # Prüft, ob der korrekte Fallback-Satz verwendet wurde.
        return self.fallback_phrase in response.strip()
    
    def check_sources_present(self, response: str) -> bool:
        # Prüft, ob Quellen vorhanden sind (außer bei Fallback).
        if self.check_fallback_usage(response):
            return True  # Bei Fallback sind keine Quellen nötig
        
        # Suche nach Quellen-Links
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, response)
        return len(urls) > 0
    
    def count_sources(self, response: str) -> int:
        # Zählt die Anzahl der Quellen-Links.
        if self.check_fallback_usage(response):
            return 0
            
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, response)
        return len(urls)
    
    def check_multi_intent_completion(self, question: str, response: str, expected_parts: int, question_data: Dict[str, Any] = None) -> bool:
        # Prüft, ob ALLE Teilfragen einer mehrteiligen Frage beantwortet wurden.
        # Striktere Prüfung: Erkennt fehlende Teilantworten bei konkreten Sub-Fragen.
        if expected_parts <= 1:
            return True

        if self.check_fallback_usage(response):
            return False

        question_lower = question.lower()
        response_lower = response.lower()

        # Strikte Sub-Fragen-Erkennung
        sub_q_checks = []

        # "Wo finde ich...?" / "Wo...?" -> Braucht Link/Formular/Quelle
        if "wo finde" in question_lower or ("wo" in question_lower and "formular" in question_lower):
            has_location = any(x in response_lower for x in ["http", "formular", "antrag", "download", "link"])
            sub_q_checks.append(has_location)

        # "Ab wann...?" / "Wenn ja, ab wann...?" -> Braucht Datum/Zeitangabe
        if "ab wann" in question_lower or "wann gilt" in question_lower:
            has_date = bool(re.search(r'\b(20\d{2}|ws|ss|semester|wintersemester|sommersemester)\b', response_lower))
            sub_q_checks.append(has_date)

        # "Wie viele...?" / "Wie lange...?" -> Braucht Zahl
        if "wie viele" in question_lower or "wie lange" in question_lower:
            has_number = bool(re.search(r'\d+', response))
            sub_q_checks.append(has_number)

        # "Gibt es...? Wenn ja,..." -> Braucht Bestätigung + Details
        if "gibt es" in question_lower and "wenn ja" in question_lower:
            has_confirmation = any(x in response_lower for x in ["ja", "nein", "existiert", "gibt es", "vorhanden"])
            sub_q_checks.append(has_confirmation)

        # Wenn Sub-Checks vorhanden: Alle müssen bestanden werden
        if sub_q_checks:
            return all(sub_q_checks)

        # Fallback: Satzanzahl-Heuristik
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        return len(sentences) >= expected_parts
    
    def check_hallucination_prevention(self, context: str, response: str) -> bool:
        # Prüft, ob die Antwort nur Informationen aus dem Kontext verwendet.
        # Einfache Heuristik basierend auf Schlüsselwörtern.
        if self.check_fallback_usage(response):
            return True

        context_l = (context or "").lower()
        response_s = (response or "")
        response_l = response_s.lower()

        # URLs aus der Antwort entfernen bevor Fakten extrahiert werden.
        response_no_urls = re.sub(r'https?://[^\s<>"]+', '', response_s)

        # Kritische Fakten: Zahlen, Paragraphen-Referenzen und datumsähnliche Tokens.
        # Heuristik: Wenn die Antwort kritische Fakten enthält, die nicht im Kontext stehen,
        # wird dies als potenzielle Halluzination gewertet.
        facts = set()

        # Paragraphen-Referenzen (z.B. "§ 12", "§12")
        facts.update(m.group(0).strip().lower() for m in re.finditer(r'§\s*\d+[a-z]?', response_no_urls, flags=re.IGNORECASE))

        # Datums-Muster (z.B. 01.09.2020, 1.9.2020)
        facts.update(m.group(0).strip().lower() for m in re.finditer(r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b', response_no_urls))

        # Einzelne Zahlen / Dezimalwerte (z.B. 150, 30, 9, 1,5)
        # Sehr kleine Zahlen überspringen, die häufig in Listenformatierung vorkommen.
        for m in re.finditer(r'\b\d{1,4}([\.,]\d+)?\b', response_no_urls):
            tok = m.group(0).strip().lower()
            # Triviale 0/1 überspringen (häufig als Aufzählungszeichen).
            if tok in {"0", "1"}:
                continue
            facts.add(tok)

        if not facts:
            return True

        # Kommas/Punkte für numerische Vergleiche normalisieren.
        def _num_variants(t: str) -> set[str]:
            if re.fullmatch(r'\d{1,4}([\.,]\d+)?', t):
                return {t, t.replace(',', '.'), t.replace('.', ',')}
            return {t}

        for fact in facts:
            if fact.startswith("§"):
                if fact not in context_l:
                    return False
                continue

            if re.fullmatch(r'\d{1,2}\.\d{1,2}\.\d{2,4}', fact):
                if fact not in context_l:
                    return False
                continue

            variants = _num_variants(fact)
            if not any(v in context_l for v in variants):
                return False

        return True
    
    def check_source_format(self, response: str) -> bool:
        # Prüft, ob Quellen korrekt formatiert sind (nicht im Fließtext).
        if self.check_fallback_usage(response):
            return True
            
        # Prüfe auf "Quellen:" Block
        has_sources_block = "Quellen:" in response or "Quelle:" in response
        
        # Prüfe, ob Links am Ende stehen (nach Quellen-Block)
        if has_sources_block:
            sources_section = response.split("Quellen:")[-1] if "Quellen:" in response else response.split("Quelle:")[-1]
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls_in_sources = re.findall(url_pattern, sources_section)
            
            # Alle Links sollten im Quellen-Block sein
            total_urls = re.findall(url_pattern, response)
            return len(urls_in_sources) == len(total_urls)
        
        return False
    
    def check_instruction_following(self, response: str) -> bool:
        # Prüft, ob die strukturellen Anweisungen befolgt wurden.
        # Erkennt auch interne RAG-Metadaten-Leaks ("Dokument 1", "Text-Chunk").
        if self.check_fallback_usage(response):
            return True
            
        # Interne Metadaten-Leak: Modell referenziert RAG-Chunk-Namen statt Quellen-Links
        leak_patterns = [
            r'(?:in|laut|aus|gemäß)\s+dokument\s+\d',
            r'text[- ]?chunk\s+\d',
            r'dokument\s+\d\s+(?:steht|sagt|beschreibt|enthält|zeigt)',
            r'(?:im|aus dem)\s+(?:ersten|zweiten|dritten|vierten|fünften)\s+dokument',
        ]
        response_lower = response.lower()
        for pattern in leak_patterns:
            if re.search(pattern, response_lower):
                return False

        # Quellen-Block vorhanden und korrekt formatiert
        has_sources_block = "Quellen:" in response or "Quelle:" in response
        has_no_sources_in_text = self.check_source_format(response)
        
        return has_sources_block and has_no_sources_in_text
    
    def check_answer_completeness(self, response: str, question_data: Dict[str, Any]) -> bool:
        # Prüft, ob bei Multi-Intent-Fragen beide Teile beantwortet wurden.
        if self.check_fallback_usage(response):
            return True
        
        multi_intent_parts = question_data.get("multi_intent_parts", 1)
        if multi_intent_parts <= 1:
            return True
        
        question = question_data.get("question", "").lower()
        response_lower = response.lower()
        
        # Prüfe auf gängige Multi-Intent-Muster
        completeness_checks = []
        
        # Muster 1: "Wo finde ich...?" -> braucht URL oder "Formular" oder "Antrag"
        if "wo finde" in question or "wo gibt" in question:
            has_location = any(x in response_lower for x in ["http", "formular", "antrag", "link", "download"])
            completeness_checks.append(has_location)
        
        # Muster 2: "Wie viele...?" -> braucht Zahl
        if "wie viele" in question or "wie lange" in question:
            has_number = bool(re.search(r'\d+', response))
            completeness_checks.append(has_number)
        
        # Muster 3: "Welche...?" -> braucht Liste oder Aufzählung
        if "welche" in question and "unterschied" in question:
            # Mindestens 2 verschiedene Punkte erforderlich
            has_enumeration = response.count("-") >= 2 or response.count("\n") >= 2
            completeness_checks.append(has_enumeration)
        
        # Wenn spezifische Prüfungen vorhanden, diese nutzen; sonst als vollständig werten
        if completeness_checks:
            return all(completeness_checks)
        
        return True
    
    def check_semantic_contradiction(self, response: str, question_data: Dict[str, Any]) -> bool:
        # Prüft auf semantische Widersprüche in der Antwort.
        # Erkennt identische Werte in Vergleichen (z.B. gleiche CP bei Unterschied-Fragen).
        if self.check_fallback_usage(response):
            return True
        
        question = question_data.get("question", "").lower()
        response_lower = response.lower()
        
        # Prüfung 1: Frage nach Unterschieden, aber Antwort zeigt gleiche Werte
        # Auch "vergleich", "unterschied", "differenz" erkennen
        if any(word in question for word in ["unterschied", "vergleich", "differenz", "zwei version"]):
            # Zahlen mit Einheiten aus der Antwort extrahieren
            numbers = re.findall(r'\b(\d+)\s*(cp|ects|lp|leistungspunkte|wochen?|monate?|semester)\b', response_lower)
            if len(numbers) >= 2:
                # Prüfe ob alle Zahlen mit gleicher Einheit identisch sind
                values = [int(n[0]) for n in numbers]
                if len(set(values)) == 1:  # Alle gleich -> Widerspruch
                    return False
            
            # Auch auf identische Zeiträume prüfen (z.B. "9 Wochen vs 9 Wochen")
            weeks = re.findall(r'\b(\d+)\s*wochen?\b', response_lower)
            if len(weeks) >= 2 and len(set(weeks)) == 1:
                return False
        
        # Prüfung 2: Frage nach "zwei Versionen", aber Antwort nennt gleiche Version doppelt
        if "zwei version" in question or "welche version" in question or "verfügbar" in question:
            # PO-Erwähnung prüfen
            po_mentions = re.findall(r'po\s*[0-9]+', response_lower)
            if len(po_mentions) >= 2 and len(set(po_mentions)) == 1:
                return False  # Gleiche PO mehrfach erwähnt
        
        # Prüfung 3: Grammatische Inkohärenz (Passiv ohne Subjekt)
        # "schriftlich dargelegt sowie glaubhaft nachgewiesen werden"
        passive_without_subject = re.search(r'(dargelegt|nachgewiesen|eingereicht)\s+(sowie|und|oder)\s+\w+\s+werden[,.]', response_lower)
        if passive_without_subject and not re.search(r'(müssen|muss|sollte|ist)\s+\w+\s+(dargelegt|nachgewiesen)', response_lower):
            return False
        
        return True
    
    def check_expected_sources(self, response: str, question_data: Dict[str, Any]) -> bool:
        # Prüft, ob die erwarteten Quellen-URLs in der Antwort zitiert werden.
        # Vergleicht Dateinamen (letzter Teil der URL) statt voller URL.
        expected = question_data.get("expected_sources", [])
        if not expected:
            return True  

        if self.check_fallback_usage(response):
            return False  

        response_lower = response.lower()
        # Extrahiere alle URLs aus der Antwort
        found_urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', response)
        found_lower = [u.lower() for u in found_urls]

        # Bestimme Modus: Wenn multi_intent_parts >= 2 und mehrere Quellen erwartet,
        # müssen alle vorhanden sein. Sonst reicht eine (alternative akzeptable Quellen).
        multi_intent = question_data.get("multi_intent_parts", 1)
        require_all = multi_intent >= 2 and len(expected) >= 2

        if require_all:
            # Alle erwarteten Quellen müssen in der Antwort vorkommen
            for exp_url in expected:
                exp_filename = exp_url.rstrip('/').split('/')[-1].lower()
                if not any(exp_filename in fu for fu in found_lower):
                    return False
            return True
        else:
            # mindestens eine der erwarteten Quellen muss vorkommen
            for exp_url in expected:
                exp_filename = exp_url.rstrip('/').split('/')[-1].lower()
                if any(exp_filename in fu for fu in found_lower):
                    return True
            return False

    def check_link_validity(self, response: str) -> bool:
        # Prüft, ob Links korrekt formatiert sind und nicht leer sind.
        # Erkennt: [], [Link], [](), und Phantom-Links.
        if self.check_fallback_usage(response):
            return True
        
        # Prüfe auf leere Markdown-Links
        empty_links = re.findall(r'\[\]|\[Link\]|\[\]\(\)', response)
        if empty_links:
            return False
        
        # Prüfe auf Links mit leeren URLs
        empty_url_links = re.findall(r'\[[^\]]+\]\(\s*\)', response)
        if empty_url_links:
            return False

        # Phantom-Link: Modell verspricht einen Link der nicht existiert
        response_lower = response.lower()
        phantom_patterns = [
            r'unter\s+folgend\w*\s+link',
            r'folgend\w*\s+links?\s*:',
            r'\[link\s*\d*\s*einfügen\]',
            r'hier\s+ist\s+der\s+link',
            r'link\s+zum\s+download\s*:',
        ]
        # Nur bestrafen wenn kein tatsächlicher URL in der Nähe steht
        for pat in phantom_patterns:
            match = re.search(pat, response_lower)
            if match:
                # Prüfe ob innerhalb von 200 Zeichen nach dem Match eine URL folgt
                after = response[match.end():match.end()+200]
                if not re.search(r'https?://', after):
                    return False
        
        return True
    
    def check_expected_keywords(self, response: str, question_data: Dict[str, Any]) -> bool:
        # Prüft, ob erwartete Keywords in der Antwort vorhanden sind (70%-Schwelle).
        expected = question_data.get("expected_answer_keywords", [])
        if not expected:
            return True 
        
        if self.check_fallback_usage(response):
            return True
        
        response_lower = response.lower()
        
        found = sum(1 for kw in expected if kw.lower() in response_lower)
        threshold = max(1, int(len(expected) * 0.7))  # Mindestens 70% oder Minimum 1
        
        return found >= threshold

    def check_required_keywords(self, response: str, question_data: Dict[str, Any]) -> bool:
        # Strikte Prüfung: alle required_keywords müssen in der Antwort vorkommen.
        required = question_data.get("required_keywords", [])
        if not required:
            return True
        
        if self.check_fallback_usage(response):
            return question_data.get("is_trick_question", False)
        
        response_lower = response.lower()
        return all(kw.lower() in response_lower for kw in required)
    
    def check_context_usage(self, context: str, response: str) -> float:
        # Misst den Prozentsatz des Kontexts, der in der Antwort verwendet wird.
        if self.check_fallback_usage(response):
            return 0.0  # Bei Fallback wird kein Kontext verwendet
            
        # Extrahiere bedeutungsvolle Wörter (länger als 3 Zeichen)
        context_words = set(w.lower() for w in re.findall(r'\b\w+\b', context) if len(w) > 3)
        response_words = set(w.lower() for w in re.findall(r'\b\w+\b', response) if len(w) > 3)
        
        if not response_words:
            return 0.0
            
        # Berechne Überlappung (Grounding/Precision): Wie viel der Antwort ist im Kontext verankert?
        overlap = len(context_words & response_words)

        if response_words:
            usage_percent = (overlap / len(response_words)) * 100
        else:
            usage_percent = 0.0
            
        return round(usage_percent, 1)
    
    def calculate_overall_score(self, metrics: Dict[str, Any]) -> int:
        # Berechnet einen Gesamtscore von 0-100.
        # Fallback wird bei Trick-Fragen belohnt (= 100 Punkte).
        score = 0
        
        # Prüfe ob es eine Trickfrage ist (unbeantwortbar)
        is_trick = metrics.get("is_trick_question", False)
        
        # Fallback-Handling (25 Punkte)
        if metrics["has_fallback"]:
            if is_trick:
                return 100
            else:
                score += 25
        elif is_trick:
            return 0
        else:
            # Quellen vorhanden (15 Punkte)
            if metrics["has_sources"]:
                score += 15
            
            # Quellen-Format korrekt (20 Punkte)
            if metrics["source_format_correct"]:
                score += 20
            
            # Multi-Intent vollständig (20 Punkte)
            if metrics["multi_intent_complete"]:
                score += 20
            
            # Keine Halluzinationen (10 Punkte)
            if metrics["hallucination_free"]:
                score += 10
            
            # Antwort-Vollständigkeit (5 Punkte)
            if metrics.get("answer_complete", True):
                score += 5
            
            # Keine semantischen Widersprüche (5 Punkte)
            if metrics.get("no_semantic_contradiction", True):
                score += 5
            
            # Erwartete Schlüsselwörter (5 Punkte)
            if metrics.get("has_expected_keywords", True):
                score += 5
            
            # Link Validity (5 Punkte)
            if metrics.get("link_validity", True):
                score += 5
            else:
                score -= 30  # Strafe für leere/kaputte Links

            # Sources Match (10 Punkte) - Korrekte Quellen zitiert
            if metrics.get("sources_match", True):
                score += 10
            else:
                score -= 5  # Falsche/fehlende Quellen
            
            # Instruction Following (5 Punkte)
            if metrics["instruction_following"]:
                score += 5
            
            # Harte Strafe: Pflicht-Keywords fehlen = faktisch falsche Antwort
            if not metrics.get("required_keywords_present", True):
                score = min(score, 50)  # Deckelt Score bei 50
        
        return min(score, 100)  # Maximal 100 Punkte
    
    def generate_evaluation_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Generiert eine Zusammenfassung aller Evaluierungsergebnisse.
        if not results:
            return {}
            
        total_evaluations = len(results)
        summary = {
            "total_evaluations": total_evaluations,
            "average_score": sum(r["overall_score"] for r in results) / total_evaluations,
            "fallback_rate": sum(1 for r in results if r["has_fallback"]) / total_evaluations,
            "sources_present_rate": sum(1 for r in results if r["has_sources"]) / total_evaluations,
            "multi_intent_completion_rate": sum(1 for r in results if r["multi_intent_complete"]) / total_evaluations,
            "hallucination_free_rate": sum(1 for r in results if r["hallucination_free"]) / total_evaluations,
            "source_format_correct_rate": sum(1 for r in results if r["source_format_correct"]) / total_evaluations,
            "instruction_following_rate": sum(1 for r in results if r["instruction_following"]) / total_evaluations,
            "average_context_usage": sum(r["context_usage_percent"] for r in results) / total_evaluations
        }
        
        return summary
