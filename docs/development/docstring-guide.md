# Docstring-Leitfaden

Um eine konsistente API-Dokumentation zu gewährleisten, nutzen wir den Google-Style für Docstrings.

## Beispiel

Hier ist ein Beispiel, wie eine Funktion dokumentiert werden sollte:

```python
def search(self, question: str, faculty_filter: str, top_k: int = 5) -> dict:
    """Führt eine hybride Suche durch: Vektor + BM25, gefolgt von Re-Ranking.

    Args:
        question (str): Die zu suchende Frage des Studierenden.
        faculty_filter (str): Filter für die Fakultät (z.B. 'F10').
        top_k (int): Anzahl der zurückzugebenden Ergebnisse. Defaults to 5.

    Returns:
        dict: Ein Dictionary mit den Schlüsseln 'documents' und 'metadatas'.

    Example:
        >>> engine = HybridRetrievalEngine()
        >>> res = engine.search("Wann ist die Prüfung?", "F10")
    """
```

## Anforderungen

1.  **Zusammenfassung**: Eine kurze Einzeilen-Beschreibung am Anfang.  
2.  **Args**: Liste aller Parameter mit Typ und Beschreibung.  
3.  **Returns**: Beschreibung des Rückgabewerts und Typs.  
4.  **Raises**: (Optional) Falls die Funktion spezifische Exceptions wirft.  
5.  **Example**: (Optional) Ein kurzes Code-Beispiel zur Nutzung.  
