# Docstring Guide

To ensure consistent API documentation, we use the Google style for docstrings.

## Example

Here is an example of how a function should be documented:

```python
def search(self, question: str, faculty_filter: str, top_k: int = 5) -> dict:
    """Performs a hybrid search: Vector + BM25, followed by re-ranking.

    Args:
        question (str): The student's question to search for.
        faculty_filter (str): Filter for the faculty (e.g., 'F10').
        top_k (int): Number of results to return. Defaults to 5.

    Returns:
        dict: A dictionary with the keys 'documents' and 'metadatas'.

    Example:
        >>> engine = HybridRetrievalEngine()
        >>> res = engine.search("When is the exam?", "F10")
    """
```

## Requirements

1.  **Summary**: A short one-line description at the beginning.  
2.  **Args**: List of all parameters with type and description.  
3.  **Returns**: Description of the return value and type.  
4.  **Raises**: (Optional) If the function raises specific exceptions.  
5.  **Example**: (Optional) A short code example for usage.  
