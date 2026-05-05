# Development

This document describes the guidelines for contributing to the PO-Chatbot.

## Local Setup

1. Install developer dependencies:  
   ```bash
   pip install -e .
   pip install black ruff pytest interrogate
   ```
2. Enable git hooks (if available) or run checks manually.  

## Code Style

We use **Black** for formatting and **Ruff** for linting.  
- Formatting: `black src evaluation`  
- Linting: `ruff check src evaluation`  

## Documentation

- Use Google-style docstrings for all public functions and classes.  
- Documentation is built with MkDocs. Run `mkdocs serve` to preview locally.  

## Testing

Current tests are located in the `evaluation/` directory. New unit tests should be added to the `tests/` directory (to be created).
