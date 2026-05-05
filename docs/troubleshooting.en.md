# Troubleshooting

Common issues and their solutions.

## Ollama Connection Issues

**Error**: `Error: Ollama is not reachable.`

*   **Solution**: Ensure that Ollama is running in the background (`ollama serve`). Check the status with `curl http://localhost:11434/api/tags`.  
*   **Solution**: If Ollama is running on a different port or machine, adjust the `OLLAMA_URL` in the scripts.  

## No Documents Found

**Error**: `No relevant documents could be found.`

*   **Cause**: The database has not been initialized.  
*   **Solution**: Run `python src/po_chatbot/ingest_data.py`. Check if there are Markdown files in `data/text_extracted`.  
*   **Cause**: The faculty or study program filter is too restrictive.  
*   **Solution**: Try using a more general setting.  

## Hallucinations in Responses

**Problem**: The chatbot invents facts or provides incorrect deadlines.

*   **Solution**: Increase the `REPEAT_PENALTY` or decrease the `TEMPERATURE`.  
*   **Solution**: Check the quality of the extracted texts in `data/chunks.json`. Use the admin interface for corrections.  

## Slow Response Generation

**Problem**: Response takes several minutes.

*   **Solution**: The PO-Chatbot runs locally on your CPU/GPU. A system with at least 16GB RAM and a dedicated GPU is recommended for the 14B model. Consider using a smaller model (e.g., `qwen2.5:7b`).  
