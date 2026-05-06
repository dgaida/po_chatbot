# Deployment on Render.com

This guide describes how to deploy the PO-Chatbot on [Render.com](https://render.com).

## Prerequisites

- A GitHub account containing the chatbot repository.  
- A Render.com account.  
- An externally accessible Ollama endpoint (as Render doesn't offer GPUs for free instances and local LLMs would be too slow or exceed RAM).  

## Blueprint

The repository contains a `render.yaml` which serves as a blueprint. It defines two web services:

1.  **po-chatbot-student**: The interface for students.  
2.  **po-chatbot-admin**: The dashboard for the examination office.  

## Deployment Steps

1.  Log in to Render.com.  
2.  Click **"New +"** and select **"Blueprint"**.  
3.  Connect your GitHub repository.  
4.  Render will automatically detect the `render.yaml` and suggest creating the services.  
5.  Provide the required environment variables (see below).  
6.  Click **"Apply"**.  

## Environment Variables

The following variables must or can be configured:

| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `OLLAMA_URL` | The URL to your Ollama endpoint. | (Placeholder) |
| `MODEL_NAME` | The model to be used. | `qwen2.5:14b` |
| `TEMPERATURE` | Generation creativity. | `0.0` |
| `MAX_CONCURRENT` | Maximum concurrent requests. | `1` |

## Data Persistence

Since Render web services have an ephemeral file system, changes to the `data/` directory (e.g., logs or newly indexed data) will be lost after a restart unless a **Render Disk** is attached.

For production use, it is recommended to:  
- Add a **Render Blueprint Disk** to `render.yaml`.  
- Or index the data locally before deployment and commit it to the repository (not recommended for large databases).  

## Ingestion on Render

If you want to index data on Render, you can open a **Shell** in the Render console and run:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/po_chatbot
python src/po_chatbot/ingest_data.py
```
