# Neuraworx-Chatbot

A Retrieval-Augmented Generation (RAG) API built with Llama 3.1 that ingests documents, stores them in a vector database, and answers natural language queries.

## Files Overview

- **api.py**: FastAPI endpoints for ingesting files and querying the RAG system
- **ingestion.py**: Document processing, embedding, and Qdrant storage functionality
- **model.py**: LLM wrapper for Llama 3.1, including model download and quantization
- **retrieval.py**: Search functionality to retrieve and rank relevant document chunks
- **main.py**: CLI entrypoint for different operations (ingest, serve, prepare-model)

## Requirements

- Python 3.8+
- Qdrant vector database (running locally or in the cloud)
- GPU recommended for model inference

## Getting Started

### 1. Prepare the Model

The system uses Llama 3.1 (8B Instruct version). Download and quantize the model:

```bash
python main.py prepare-model
```

### 2. Ingest Documents

Process documents from a directory:

```bash
python main.py ingest /path/to/documents
```

Supported file formats: `.txt`, `.md`, `.pdf`, `.docx`

### 3. Start the API Server

```bash
python main.py serve --host 0.0.0.0 --port 8000
```

## API Usage

### Ingest a File
```
POST /ingest/file
```
Upload a file using multipart/form-data.

### Query the System
```
POST /query
```
Body:
```json
{
  "query": "Your question here",
  "chat_history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```

### Health Check
```
GET /health
```

## Environment Variables

- `QDRANT_URL`: URL of your Qdrant instance (default: `http://localhost:6333`)
- `QDRANT_API_KEY`: Optional API key for Qdrant

## Features

- Automatic document chunking during ingestion
- Query optimization based on chat history
- Context-aware retrieval and re-ranking
- Source attribution in responses
