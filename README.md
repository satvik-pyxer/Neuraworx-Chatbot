# Neuraworx-Chatbot

A Retrieval-Augmented Generation (RAG) application built with Llama 3.1 that ingests documents, stores them in a vector database, and answers natural language queries. Features a modern web interface for document management and chat interactions with persistent chat history.

## Files Overview

- **api.py**: FastAPI endpoints for ingesting files, querying the RAG system, and managing chat history
- **ingestion.py**: Document processing, embedding, and Qdrant storage functionality
- **model.py**: LLM wrapper for Llama 3.1, including model download and quantization
- **retrieval.py**: Search functionality to retrieve and rank relevant document chunks
- **main.py**: CLI entrypoint for different operations (ingest, serve, prepare-model)
- **templates/index.html**: Web interface for document management and chat interactions
- **chat_history.json**: Persistent storage for chat history
- **document_metadata.json**: Metadata storage for document tracking

## Requirements

- Python 3.8+
- Qdrant vector database (running locally or in the cloud)
- GPU recommended for model inference

## Getting Started

### 1. Get the Model
```bash
# Create the models directory if it doesn't exist (within the repository folder Neuraworx-Chatbot)
mkdir -p models

# Download the quantized model
wget -O models/llama-3.1-8b-instruct-q4_k_m.gguf https://huggingface.co/modularai/Llama-3.1-8B-Instruct-GGUF/resolve/main/llama-3.1-8b-instruct-q4_k_m.gguf
```
### 2. Create Documents Directory

```bash
mkdir -p documents
```

Supported file formats: `.txt`, `.md`, `.pdf`, `.docx`

### 3. Start the Application

```bash
python main.py serve --host 0.0.0.0 --port 8000
```

### 4. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:8000
```

The application will automatically ingest documents placed in the `documents` directory on startup.

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

### Get Documents
```
GET /documents
```
Returns a list of all documents in the system with their metadata.

### Get Chat History
```
GET /chat/history
```
Optionally filter by document_id:
```
GET /chat/history?document_id=your_document_id
```

### Delete Chat History
```
DELETE /chat/history/{chat_id}
```
Delete a specific chat history item by ID.

```
DELETE /chat/history
```
Delete all chat history or filter by document_id:
```
DELETE /chat/history?document_id=your_document_id
```

### Health Check
```
GET /health
```

## Environment Variables

- `QDRANT_URL`: URL of your Qdrant instance (default: `http://localhost:6333`)
- `QDRANT_API_KEY`: Optional API key for Qdrant

## Features

- Modern web interface for document management and chat interactions
- Persistent chat history across sessions
- WebSocket support for real-time document updates
- Automatic document chunking during ingestion
- Smart document management (only embeds new files, removes deleted ones)
- Query optimization based on chat history
- Context-aware retrieval and re-ranking
- Source attribution in responses with document references
- Markdown support for rich text formatting in responses
