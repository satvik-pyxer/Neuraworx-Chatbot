import os
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from ingestion import ingest_file, get_embedder, get_qdrant_client
from retrieval import retrieve_chunks, generate_answer
from model import get_llm, check_dependencies

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = None

class IngestResponse(BaseModel):
    success: bool
    message: str
    files_processed: int
    chunks_created: int

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    model_type: str = "Llama 3.1"

# --- App Setup ---
# Define the FastAPI application with explicit app variable name
app = FastAPI(
    title="RAG API with Llama 3.1",
    description="Retrieval-Augmented Generation API for local documents using Llama 3.1",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---
@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file_endpoint(
    file: UploadFile = File(...),
):
    """
    Ingest a file into the RAG system.
    Supported formats: .txt, .md, .pdf, .docx
    """
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in [".txt", ".md", ".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(await file.read())
        temp_path = Path(temp_file.name)
    
    try:
        # Initialize the embedding model and database client
        embedder = get_embedder()
        qdrant = get_qdrant_client()
        
        # Process the file
        num_chunks = ingest_file(temp_path, qdrant, embedder)
        
        return IngestResponse(
            success=True,
            message=f"Successfully ingested {file.filename}",
            files_processed=1,
            chunks_created=num_chunks
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting file: {str(e)}")
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the RAG system with a question.
    """
    try:
        # Convert chat history to the expected format
        chat_history = None
        if request.chat_history:
            chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.chat_history
            ]
        
        # Retrieve relevant chunks
        chunks = retrieve_chunks(request.query, chat_history)
        
        # Generate answer
        answer = generate_answer(request.query, chunks, chat_history)
        
        # Prepare sources information
        sources = [
            {
                "text": chunk["text"],
                "filename": chunk["metadata"].get("filename", "Unknown"),
                "score": chunk.get("rerank_score", chunk.get("score", 0))
            }
            for chunk in chunks
        ]
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint that also reports the status of components.
    """
    # Check if model is loaded or can be loaded
    model_loaded = False
    try:
        # Try to initialize the model (will use cached instance if already loaded)
        get_llm()
        model_loaded = True
    except Exception:
        pass
    
    # Check if database is connected
    db_connected = False
    try:
        qdrant = get_qdrant_client()
        # Simple operation to check connection
        qdrant.get_collections()
        db_connected = True
    except Exception:
        pass
    
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        database_connected=db_connected,
        model_type="Llama 3.1"  # Specify we're using Llama 3.1
    )

# Add a root endpoint for easy testing
@app.get("/")
async def root():
    """
    Root endpoint to verify the API is running.
    """
    return JSONResponse(
        content={
            "message": "RAG API with Llama 3.1 is running. Use /docs to see available endpoints.",
            "documentation": "/docs"
        }
    )

# This is important: DO NOT include the server starter code here
# The app instance should just be defined, not started
# Leave the server starting to main.py

# Export the app variable explicitly at the module level
__all__ = ["app"]