import os
import tempfile
import requests
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, BackgroundTasks, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import json

from ingestion import ingest_file, get_embedder, get_qdrant_client
from retrieval import retrieve_chunks, generate_answer
from model import get_llm, check_dependencies, reload_llm
from models_config import ModelManager, AVAILABLE_MODELS
from qdrant_client import models

# Import GPT4All for model downloads
try:
    from gpt4all import GPT4All
    GPT4ALL_AVAILABLE = True
except ImportError:
    GPT4ALL_AVAILABLE = False

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
    id: Optional[str] = None

class ChatHistoryItem(BaseModel):
    id: str
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    timestamp: str
    document_id: Optional[str] = None

class ChatHistoryResponse(BaseModel):
    history: List[ChatHistoryItem]

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

# Set up static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize model manager
model_manager = ModelManager()

# Global storage for documents metadata and chat history
documents_metadata = {}
chat_history = []

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # If sending fails, we'll handle it during the next operation
                pass

# Initialize the connection manager
manager = ConnectionManager()

# Function to save chat history to file
def save_chat_history():
    history_file = Path("chat_history.json")
    try:
        with open(history_file, "w") as f:
            json.dump({"history": chat_history}, f, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")

# Function to load chat history from file
def load_chat_history():
    global chat_history
    history_file = Path("chat_history.json")
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                data = json.load(f)
                chat_history = data.get("history", [])
        except Exception as e:
            print(f"Error loading chat history: {e}")
            chat_history = []
    else:
        chat_history = []

# Function to update document metadata
def update_document_metadata(doc_id: str, filename: str, chunks: int, size: int):
    # Load existing metadata from file
    metadata_file = Path("document_metadata.json")
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                metadata = {"documents": {}}
    else:
        metadata = {"documents": {}}
    
    # Update metadata for this document
    metadata["documents"][filename] = {
        "id": doc_id,
        "filename": filename,
        "chunks": chunks,
        "size": size,
        "date_added": str(datetime.now())
    }
    
    # Also update in-memory dictionary for backward compatibility
    documents_metadata[doc_id] = {
        "filename": filename,
        "chunks": chunks,
        "size": size,
        "date_added": str(datetime.now())
    }
    
    # Save metadata to file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

# --- Endpoints ---
@app.websocket("/ws/documents")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive, waiting for messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

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
    
    # Create documents directory if it doesn't exist
    documents_dir = Path("documents")
    documents_dir.mkdir(exist_ok=True)
    
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
        
        # Generate a unique ID for the document
        import uuid
        from datetime import datetime
        doc_id = str(uuid.uuid4())
        
        # Save the file to the documents directory
        target_path = documents_dir / file.filename
        shutil.copy2(temp_path, target_path)
        
        # Store document metadata
        file_size = os.path.getsize(temp_path)
        update_document_metadata(doc_id, file.filename, num_chunks, file_size)
        
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
        formatted_chat_history = None
        if request.chat_history:
            formatted_chat_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.chat_history
            ]
        
        # Retrieve relevant chunks
        chunks = retrieve_chunks(request.query, formatted_chat_history)
        
        # Generate answer
        answer = generate_answer(request.query, chunks, formatted_chat_history)
        print("Query: \n", request.query, "\nLLM Response: \n", answer)

        # Prepare sources information
        sources = [
            {
                "text": chunk["text"],
                "filename": chunk["metadata"].get("filename", "Unknown"),
                "score": chunk.get("rerank_score", chunk.get("score", 0))
            }
            for chunk in chunks
        ]
        
        # Create a unique ID for this chat item
        chat_id = f"chat_{int(datetime.now().timestamp())}_{hash(request.query)%10000}"
        
        # Store in chat history
        history_item = ChatHistoryItem(
            id=chat_id,
            question=request.query,
            answer=answer,
            sources=sources,
            timestamp=str(datetime.now()),
            document_id=sources[0]["filename"] if sources else None
        )
        
        # Add to in-memory history
        chat_history.append(history_item.dict())
        
        # Save history to file
        save_chat_history()
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            id=chat_id
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

# Frontend routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """
    Serve the main frontend page
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    """
    Serve the admin page
    """
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request):
    """
    Serve the profile page
    """
    return templates.TemplateResponse("profile.html", {"request": request})

@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    """
    Serve the help page
    """
    return templates.TemplateResponse("help.html", {"request": request})

@app.get("/health-page", response_class=HTMLResponse)
async def health_status_page(request: Request):
    """
    Serve the health status page
    """
    return templates.TemplateResponse("status.html", {"request": request})

# Document metadata endpoint
@app.get("/documents")
async def get_documents():
    """
    Get metadata for all ingested documents
    """
    # Load metadata from file
    metadata_file = Path("document_metadata.json")
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            try:
                metadata = json.load(f)
                # Return the documents section of the metadata
                return metadata.get("documents", {})
            except json.JSONDecodeError:
                pass
    
    # If no metadata file or error reading it, fall back to in-memory metadata
    # First rescan the documents directory to ensure we have the latest files
    await scan_documents_directory(auto_ingest=False)
    
    # Convert documents_metadata to the format expected by the frontend
    result = {}
    for doc_id, doc_info in documents_metadata.items():
        filename = doc_info.get("filename")
        if filename:
            result[filename] = {
                "id": doc_id,
                **doc_info
            }
    
    return result

# Model management endpoints
@app.get("/admin/models")
async def get_models():
    """
    Get model status information
    """
    return model_manager.get_model_status()

@app.post("/admin/models/activate")
async def activate_model(model_data: dict):
    """
    Activate a model
    """
    try:
        model_filename = model_data["model_filename"]
        if not model_manager.is_model_downloaded(model_filename):
            raise HTTPException(status_code=400, detail="Model not downloaded")
        
        # Update active model in config
        model_manager.set_active_model(model_filename)
        
        # Reinitialize the model
        reload_llm(model_filename)
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/models/download")
async def download_model(model_data: dict):
    """
    Download a model
    """
    try:
        model_filename = model_data["model_filename"]
        model_info = model_manager.get_model_info(model_filename)
        
        if not model_info:
            raise HTTPException(status_code=400, detail="Invalid model")
        
        if model_manager.is_model_downloaded(model_filename):
            return {"status": "already_downloaded"}
        
        # Download the model using GPT4All if available
        if GPT4ALL_AVAILABLE:
            try:
                # Initialize a temporary model instance to trigger download
                # GPT4All will automatically download the model to the specified path
                temp_model = GPT4All(model_filename, model_path=str(model_manager.models_dir))
                return {"status": "success"}
            except Exception as gpt4all_error:
                # Log the GPT4All error and fall back to manual download
                print(f"GPT4All download failed: {str(gpt4all_error)}. Falling back to manual download.")
        
        # Fallback: Manual download using requests
        response = requests.get(model_info.download_url, stream=True)
        if response.status_code == 200:
            with open(model_manager.models_dir / model_filename, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            return {"status": "success"}
        else:
            raise HTTPException(status_code=response.status_code, 
                                detail=f"Failed to download model: {response.reason}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to get document statistics
@app.get("/admin/stats")
async def get_document_stats():
    """
    Get statistics about documents and chunks
    """
    # Rescan documents directory
    await scan_documents_directory(auto_ingest=False)
    
    # Calculate total documents and chunks
    total_documents = len(documents_metadata)
    total_chunks = sum(doc.get("chunks", 0) for doc in documents_metadata.values())
    total_size = sum(doc.get("size", 0) for doc in documents_metadata.values())
    
    # Get Qdrant collection stats if possible
    qdrant_stats = {}
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        if any(c.name == "rag_chunks" for c in collections):
            collection_info = client.get_collection("rag_chunks")
            qdrant_stats = {
                "vector_count": collection_info.vectors_count,
                "dimension": collection_info.config.params.vector_size,
                "collection_name": "rag_chunks"
            }
    except Exception as e:
        print(f"Error getting Qdrant stats: {e}")
    
    return {
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "total_size_bytes": total_size,
        "qdrant": qdrant_stats
    }

# Function to scan documents directory and add existing documents to metadata
async def scan_documents_directory(auto_ingest: bool = False):
    documents_dir = Path("documents")
    if not documents_dir.exists():
        documents_dir.mkdir(exist_ok=True)
        return []
    
    # Load metadata from file
    metadata_file = Path("document_metadata.json")
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                metadata = {"documents": {}}
    else:
        metadata = {"documents": {}}
    
    # Get all document files in the documents directory
    doc_files = list(documents_dir.glob("*.pdf")) + \
               list(documents_dir.glob("*.txt")) + \
               list(documents_dir.glob("*.docx")) + \
               list(documents_dir.glob("*.md"))
    
    # Get list of filenames in the documents directory
    doc_file_names = {doc_file.name for doc_file in doc_files}
    
    # Get list of filenames in the metadata
    metadata_files = set(metadata.get("documents", {}).keys())
    
    # Find files to remove (in metadata but not in directory)
    files_to_remove = metadata_files - doc_file_names
    
    # Remove files from metadata that no longer exist in the directory
    for filename in files_to_remove:
        if filename in metadata["documents"]:
            del metadata["documents"][filename]
    
    # Check which files have embeddings in Qdrant
    files_with_embeddings = set()
    try:
        # Get Qdrant client
        qdrant = get_qdrant_client()
        
        # Check if the collection exists
        collections = qdrant.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if "rag_chunks" in collection_names:
            # For each file, check if it has embeddings
            for filename in doc_file_names:
                filter_query = {
                    "must": [
                        {
                            "key": "filename",
                            "match": {"value": filename}
                        }
                    ]
                }
                
                # Search for points with this filename
                search_result = qdrant.scroll(
                    collection_name="rag_chunks",
                    filter=filter_query,
                    limit=1  # We only need to know if any exist
                )
                
                # If points found, file has embeddings
                if search_result[0]:
                    files_with_embeddings.add(filename)
    except Exception as e:
        print(f"Error checking Qdrant: {e}")
        # If Qdrant check fails, assume all files have embeddings
        files_with_embeddings = doc_file_names
    
    # Update metadata for all files
    for doc_file in doc_files:
        # Check if file is already in metadata
        if doc_file.name in metadata["documents"]:
            # Update existing metadata if needed
            doc_info = metadata["documents"][doc_file.name]
            doc_id = doc_info.get("id", str(doc_file.stem))
            
            # Update size if it changed
            current_size = doc_file.stat().st_size
            if doc_info.get("size", 0) != current_size:
                doc_info["size"] = current_size
                
            # Update chunks count if file has embeddings but chunks is 0
            if doc_file.name in files_with_embeddings and doc_info.get("chunks", 0) == 0:
                # Estimate chunks based on file size
                estimated_chunks = max(1, current_size // 2000)  # Rough estimate
                doc_info["chunks"] = estimated_chunks
        else:
            # Add new file to metadata
            doc_id = str(doc_file.stem)
            size = doc_file.stat().st_size
            
            # Determine chunks count
            chunks = 0
            if doc_file.name in files_with_embeddings:
                # Estimate chunks based on file size
                chunks = max(1, size // 2000)  # Rough estimate
            
            # Add to metadata
            metadata["documents"][doc_file.name] = {
                "id": doc_id,
                "filename": doc_file.name,
                "chunks": chunks,
                "size": size,
                "date_added": str(datetime.now())
            }
    
    # Save metadata to file
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Update in-memory metadata for backward compatibility
    documents_metadata.clear()
    for filename, doc_info in metadata["documents"].items():
        doc_id = doc_info.get("id", str(Path(filename).stem))
        documents_metadata[doc_id] = {
            "filename": filename,
            "chunks": doc_info.get("chunks", 0),
            "size": doc_info.get("size", 0),
            "date_added": doc_info.get("date_added", str(datetime.now()))
        }
    
    # Return list of documents that need ingestion
    if auto_ingest:
        documents_to_ingest = []
        for doc_file in doc_files:
            if doc_file.name not in files_with_embeddings:
                documents_to_ingest.append(doc_file)
        return documents_to_ingest
    
    return []

# Add endpoint to check for documents that need ingestion
@app.get("/admin/check-documents")
async def check_documents_for_ingestion():
    """
    Check for documents in the documents directory that need to be ingested
    """
    documents_dir = Path("documents")
    if not documents_dir.exists():
        documents_dir.mkdir(exist_ok=True)
        return {"documents_to_ingest": []}
    
    # Get all document files
    doc_files = list(documents_dir.glob("*.pdf")) + \
               list(documents_dir.glob("*.txt")) + \
               list(documents_dir.glob("*.docx")) + \
               list(documents_dir.glob("*.md"))
    
    # Get document IDs that are already embedded
    embedded_doc_ids = set()
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        if any(c.name == "rag_chunks" for c in collections):
            # Get all points from the collection
            scroll_result = client.scroll(
                collection_name="rag_chunks",
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            # Extract unique document IDs from the payload
            for point in scroll_result[0]:
                if point.payload and "doc_id" in point.payload:
                    embedded_doc_ids.add(point.payload["doc_id"])
    except Exception as e:
        print(f"Error checking Qdrant: {e}")
        return {"error": str(e), "documents_to_ingest": []}
    
    # Find documents that need to be ingested
    documents_to_ingest = []
    for doc_file in doc_files:
        doc_id = f"doc_{doc_file.stem}_{hash(str(doc_file.absolute()))}"[:32]
        if doc_id not in embedded_doc_ids:
            documents_to_ingest.append({
                "filename": doc_file.name,
                "path": str(doc_file),
                "size": doc_file.stat().st_size,
                "doc_id": doc_id
            })
    
    return {"documents_to_ingest": documents_to_ingest}

@app.post("/admin/ingest-document")
async def ingest_document_by_path(document_data: dict):
    """
    Ingest a document that already exists in the documents directory
    """
    try:
        document_path = document_data.get("path")
        if not document_path:
            raise HTTPException(status_code=400, detail="Document path is required")
            
        doc_path = Path(document_path)
        if not doc_path.exists():
            raise HTTPException(status_code=404, detail=f"Document not found: {document_path}")
        
        # Process the document
        with open(doc_path, "rb") as f:
            file_content = f.read()
        
        # Create a temporary file with the same content
        with tempfile.NamedTemporaryFile(suffix=doc_path.suffix) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            
            # Ingest the document
            chunks, doc_id = await ingest_file(temp_file.name, doc_path.name)
            
            # Update metadata
            update_document_metadata(doc_id, doc_path.name, len(chunks), doc_path.stat().st_size)
            
            return {"status": "success", "doc_id": doc_id, "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a simple endpoint to ingest all documents in the documents folder
@app.post("/admin/ingest-all-documents")
async def ingest_all_documents():
    """Ingest all documents in the documents folder that aren't already embedded"""
    documents_to_ingest = await scan_documents_directory(auto_ingest=True)
    
    if not documents_to_ingest:
        return {"status": "success", "message": "No documents need ingestion"}
    
    ingested_count = 0
    errors = []
    
    for doc_file in documents_to_ingest:
        try:
            # Process the document
            with open(doc_file, "rb") as f:
                file_content = f.read()
            
            # Create a temporary file with the same content
            with tempfile.NamedTemporaryFile(suffix=doc_file.suffix) as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                
                # Ingest the document
                chunks, doc_id = await ingest_file(temp_file.name, doc_file.name)
                
                # Update metadata
                update_document_metadata(doc_id, doc_file.name, len(chunks), doc_file.stat().st_size)
                ingested_count += 1
        except Exception as e:
            errors.append({"filename": doc_file.name, "error": str(e)})
    
    return {
        "status": "success", 
        "ingested": ingested_count,
        "errors": errors
    }

# Define a background task to ingest documents
# Add a simple endpoint to reset and ingest all documents
@app.post("/admin/reset-and-ingest")
def reset_and_ingest():
    """Reset the database and ingest all documents"""
    try:
        # Get Qdrant client
        qdrant = get_qdrant_client()
        
        # Check if collection exists and delete it
        try:
            collections = qdrant.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if "rag_chunks" in collection_names:
                print("Deleting existing collection...")
                qdrant.delete_collection(collection_name="rag_chunks")
                # Add a small delay to ensure deletion is complete
                time.sleep(2)
        except Exception as e:
            print(f"Error checking/deleting collection: {e}")
        
        # Create the collection
        try:
            from qdrant_client.models import Distance, VectorParams
            embedder = get_embedder()
            vector_size = 384  # Size for all-MiniLM-L6-v2
            
            # Use create_collection instead of recreate_collection
            qdrant.create_collection(
                collection_name="rag_chunks",
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
        except Exception as e:
            print(f"Error creating collection: {e}")
            # If creation fails, try to continue with existing collection
        
        # Ingest all documents
        result = ingest_all_documents()
        return {
            "status": "success",
            "message": "Database reset and documents ingested",
            "details": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        }

# Smart document management on startup
@app.on_event("startup")
async def on_startup():
    print("\n===== STARTING DOCUMENT MANAGEMENT =====\n")
    
    try:
        # Get Qdrant client
        qdrant = get_qdrant_client()
        embedder = get_embedder()
        
        # Ensure the collection exists
        try:
            collections = qdrant.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if "rag_chunks" not in collection_names:
                # Create the collection if it doesn't exist
                from qdrant_client.models import Distance, VectorParams
                print("Creating new collection...")
                vector_size = 384  # Size for all-MiniLM-L6-v2
                qdrant.create_collection(
                    collection_name="rag_chunks",
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                print("Collection created successfully")
        except Exception as e:
            print(f"Error checking/creating collection: {e}")
            return
        
        # Get all documents in the documents directory
        documents_dir = Path("documents")
        if not documents_dir.exists():
            documents_dir.mkdir(exist_ok=True)
            print("Documents directory created")
        
        # Get all document files
        doc_files = list(documents_dir.glob("*.pdf")) + \
                   list(documents_dir.glob("*.txt")) + \
                   list(documents_dir.glob("*.docx")) + \
                   list(documents_dir.glob("*.md"))
        
        # Get all documents in the metadata
        metadata_file = Path("document_metadata.json")
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    metadata = {"documents": {}}
        else:
            metadata = {"documents": {}}
        
        # Get list of files in the metadata
        metadata_files = set(metadata.get("documents", {}).keys())
        
        # Get list of files in the documents directory
        doc_file_names = {doc_file.name for doc_file in doc_files}
        
        # Find files to add (in directory but not in metadata)
        files_to_add = doc_file_names - metadata_files
        
        # Find files to remove (in metadata but not in directory)
        files_to_remove = metadata_files - doc_file_names
        
        print(f"Found {len(doc_file_names)} documents in directory")
        print(f"Found {len(metadata_files)} documents in metadata")
        print(f"Files to add: {len(files_to_add)}")
        print(f"Files to remove: {len(files_to_remove)}")
        
        # Process files to remove
        if files_to_remove:
            print("\nRemoving embeddings for deleted files...")
            for filename in files_to_remove:
                try:
                    # Get document ID from metadata
                    doc_id = metadata["documents"][filename].get("id", filename)
                    
                    # Delete points with this document ID
                    filter_query = {
                        "must": [
                            {
                                "key": "filename",
                                "match": {"value": filename}
                            }
                        ]
                    }
                    
                    # Delete points matching the filter
                    qdrant.delete(
                        collection_name="rag_chunks",
                        points_selector=models.FilterSelector(filter=filter_query)
                    )
                    
                    # Remove from metadata
                    del metadata["documents"][filename]
                    print(f"Removed embeddings for {filename}")
                    
                    # Save updated metadata to file
                    with open(metadata_file, "w") as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Notify frontend about document removal
                    await manager.broadcast({
                        "type": "document_removed",
                        "filename": filename
                    })
                except Exception as e:
                    print(f"Error removing embeddings for {filename}: {e}")
        
        # Process files to add
        if files_to_add:
            print("\nAdding embeddings for new files...")
            for filename in files_to_add:
                try:
                    # Find the file in doc_files
                    doc_file = next(df for df in doc_files if df.name == filename)
                    print(f"Processing: {doc_file.name}")
                    
                    # Read the file content
                    with open(doc_file, "rb") as f:
                        file_content = f.read()
                    
                    # Create a temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=doc_file.suffix)
                    temp_file.write(file_content)
                    temp_file.close()
                    
                    try:
                        # Ingest the document
                        num_chunks = ingest_file(temp_file.name, qdrant, embedder)
                        
                        # Generate a document ID
                        doc_id = str(doc_file.stem)
                        
                        # Update metadata
                        update_document_metadata(doc_id, doc_file.name, num_chunks, doc_file.stat().st_size)
                        print(f"Successfully ingested {doc_file.name} with {num_chunks} chunks")
                        
                        # Notify frontend about new document
                        await manager.broadcast({
                            "type": "document_added",
                            "filename": doc_file.name,
                            "chunks": num_chunks,
                            "size": doc_file.stat().st_size
                        })
                    finally:
                        # Clean up the temporary file
                        os.unlink(temp_file.name)
                except Exception as e:
                    print(f"Error ingesting {filename}: {e}")
        
        # No need to save metadata here as it's already saved in update_document_metadata
        
        print(f"\n===== DOCUMENT MANAGEMENT COMPLETE =====\n")
            
    except Exception as e:
        print(f"Error during document management: {e}")

# This is important: DO NOT include the server starter code here
# The app instance should just be defined, not started
# Leave the server starting to main.py

# Chat history endpoints
@app.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(document_id: Optional[str] = None):
    """
    Get chat history, optionally filtered by document ID
    """
    # Load history from file to ensure we have the latest
    load_chat_history()
    
    # Filter by document if requested
    if document_id:
        filtered_history = [item for item in chat_history if item.get("document_id") == document_id]
    else:
        filtered_history = chat_history
    
    return ChatHistoryResponse(history=filtered_history)

@app.delete("/chat/history/{chat_id}")
async def delete_chat_item(chat_id: str):
    """
    Delete a specific chat history item
    """
    global chat_history
    
    # Load history from file to ensure we have the latest
    load_chat_history()
    
    # Find and remove the item
    original_length = len(chat_history)
    chat_history = [item for item in chat_history if item.get("id") != chat_id]
    
    if len(chat_history) < original_length:
        # Save the updated history
        save_chat_history()
        return {"success": True, "message": f"Chat item {chat_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Chat item {chat_id} not found")

@app.delete("/chat/history")
async def clear_chat_history(document_id: Optional[str] = None):
    """
    Clear all chat history or just for a specific document
    """
    global chat_history
    
    # Load history from file to ensure we have the latest
    load_chat_history()
    
    if document_id:
        # Only clear history for the specified document
        original_length = len(chat_history)
        chat_history = [item for item in chat_history if item.get("document_id") != document_id]
        
        if len(chat_history) < original_length:
            save_chat_history()
            return {"success": True, "message": f"Chat history for document {document_id} cleared"}
        else:
            raise HTTPException(status_code=404, detail=f"No chat history found for document {document_id}")
    else:
        # Clear all history
        chat_history = []
        save_chat_history()
        return {"success": True, "message": "All chat history cleared"}

# Load chat history on startup
@app.on_event("startup")
async def load_history_on_startup():
    load_chat_history()

# Export the app variable explicitly at the module level
__all__ = ["app"]