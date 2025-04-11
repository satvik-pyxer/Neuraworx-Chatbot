import os
import uuid
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import fitz  # PyMuPDF
from docx import Document

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# --- Configs ---
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_DIR = PROJECT_ROOT / "models/embeddings"

SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf", ".docx"]
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# --- Loaders ---
def load_txt(path):
    return Path(path).read_text(encoding="utf-8")

def load_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def load_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_file(path):
    ext = path.suffix.lower()
    if ext == ".txt" or ext == ".md":
        return load_txt(path)
    elif ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def extract_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract rich metadata from the file."""
    stats = file_path.stat()
    
    metadata = {
        "filename": file_path.name,
        "file_path": str(file_path.absolute()),
        "file_type": file_path.suffix.lower(),
        "file_size_bytes": stats.st_size,
        "created_at": datetime.datetime.fromtimestamp(stats.st_ctime).isoformat(),
        "modified_at": datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
        "ingestion_timestamp": datetime.datetime.now().isoformat(),
    }
    
    # Additional metadata for PDF files
    if file_path.suffix.lower() == ".pdf":
        try:
            doc = fitz.open(file_path)
            if doc.metadata:
                pdf_meta = doc.metadata
                for key, value in pdf_meta.items():
                    if value:  # Only add non-empty values
                        metadata[f"pdf_{key.lower()}"] = value
        except Exception as e:
            print(f"Failed to extract PDF metadata: {e}")
    
    return metadata

# --- Chunking ---
def chunk_document(text, metadata=None):
    """Split text into chunks and add positional metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    
    result = []
    for i, chunk in enumerate(chunks):
        chunk_metadata = metadata.copy() if metadata else {}
        chunk_metadata.update({
            "chunk_index": i,
            "chunk_count": len(chunks),
            "chunk_size": len(chunk),
            "relative_position": float(i) / max(1, len(chunks) - 1),  # 0.0 to 1.0
            "is_first_chunk": i == 0,
            "is_last_chunk": i == len(chunks) - 1,
        })
        result.append({"text": chunk, "metadata": chunk_metadata})
    
    return result

# --- Embedding ---
def get_embedder():
    """Get the embedding model."""
    # Ensure embeddings directory exists
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(EMBEDDINGS_DIR))

def embed_chunks(chunks, model):
    """Embed the text of each chunk."""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts).tolist()
    return embeddings

# --- Qdrant Storage ---
def get_qdrant_client():
    """Get the Qdrant client."""
    return QdrantClient(
        url=QDRANT_URL,
        api_key=os.getenv("QDRANT_API_KEY"),
    )

def store_in_qdrant(chunks_with_metadata, embeddings, qdrant: QdrantClient):
    """Store chunks with metadata and embeddings in Qdrant."""
    if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
        )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk["text"],
                **chunk["metadata"]  # Include all metadata
            }
        )
        for chunk, embedding in zip(chunks_with_metadata, embeddings)
    ]

    qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)
    return len(points)

# --- Ingest File ---
def ingest_file(file_path: Path, qdrant=None, embedder=None) -> int:
    """Process a single file and store it in Qdrant. Returns the number of chunks processed."""
    file_path = Path(file_path)  # Ensure it's a Path object
    
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Skipping unsupported file: {file_path.name}")
        return 0
    
    try:
        print(f"Processing: {file_path.name}")
        
        # Initialize clients if not provided
        qdrant = qdrant or get_qdrant_client()
        embedder = embedder or get_embedder()
        
        # Extract file metadata
        metadata = extract_metadata(file_path)
        
        # Load and chunk the document
        text = load_file(file_path)
        chunks_with_metadata = chunk_document(text, metadata)
        
        # Generate embeddings
        embeddings = embed_chunks(chunks_with_metadata, embedder)
        
        # Store in Qdrant
        num_chunks = store_in_qdrant(chunks_with_metadata, embeddings, qdrant)
        
        print(f"Stored {num_chunks} chunks from {file_path.name}")
        return num_chunks
    
    except Exception as e:
        print(f"Failed to process {file_path.name}: {e}")
        return 0

# --- Ingest Folder ---
def ingest_folder(folder_path, qdrant_url=None, qdrant_api_key=None):
    """Ingest all supported files in a folder. Returns a summary of the ingestion."""
    
    folder_path = Path(folder_path).absolute()  # Ensure it's an absolute Path
    
    # Log the folder path for debugging
    print(f"Ingesting documents from folder: {folder_path}")
    
    qdrant_url = qdrant_url or QDRANT_URL
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    
    embedder = get_embedder()
    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    total_files = 0
    total_chunks = 0
    processed_files = []
    
    for file in folder_path.glob("*"):
        if file.suffix.lower() in SUPPORTED_EXTENSIONS:
            total_files += 1
            chunks = ingest_file(file, qdrant, embedder)
            total_chunks += chunks
            if chunks > 0:
                processed_files.append({
                    "filename": file.name,
                    "chunks": chunks
                })

    summary = {
        "total_files": total_files,
        "processed_files": len(processed_files),
        "total_chunks": total_chunks,
        "file_details": processed_files
    }
    
    print(f"🚀 Ingestion complete! Processed {len(processed_files)}/{total_files} files, created {total_chunks} chunks.")
    return summary

# Example usage
if __name__ == "__main__":
    documents_dir = PROJECT_ROOT / "documents"
    documents_dir.mkdir(exist_ok=True)
    ingest_folder(documents_dir)