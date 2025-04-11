import os
import argparse
import logging
import sys
from pathlib import Path

from ingestion import ingest_folder
from model import get_llm, QUANTIZED_MODEL_PATH, MODEL_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

def setup_environment():
    """Set up the environment and directories."""
    # Ensure directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Log paths for debugging
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Model directory: {MODEL_DIR}")
    logger.info(f"Documents directory: {DOCUMENTS_DIR}")
    
    # Make sure the project root is in the Python path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    # Set up Qdrant URL
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

def check_model():
    """Check if the pre-quantized model exists."""
    try:
        logger.info(f"Looking for model at: {QUANTIZED_MODEL_PATH}")
        
        if not QUANTIZED_MODEL_PATH.exists():
            logger.error(f"Pre-quantized model not found at {QUANTIZED_MODEL_PATH}")
            logger.error("Please download the model from Hugging Face and place it in the models directory")
            return False
        
        # Try to load the model to verify it works
        llm = get_llm()
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def start_api_server(host, port, reload_option):
    """
    Start the FastAPI server using uvicorn.
    This function avoids circular imports by using string-based import in uvicorn.
    """
    import uvicorn
    
    logger.info(f"Starting API server at {host}:{port}")
    logger.info(f"API documentation will be available at http://{host}:{port}/docs")
    
    # Run the uvicorn server with the api module's app instance
    uvicorn.run(
        "api:app",  # Use string-based import
        host=host,
        port=port,
        reload=reload_option,
        log_level="info"
    )

def main():
    """Main entry point for the RAG application."""
    parser = argparse.ArgumentParser(description="RAG Application with Llama 3.1")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("folder", help="Folder containing documents to ingest")
    
    # API server command
    api_parser = subparsers.add_parser("serve", help="Start the API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Set up environment (including Python path)
    setup_environment()
    
    if args.command == "ingest":
        folder_path = Path(args.folder).absolute()
        if not folder_path.exists() or not folder_path.is_dir():
            logger.error(f"Error: {args.folder} is not a valid directory")
            return
        
        logger.info(f"Ingesting documents from {folder_path}")
        ingest_folder(folder_path)
    
    elif args.command == "serve":
        # Check if the model is available before starting the server
        model_ready = check_model()
        if not model_ready:
            logger.warning("Model not ready. API may have limited functionality.")
        
        # Start the API server
        start_api_server(args.host, args.port, args.reload)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()