import os
import logging
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants with absolute paths ---
# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
MODEL_DIR = PROJECT_ROOT / "models"  # Using absolute path instead of environment variable
QUANTIZED_MODEL_PATH = MODEL_DIR / "llama-3.1-8b-instruct-q4_k_m.gguf"  # Pre-quantized model path
DEFAULT_MAX_TOKENS = 2048

class LLMWrapper:
    """Simple wrapper for LLM inference using llama.cpp-python or ctransformers."""
    
    def __init__(self, model_path: str):
        # Convert to Path and then back to string to normalize the path
        self.model_path = str(Path(model_path))
        self.model = None
        logger.info(f"Initializing LLM with model path: {self.model_path}")
        self._load_model()
    
    def _load_model(self):
        """Load the LLM model."""
        try:
            # Try to load with llama-cpp-python
            from llama_cpp import Llama
            
            logger.info(f"Loading model from {self.model_path}")
            # Make sure the file exists and is readable
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,  # Context window size
                n_gpu_layers=-1,  # Auto-detect GPU layers
                verbose=False
            )
            logger.info("Model loaded successfully with llama-cpp-python")
        except ImportError:
            logger.error("llama-cpp-python not installed. Trying ctransformers...")
            try:
                # Try to load with ctransformers as a fallback
                from ctransformers import AutoModelForCausalLM
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    model_type="llama",
                    max_new_tokens=DEFAULT_MAX_TOKENS
                )
                logger.info("Model loaded successfully with ctransformers")
            except ImportError:
                logger.error("Neither llama-cpp-python nor ctransformers is installed.")
                raise ImportError("Please install llama-cpp-python: pip install llama-cpp-python")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.7, **kwargs) -> str:
        """Generate text from the model."""
        try:
            # Format prompt for Llama 3.1
            formatted_prompt = self._format_llama3_prompt(prompt)
            
            # Handle different interfaces depending on the loaded module
            if hasattr(self.model, "__call__"):  # llama-cpp-python interface
                result = self.model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return result["choices"][0]["text"]
            else:  # ctransformers interface
                return self.model(
                    formatted_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error generating response: {str(e)}"
    
    def _format_llama3_prompt(self, prompt: str) -> str:
        """Format prompt for Llama 3.1 models according to their expected format."""
        # Check if the prompt is already formatted
        if "<|im_start|>" in prompt:
            return prompt
            
        # Format according to Llama 3.1 chat format
        formatted_prompt = "<|im_start|>system\nYou are a helpful AI assistant.\n<|im_end|>\n"
        formatted_prompt += f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

# Singleton instance for the LLM
_llm_instance = None

def get_llm() -> LLMWrapper:
    """Get or initialize the LLM instance."""
    global _llm_instance
    
    if _llm_instance is None:
        # Ensure model directory exists
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Print the full path for debugging
        logger.info(f"Looking for model at: {QUANTIZED_MODEL_PATH}")
        
        # Check if quantized model exists
        if not QUANTIZED_MODEL_PATH.exists():
            logger.warning(f"Pre-quantized model not found at {QUANTIZED_MODEL_PATH}")
            
            # Check for any existing GGUF model as fallback
            gguf_files = list(MODEL_DIR.glob("*.gguf"))
            if gguf_files:
                logger.info(f"Found existing GGUF model: {gguf_files[0]}")
                _llm_instance = LLMWrapper(str(gguf_files[0]))
                return _llm_instance
            else:
                raise FileNotFoundError(
                    f"No GGUF model found at {QUANTIZED_MODEL_PATH}. " 
                    "Please download the pre-quantized model from Hugging Face and place it in the models directory."
                )
        
        # Load the model
        _llm_instance = LLMWrapper(str(QUANTIZED_MODEL_PATH))
    
    return _llm_instance

def check_dependencies():
    """Check if necessary dependencies are installed."""
    try:
        # Check for llama-cpp-python
        import llama_cpp
        logger.info("llama-cpp-python is installed.")
        return True
    except ImportError:
        logger.warning("llama-cpp-python is not installed. Checking for alternatives...")
        
        # Check for ctransformers as fallback
        try:
            import ctransformers
            logger.info("ctransformers is installed as a fallback option.")
            return True
        except ImportError:
            logger.error("Neither llama-cpp-python nor ctransformers is installed.")
            logger.error("Please install one of them: pip install llama-cpp-python")
            return False