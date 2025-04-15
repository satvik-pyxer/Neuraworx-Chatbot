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
                raw_text = result["choices"][0]["text"]
            else:  # ctransformers interface
                raw_text = self.model(
                    formatted_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            
            # Clean up the response by removing special tokens and formatting artifacts
            cleaned_text = self._clean_model_output(raw_text)
            return cleaned_text
            
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
        
    def _clean_model_output(self, text: str) -> str:
        """Clean up the model's output by removing special tokens and formatting artifacts."""
        # Remove any trailing special tokens
        if "<|im_end|>" in text:
            text = text.split("<|im_end|>")[0]
            
        # Remove any repeated assistant tokens that might appear in the output
        text = text.replace("<|im_start|>assistant", "")
        
        # Remove any other special tokens that might be in the output
        text = text.replace("<|im_start|>", "")
        text = text.replace("<|im_end|>", "")
        
        # Clean up any repeated newlines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
            
        # Trim whitespace
        text = text.strip()
        
        return text

# Singleton instance for the LLM
_llm_instance = None
_current_model_path = None

def get_llm() -> LLMWrapper:
    """Get or initialize the LLM instance."""
    global _llm_instance, _current_model_path
    
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
                model_path = str(gguf_files[0])
                _llm_instance = LLMWrapper(model_path)
                _current_model_path = model_path
                return _llm_instance
            else:
                raise FileNotFoundError(
                    f"No GGUF model found at {QUANTIZED_MODEL_PATH}. " 
                    "Please download the pre-quantized model from Hugging Face and place it in the models directory."
                )
        
        # Load the model
        model_path = str(QUANTIZED_MODEL_PATH)
        _llm_instance = LLMWrapper(model_path)
        _current_model_path = model_path
    
    return _llm_instance

def reload_llm(model_filename: str) -> None:
    """Reload the LLM with a different model."""
    global _llm_instance, _current_model_path
    
    model_path = str(MODEL_DIR / model_filename)
    
    # If the model is already loaded, do nothing
    if _current_model_path == model_path:
        logger.info(f"Model {model_filename} is already loaded")
        return
    
    logger.info(f"Reloading LLM with model: {model_filename}")
    
    # Delete the old instance to free up memory
    if _llm_instance is not None:
        del _llm_instance
    
    # Create a new instance with the new model
    _llm_instance = LLMWrapper(model_path)
    _current_model_path = model_path
    logger.info(f"Model reloaded successfully: {model_filename}")


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