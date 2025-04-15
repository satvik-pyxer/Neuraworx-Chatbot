"""Model configuration and management for Neuraworx-Chatbot."""
from dataclasses import dataclass
from typing import List, Optional
import json
from pathlib import Path

@dataclass
class ModelInfo:
    name: str
    file_name: str
    description: str
    size_mb: int
    quantization: str
    context_window: int
    download_url: str

# List of supported models
AVAILABLE_MODELS = [
    ModelInfo(
        name="Llama 3.1 8B Instruct",
        file_name="llama-3.1-8b-instruct-q4_k_m.gguf",
        description="Default model. Balanced performance and quality for general Q&A tasks.",
        size_mb=4096,
        quantization="4-bit",
        context_window=8192,
        download_url="https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF/resolve/main/llama-3.1-8b-instruct-q4_k_m.gguf"
    ),
    ModelInfo(
        name="Mistral 7B Instruct",
        file_name="mistral-7b-instruct-v0.1.Q4_0.gguf",
        description="Alternative model with good performance for instruction following.",
        size_mb=4096,
        quantization="4-bit",
        context_window=8192,
        download_url="https://gpt4all.io/models/gguf/mistral-7b-instruct-v0.1.Q4_0.gguf"
    ),
    ModelInfo(
        name="Wizard Vicuna 7B",
        file_name="wizard-vicuna-7b.Q4_0.gguf",
        description="Optimized for instruction following and detailed responses.",
        size_mb=4096,
        quantization="4-bit",
        context_window=2048,
        download_url="https://gpt4all.io/models/gguf/wizard-vicuna-7b.Q4_0.gguf"
    )
]

class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.config_file = self.models_dir / "config.json"
        self.load_config()

    def load_config(self) -> None:
        """Load model configuration from disk."""
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                "active_model": "llama-3.1-8b-instruct-q4_k_m.gguf",
                "downloaded_models": []
            }
            self.save_config()

    def save_config(self) -> None:
        """Save model configuration to disk."""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)

    def get_active_model(self) -> str:
        """Get the currently active model filename."""
        return self.config["active_model"]

    def set_active_model(self, model_filename: str) -> None:
        """Set the active model by filename."""
        if not any(m.file_name == model_filename for m in AVAILABLE_MODELS):
            raise ValueError(f"Invalid model filename: {model_filename}")
        self.config["active_model"] = model_filename
        self.save_config()

    def get_model_info(self, model_filename: str) -> Optional[ModelInfo]:
        """Get model info by filename."""
        return next((m for m in AVAILABLE_MODELS if m.file_name == model_filename), None)

    def is_model_downloaded(self, model_filename: str) -> bool:
        """Check if a model file exists in the models directory."""
        return (self.models_dir / model_filename).exists()

    def get_downloaded_models(self) -> List[str]:
        """Get list of downloaded model filenames."""
        return [f.name for f in self.models_dir.glob("*.gguf")]

    def get_model_status(self) -> dict:
        """Get current status of all models."""
        active_model = self.get_active_model()
        downloaded_models = self.get_downloaded_models()
        
        return {
            "active_model": active_model,
            "models": [
                {
                    **vars(model),
                    "is_active": model.file_name == active_model,
                    "is_downloaded": model.file_name in downloaded_models
                }
                for model in AVAILABLE_MODELS
            ]
        }
