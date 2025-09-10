"""
Configuration settings for the neural network project.
"""
import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "assets" / "data"
MODELS_DIR = PROJECT_ROOT / "assets" / "models"
LOGS_DIR = PROJECT_ROOT / "assets" / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "assets" / "checkpoints"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG: Dict[str, Any] = {
    "input_size": 784,  # 28x28 for MNIST
    "hidden_sizes": [128, 64],
    "output_size": 10,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
}

# Training configuration
TRAINING_CONFIG: Dict[str, Any] = {
    "early_stopping_patience": 5,
    "save_best_model": True,
    "log_interval": 100,
    "validation_split": 0.2,
    "random_seed": 42
}

# Data configuration
DATA_CONFIG: Dict[str, Any] = {
    "dataset_name": "MNIST",
    "download": True,
    "transform": True,
    "normalize": True
}

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "training.log"
}
