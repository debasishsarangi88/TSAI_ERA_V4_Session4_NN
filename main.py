"""
Main script for training and evaluating neural networks.
"""
import logging
import torch
from pathlib import Path
import argparse

from src.config.settings import (
    PROJECT_ROOT, MODELS_DIR, LOGS_DIR, 
    MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG
)
from src.models.neural_network import NeuralNetwork, CNN
from src.training.trainer import Trainer
from src.data.dataset import DataManager


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / "training.log"),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Neural Network Training")
    parser.add_argument("--model", type=str, default="mlp", 
                       choices=["mlp", "cnn"], help="Model type to train")
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, 
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                       help="Learning rate for optimizer")
    parser.add_argument("--save-model", action="store_true", 
                       help="Save the trained model")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting neural network training")
    logger.info(f"Arguments: {args}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(TRAINING_CONFIG["random_seed"])
    
    # Initialize data manager
    data_manager = DataManager(DATA_CONFIG)
    
    # Load data
    train_loader, val_loader, test_loader = data_manager.load_mnist(
        data_dir=PROJECT_ROOT / "assets" / "data",
        batch_size=args.batch_size,
        validation_split=TRAINING_CONFIG["validation_split"]
    )
    
    # Initialize model
    if args.model == "mlp":
        model = NeuralNetwork(
            input_size=MODEL_CONFIG["input_size"],
            hidden_sizes=MODEL_CONFIG["hidden_sizes"],
            output_size=MODEL_CONFIG["output_size"],
            dropout_rate=MODEL_CONFIG["dropout_rate"]
        )
    elif args.model == "cnn":
        model = CNN(
            input_channels=1,
            num_classes=MODEL_CONFIG["output_size"],
            dropout_rate=MODEL_CONFIG["dropout_rate"]
        )
    
    logger.info(f"Model info: {model.get_model_info() if hasattr(model, 'get_model_info') else 'CNN model'}")
    
    # Initialize trainer
    training_config = TRAINING_CONFIG.copy()
    training_config.update({
        "learning_rate": args.learning_rate,
        "log_interval": 100
    })
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )
    
    # Train model
    save_path = MODELS_DIR / f"{args.model}_best.pth" if args.save_model else None
    history = trainer.train(epochs=args.epochs, save_path=save_path)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_acc = trainer.validate()
    trainer.train_loader = test_loader  # Temporarily replace for test evaluation
    test_loss, test_acc = trainer.validate()
    
    logger.info(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
    # Save training history
    if args.save_model:
        torch.save(history, MODELS_DIR / f"{args.model}_history.pth")
        logger.info(f"Saved training history to {MODELS_DIR / f'{args.model}_history.pth'}")


if __name__ == "__main__":
    main()