# Neural Network Project

A comprehensive neural network project built with PyTorch and managed with UV for fast Python package management.

## Features

- **Modular Architecture**: Clean separation of models, training, data, and configuration
- **Multiple Model Types**: Support for both MLP and CNN architectures
- **Fast Package Management**: Uses UV for lightning-fast dependency management
- **Comprehensive Logging**: Detailed training logs and metrics
- **Easy Configuration**: Centralized configuration management
- **Reproducible Results**: Fixed random seeds and proper data splitting

## Project Structure

```
neural-network-project/
├── src/
│   ├── models/           # Neural network model definitions
│   │   ├── __init__.py
│   │   └── neural_network.py
│   ├── training/         # Training utilities and trainer class
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── data/            # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── config/          # Configuration settings
│   │   ├── __init__.py
│   │   └── settings.py
│   └── utils/           # Utility functions
│       └── __init__.py
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks for experimentation
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── assets/              # Data, models, and logs
│   ├── data/           # Dataset storage
│   ├── models/         # Saved model checkpoints
│   ├── logs/           # Training logs
│   └── checkpoints/    # Model checkpoints
├── main.py             # Main training script
├── pyproject.toml      # Project configuration and dependencies
└── README.md           # This file
```

## Installation

### Prerequisites

- Python 3.8+
- UV package manager

### Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd neural-network-project
   ```

2. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

## Usage

### Training a Model

#### Train an MLP (Multi-Layer Perceptron):
```bash
python main.py --model mlp --epochs 10 --batch-size 64 --learning-rate 0.001 --save-model
```

#### Train a CNN (Convolutional Neural Network):
```bash
python main.py --model cnn --epochs 15 --batch-size 32 --learning-rate 0.0005 --save-model
```

### Command Line Arguments

- `--model`: Model type to train (`mlp` or `cnn`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 64)
- `--learning-rate`: Learning rate for optimizer (default: 0.001)
- `--save-model`: Save the trained model and training history

### Example Training Output

```
2024-01-15 10:30:00 - __main__ - INFO - Starting neural network training
2024-01-15 10:30:00 - __main__ - INFO - Arguments: Namespace(model='mlp', epochs=10, batch_size=64, learning_rate=0.001, save_model=True)
2024-01-15 10:30:01 - src.data.dataset - INFO - Loaded MNIST dataset:
2024-01-15 10:30:01 - src.data.dataset - INFO -   Training samples: 48000
2024-01-15 10:30:01 - src.data.dataset - INFO -   Validation samples: 12000
2024-01-15 10:30:01 - src.data.dataset - INFO -   Test samples: 10000
2024-01-15 10:30:01 - __main__ - INFO - Model info: {'input_size': 784, 'hidden_sizes': [128, 64], 'output_size': 10, 'dropout_rate': 0.2, 'total_parameters': 109386, 'trainable_parameters': 109386}
2024-01-15 10:30:01 - src.training.trainer - INFO - Starting training for 10 epochs
2024-01-15 10:30:01 - src.training.trainer - INFO - Device: cuda
2024-01-15 10:30:01 - src.training.trainer - INFO - Model parameters: 109386
...
```

## Configuration

The project uses centralized configuration in `src/config/settings.py`. You can modify:

- **Model Configuration**: Architecture parameters, learning rate, batch size
- **Training Configuration**: Early stopping, validation split, random seed
- **Data Configuration**: Dataset settings, normalization, augmentation
- **Logging Configuration**: Log levels and file paths

## Development

### Adding New Models

1. Create a new model class in `src/models/`
2. Inherit from `torch.nn.Module`
3. Implement `forward()` method
4. Add model initialization in `main.py`

### Adding New Datasets

1. Extend the `DataManager` class in `src/data/dataset.py`
2. Implement dataset-specific loading logic
3. Update configuration in `src/config/settings.py`

### Running Tests

```bash
uv run pytest tests/
```

### Code Formatting

```bash
uv run black src/ tests/
uv run isort src/ tests/
```

## Dependencies

The project uses the following key dependencies:

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **Scikit-learn**: Machine learning utilities
- **Jupyter**: Interactive development
- **TensorBoard**: Training visualization
- **Weights & Biases**: Experiment tracking

## Performance

- **UV Package Manager**: 10-100x faster than pip for dependency resolution
- **Efficient Data Loading**: Multi-process data loading with pinned memory
- **GPU Support**: Automatic CUDA detection and usage
- **Memory Optimization**: Gradient checkpointing and efficient batching

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with PyTorch and UV
- Inspired by modern ML engineering practices
- Designed for educational and research purposes
