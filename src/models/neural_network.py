"""
Neural Network model implementation using PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network with configurable architecture.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Build the network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions on input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "dropout_rate": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }


class CNN(nn.Module):
    """
    Convolutional Neural Network for image classification.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the CNN.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the size after convolutions and pooling
        # For 28x28 input: 28 -> 14 -> 7 -> 3 (after 3 conv+pool layers)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
