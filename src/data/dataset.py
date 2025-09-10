"""
Data loading and preprocessing utilities.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


class DataManager:
    """
    Data manager for handling datasets and data loaders.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data manager.
        
        Args:
            config: Data configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def get_transforms(self, is_training: bool = True) -> transforms.Compose:
        """
        Get data transforms for training or testing.
        
        Args:
            is_training: Whether to apply training transforms
            
        Returns:
            Composed transforms
        """
        if is_training:
            transform_list = [
                transforms.ToTensor(),
            ]
            
            if self.config.get("normalize", True):
                transform_list.append(
                    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
                )
            
            if self.config.get("augment", False):
                transform_list.insert(-1, transforms.RandomRotation(10))
                transform_list.insert(-1, transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
                
        else:
            transform_list = [transforms.ToTensor()]
            
            if self.config.get("normalize", True):
                transform_list.append(
                    transforms.Normalize((0.1307,), (0.3081,))
                )
        
        return transforms.Compose(transform_list)
    
    def load_mnist(
        self,
        data_dir: Path,
        batch_size: int = 64,
        validation_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load MNIST dataset with train/validation/test splits.
        
        Args:
            data_dir: Directory to store the dataset
            batch_size: Batch size for data loaders
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=self.config.get("download", True),
            transform=self.get_transforms(is_training=True)
        )
        
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=self.config.get("download", True),
            transform=self.get_transforms(is_training=False)
        )
        
        # Split training data into train and validation
        train_size = int((1 - validation_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.get("random_seed", 42))
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.logger.info(f"Loaded MNIST dataset:")
        self.logger.info(f"  Training samples: {len(train_dataset)}")
        self.logger.info(f"  Validation samples: {len(val_dataset)}")
        self.logger.info(f"  Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def load_custom_dataset(
        self,
        data_path: Path,
        batch_size: int = 64,
        **kwargs
    ) -> DataLoader:
        """
        Load a custom dataset from a directory.
        
        Args:
            data_path: Path to the dataset directory
            batch_size: Batch size for the data loader
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            DataLoader for the custom dataset
        """
        # This is a placeholder for custom dataset loading
        # You can implement specific logic based on your dataset format
        self.logger.info(f"Loading custom dataset from {data_path}")
        
        # Example implementation for image folder structure
        dataset = datasets.ImageFolder(
            root=data_path,
            transform=self.get_transforms(is_training=True)
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            **kwargs
        )
