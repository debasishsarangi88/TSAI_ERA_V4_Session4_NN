"""
Efficient MNIST Training Script
Target: <25k parameters, >95% accuracy in 1 epoch

This script demonstrates how to achieve high accuracy on MNIST with minimal parameters
in just one epoch through careful architecture design and training optimization.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time


class EfficientMNIST(nn.Module):
    """
    Highly efficient CNN for MNIST classification.
    Achieves >95% accuracy in 1 epoch with <25k parameters.
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.1):
        super(EfficientMNIST, self).__init__()
        
        # Efficient CNN architecture optimized for MNIST
        # Total parameters: 24,048 (within 25k limit)
        
        # First conv block: 1->16 channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second conv block: 16->32 channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Third conv block: 32->64 channels, 3x3 kernel
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Global average pooling instead of fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final classification layer (no bias to save parameters)
        self.classifier = nn.Linear(64, num_classes, bias=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # First conv block: 28x28 -> 14x14
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        
        # Second conv block: 14x14 -> 7x7
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        
        # Third conv block: 7x7 -> 3x3 (after pooling)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        
        # Global average pooling: 3x3 -> 1x1
        x = self.global_avg_pool(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    
    def get_parameter_count(self):
        """Get parameter breakdown."""
        total_params = sum(p.numel() for p in self.parameters())
        conv_params = sum(p.numel() for name, p in self.named_parameters() if 'conv' in name)
        bn_params = sum(p.numel() for name, p in self.named_parameters() if 'bn' in name)
        classifier_params = sum(p.numel() for name, p in self.named_parameters() if 'classifier' in name)
        
        return {
            "total": total_params,
            "conv": conv_params,
            "batch_norm": bn_params,
            "classifier": classifier_params
        }


def load_mnist_data(batch_size=64, validation_split=0.1):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, test_loader, epochs=1, lr=0.015):
    """Train the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.001)
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                current_acc = 100.0 * correct / total
                print(f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        scheduler.step()
        
        train_acc = 100.0 * correct / total
        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += target.size(0)
    
    test_acc = 100.0 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    if test_acc >= 95.0:
        print("🎉 TARGET ACHIEVED: >95% accuracy in 1 epoch!")
    else:
        print(f"Target not met. Current accuracy: {test_acc:.2f}%")
    
    return test_acc


def main():
    """Main training function."""
    print("=" * 60)
    print("EFFICIENT MNIST TRAINING")
    print("Target: <25k parameters, >95% accuracy in 1 epoch")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model
    model = EfficientMNIST()
    
    # Display model information
    params = model.get_parameter_count()
    print(f"Model Architecture:")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Conv layers: {params['conv']:,}")
    print(f"  Batch norm: {params['batch_norm']:,}")
    print(f"  Classifier: {params['classifier']:,}")
    print()
    
    if params['total'] > 25000:
        print("⚠️  Model exceeds 25k parameter limit!")
        return
    else:
        print("✅ Model within 25k parameter limit")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, val_loader, test_loader = load_mnist_data(batch_size=64)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()
    
    # Train model
    start_time = time.time()
    test_accuracy = train_model(model, train_loader, val_loader, test_loader)
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
