"""
Training script for Human Activity Recognition model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np

from model import create_model
from data_loader import create_data_loaders
from utils import MetricsCalculator, save_checkpoint, print_metrics_report


class Trainer:
    """Trainer class for model training"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, dataset_info,
                 device, checkpoint_dir='checkpoints', lr=0.001, weight_decay=1e-5):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            dataset_info: Dataset information dictionary
            device: torch device
            checkpoint_dir: Directory to save checkpoints
            lr: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dataset_info = dataset_info
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3,
            min_lr=1e-6
        )
        
        # Best validation loss for model selection
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        
        # Metrics
        self.class_names = dataset_info['class_names']
        self.num_classes = dataset_info['num_classes']
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        metrics = MetricsCalculator(self.num_classes, self.class_names)
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for frames, labels in progress_bar:
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            metrics.update(preds, labels_np)
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_metrics = metrics.get_metrics()
        
        return epoch_loss, epoch_metrics
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        metrics = MetricsCalculator(self.num_classes, self.class_names)
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')
            for frames, labels in progress_bar:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                metrics.update(preds, labels_np)
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_metrics = metrics.get_metrics()
        
        return epoch_loss, epoch_metrics
    
    def train(self, num_epochs=50):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs
        """
        print("="*60)
        print("Starting Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.class_names}")
        print(f"Training samples: {self.dataset_info['train_size']}")
        print(f"Validation samples: {self.dataset_info['val_size']}")
        print("="*60)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_metrics['accuracy'])
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                checkpoint_path = self.checkpoint_dir / 'best_model.pth'
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, checkpoint_path
                )
                print(f"Saved best model with val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time/60:.2f} minutes")
        
        # Test on test set
        print("\n" + "="*60)
        print("Evaluating on Test Set")
        print("="*60)
        self.test()
        
        return history
    
    def test(self):
        """Test the model"""
        self.model.eval()
        metrics = MetricsCalculator(self.num_classes, self.class_names)
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc='Testing')
            for frames, labels in progress_bar:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                metrics.update(preds, labels_np)
        
        test_metrics = metrics.get_metrics()
        cm = metrics.get_confusion_matrix()
        
        print_metrics_report(test_metrics, cm, self.class_names)
        
        return test_metrics, cm


def main():
    parser = argparse.ArgumentParser(description='Train Human Activity Recognition Model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--model-type', type=str, default='cnnlstm',
                       choices=['cnnlstm', 'cnn3d', 'simplelstm'],
                       help='Type of model to use')
    parser.add_argument('--num-frames', type=int, default=16,
                       help='Number of frames to extract from video')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    print(f"\nLoading data from {args.data_dir}...")
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        num_classes=dataset_info['num_classes']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        dataset_info=dataset_info,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Train
    history = trainer.train(num_epochs=args.num_epochs)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
