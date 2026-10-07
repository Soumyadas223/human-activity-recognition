"""
Evaluation script for Human Activity Recognition model
"""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import create_model
from data_loader import create_data_loaders, create_single_video_loader
from utils import (MetricsCalculator, load_checkpoint, plot_confusion_matrix, 
                   plot_metrics, print_metrics_report, draw_activity_label, save_video)


class Evaluator:
    """Evaluator class for model evaluation"""
    
    def __init__(self, model, test_loader, dataset_info, device):
        """
        Initialize evaluator
        
        Args:
            model: PyTorch model
            test_loader: Test data loader
            dataset_info: Dataset information
            device: torch device
        """
        self.model = model
        self.test_loader = test_loader
        self.dataset_info = dataset_info
        self.device = device
        
        self.class_names = dataset_info['class_names']
        self.num_classes = dataset_info['num_classes']
        self.class_to_idx = dataset_info['class_to_idx']
    
    def evaluate(self):
        """Evaluate model on test set"""
        self.model.eval()
        metrics = MetricsCalculator(self.num_classes, self.class_names)
        
        all_probs = []
        
        print("Evaluating model on test set...")
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc='Evaluating')
            for frames, labels in progress_bar:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                probs = torch.softmax(outputs, dim=1)
                
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                probs_np = probs.cpu().numpy()
                
                metrics.update(preds, labels_np)
                all_probs.extend(probs_np)
        
        test_metrics = metrics.get_metrics()
        cm = metrics.get_confusion_matrix()
        
        print_metrics_report(test_metrics, cm, self.class_names)
        
        return test_metrics, cm, np.array(all_probs)
    
    def analyze_per_class(self):
        """Analyze performance per class"""
        self.model.eval()
        class_metrics = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in self.class_names}
        
        with torch.no_grad():
            for frames, labels in tqdm(self.test_loader, desc='Analyzing per-class'):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                preds = torch.argmax(outputs, dim=1)
                
                for pred, label in zip(preds, labels):
                    pred = pred.item()
                    label = label.item()
                    
                    if pred == label:
                        class_metrics[self.class_names[label]]['tp'] += 1
                    else:
                        class_metrics[self.class_names[pred]]['fp'] += 1
                        class_metrics[self.class_names[label]]['fn'] += 1
        
        # Calculate metrics
        print("\nPer-Class Performance:")
        print("-" * 70)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        
        for cls in self.class_names:
            tp = class_metrics[cls]['tp']
            fp = class_metrics[cls]['fp']
            fn = class_metrics[cls]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{cls:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Human Activity Recognition Model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='cnnlstm',
                       choices=['cnnlstm', 'cnn3d', 'simplelstm'],
                       help='Type of model')
    parser.add_argument('--num-frames', type=int, default=16,
                       help='Number of frames')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    train_loader, val_loader, test_loader, dataset_info = create_data_loaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        num_classes=dataset_info['num_classes']
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    model, checkpoint_info = load_checkpoint(model, args.checkpoint, device)
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        dataset_info=dataset_info,
        device=device
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    test_metrics, cm, probs = evaluator.evaluate()
    
    # Per-class analysis
    print("\n" + "="*60)
    print("Per-Class Analysis")
    print("="*60)
    evaluator.analyze_per_class()
    
    # Plot results
    print(f"\nSaving plots to {output_dir}...")
    plot_confusion_matrix(cm, dataset_info['class_names'], 
                         save_path=output_dir / 'confusion_matrix.png')
    plot_metrics(test_metrics, save_path=output_dir / 'metrics.png')
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
