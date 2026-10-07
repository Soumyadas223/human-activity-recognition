"""
Utility functions for Human Activity Recognition System
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch


class MetricsCalculator:
    """Calculate and store evaluation metrics"""
    
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.all_preds = []
        self.all_targets = []
    
    def update(self, predictions, targets):
        """Update predictions and targets"""
        self.all_preds.extend(predictions)
        self.all_targets.extend(targets)
    
    def get_metrics(self):
        """Calculate all metrics"""
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, average='weighted', zero_division=0)
        recall = recall_score(targets, preds, average='weighted', zero_division=0)
        f1 = f1_score(targets, preds, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def get_confusion_matrix(self):
        """Get confusion matrix"""
        return confusion_matrix(self.all_targets, self.all_preds)
    
    def reset(self):
        """Reset metrics"""
        self.all_preds = []
        self.all_targets = []


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def plot_metrics(metrics_dict, save_path=None):
    """
    Plot evaluation metrics
    
    Args:
        metrics_dict: Dictionary with metrics
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]
        value = metrics_dict[metric]
        
        bars = ax.bar(['Score'], [value], color=color, alpha=0.7)
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        
        # Add value text on bar
        ax.text(0, value + 0.02, f'{value:.4f}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    else:
        plt.show()


def draw_activity_label(frame, activity, confidence, position=(30, 50)):
    """
    Draw activity label and confidence on frame
    
    Args:
        frame: Input frame (BGR)
        activity: Activity label string
        confidence: Confidence score (0-1)
        position: (x, y) position for text
    
    Returns:
        Frame with drawn text
    """
    frame = frame.copy()
    color = (0, 255, 0)  # Green for good confidence
    
    if confidence < 0.6:
        color = (0, 165, 255)  # Orange for medium confidence
    if confidence < 0.4:
        color = (0, 0, 255)  # Red for low confidence
    
    text = f"{activity}: {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    
    # Add background rectangle
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(frame, (x - 10, y - 30), (x + text_size[0] + 10, y + 10), 
                  (0, 0, 0), -1)
    
    # Add text
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    
    return frame


def draw_bounding_box(frame, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box on frame
    
    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        color: RGB color
        thickness: Line thickness
    
    Returns:
        Frame with bounding box
    """
    frame = frame.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def save_video(frames, output_path, fps=30, frame_size=None):
    """
    Save list of frames as video
    
    Args:
        frames: List of frames
        output_path: Output video path
        fps: Frames per second
        frame_size: (width, height) for output
    """
    if len(frames) == 0:
        print("No frames to save")
        return
    
    if frame_size is None:
        h, w = frames[0].shape[:2]
        frame_size = (w, h)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    for frame in frames:
        if frame.shape[:2] != (frame_size[1], frame_size[0]):
            frame = cv2.resize(frame, frame_size)
        writer.write(frame)
    
    writer.release()
    print(f"Video saved to {output_path}")

def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path):
    """Save model checkpoint."""

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }

    torch.save(checkpoint, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}") 
def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    val_loss = checkpoint.get("val_loss", 0.0)

    return model, checkpoint
def print_metrics_report(metrics, class_cm=None, class_names=None):
    """Print formatted metrics report."""

    print("\n" + "=" * 50)
    print("EVALUATION METRICS REPORT")
    print("=" * 50)

    for metric, value in metrics.items():
        print(f"{metric.upper():<20}: {value:.4f}")

    if class_cm is not None and class_names is not None:
        print("\n" + "=" * 50)
        print("PER-CLASS ANALYSIS")
        print("=" * 50)

        for i, class_name in enumerate(class_names):

            if i >= class_cm.shape[0] or i >= class_cm.shape[1]:
                print(f"\n{class_name}: No validation samples")
                print("Precision: 0.0000")
                print("Recall: 0.0000")
                print("F1-Score: 0.0000")
                continue

            tp = class_cm[i, i]
            fp = class_cm[:, i].sum() - tp
            fn = class_cm[i, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0 else 0
            )

            print(f"\n{class_name}:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")