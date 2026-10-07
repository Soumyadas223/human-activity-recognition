"""
Inference script for Human Activity Recognition
"""

import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from model import create_model
from data_loader import create_single_video_loader
from utils import load_checkpoint, draw_activity_label, save_video


class ActivityRecognizer:
    """Activity Recognition Inference Engine"""
    
    def __init__(self, model, class_names, device, num_frames=16, frame_size=(224, 224)):
        """
        Initialize recognizer
        
        Args:
            model: PyTorch model
            class_names: List of class names
            device: torch device
            num_frames: Number of frames
            frame_size: Frame size (height, width)
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.model.eval()
    
    def predict_video(self, video_path):
        """
        Predict activity for entire video
        
        Args:
            video_path: Path to video file
        
        Returns:
            predictions: Dictionary with results
        """
        # Load video data
        loader = create_single_video_loader(
            video_path,
            num_frames=self.num_frames,
            frame_size=self.frame_size
        )
        
        with torch.no_grad():
            for frames, _ in loader:
                frames = frames.to(self.device)
                outputs = self.model(frames)
                probs = torch.softmax(outputs, dim=1)
                
                pred_class = torch.argmax(outputs, dim=1).item()
                confidence = probs[0, pred_class].item()
                all_probs = probs[0].cpu().numpy()
        
        predictions = {
            'class': self.class_names[pred_class],
            'class_idx': pred_class,
            'confidence': confidence,
            'all_probabilities': {
                self.class_names[i]: float(all_probs[i])
                for i in range(len(self.class_names))
            }
        }
        
        return predictions
    
    def predict_video_with_frames(self, video_path):
        """
        Predict activity and process video frames
        
        Args:
            video_path: Path to video file
        
        Returns:
            processed_frames: List of processed frames
            predictions: Prediction results
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Read all frames
        frames = []
        all_frames_original = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Reading video: {total_frames} frames at {fps:.2f} fps")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames_original.append(frame)
        
        cap.release()
        
        # Sample frames uniformly
        if len(all_frames_original) > self.num_frames:
            indices = np.linspace(0, len(all_frames_original) - 1, 
                                 self.num_frames, dtype=int)
        else:
            indices = np.arange(len(all_frames_original))
        
        # Process sampled frames
        for idx in indices:
            frame = all_frames_original[int(idx)]
            frame_resized = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb / 255.0)
        
        # Pad if necessary
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        frames = frames[:self.num_frames]
        
        # Convert to tensor
        frames_tensor = np.array(frames, dtype=np.float32)
        frames_tensor = torch.from_numpy(frames_tensor)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(frames_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            pred_class = torch.argmax(outputs, dim=1).item()
            confidence = probs[0, pred_class].item()
            all_probs = probs[0].cpu().numpy()
        
        predictions = {
            'class': self.class_names[pred_class],
            'class_idx': pred_class,
            'confidence': confidence,
            'all_probabilities': {
                self.class_names[i]: float(all_probs[i])
                for i in range(len(self.class_names))
            }
        }
        
        # Process all frames for output video
        processed_frames = []
        for frame in all_frames_original:
            frame_with_label = draw_activity_label(
                frame,
                predictions['class'],
                predictions['confidence']
            )
            processed_frames.append(frame_with_label)
        
        return processed_frames, predictions
    
    def predict_single_frame(self, frame):
        """
        Predict activity for a single frame
        
        Args:
            frame: Single frame (height, width, 3) in BGR
        
        Returns:
            prediction: Dictionary with results
        """
        # Preprocess frame
        frame_resized = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb / 255.0
        
        # Create sequence of same frame (for temporal models)
        frames_sequence = np.repeat(
            frame_normalized[np.newaxis, :, :, :],
            self.num_frames, axis=0
        )
        
        # Convert to tensor
        frames_tensor = torch.from_numpy(frames_sequence).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(frames_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            pred_class = torch.argmax(outputs, dim=1).item()
            confidence = probs[0, pred_class].item()
            all_probs = probs[0].cpu().numpy()
        
        prediction = {
            'class': self.class_names[pred_class],
            'class_idx': pred_class,
            'confidence': confidence,
            'all_probabilities': {
                self.class_names[i]: float(all_probs[i])
                for i in range(len(self.class_names))
            }
        }
        
        return prediction


def main():
    parser = argparse.ArgumentParser(description='Run inference on video')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--class-names', type=str, required=True,
                       help='Comma-separated class names (e.g., walking,running,sitting,standing)')
    parser.add_argument('--model-type', type=str, default='cnnlstm',
                       choices=['cnnlstm', 'cnn3d', 'simplelstm'],
                       help='Type of model')
    parser.add_argument('--num-frames', type=int, default=16,
                       help='Number of frames')
    parser.add_argument('--frame-size', type=int, nargs=2, default=[224, 224],
                       help='Frame size (height width)')
    parser.add_argument('--output-video', type=str, default=None,
                       help='Path to save output video with predictions')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Parse class names
    class_names = args.class_names.split(',')
    num_classes = len(class_names)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    print(f"Creating {args.model_type} model with {num_classes} classes...")
    model = create_model(
        model_type=args.model_type,
        num_classes=num_classes
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, _ = load_checkpoint(model, args.checkpoint, device)
    
    # Create recognizer
    recognizer = ActivityRecognizer(
        model=model,
        class_names=class_names,
        device=device,
        num_frames=args.num_frames,
        frame_size=tuple(args.frame_size)
    )
    
    # Run inference
    print(f"\nRunning inference on {args.video}...")
    
    if args.output_video:
        processed_frames, predictions = recognizer.predict_video_with_frames(args.video)
        
        print(f"\nPrediction: {predictions['class']}")
        print(f"Confidence: {predictions['confidence']:.4f}")
        print("\nAll probabilities:")
        for cls, prob in predictions['all_probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
        
        # Save output video
        print(f"\nSaving output video to {args.output_video}...")
        save_video(processed_frames, args.output_video, fps=30)
    else:
        predictions = recognizer.predict_video(args.video)
        
        print(f"\nPrediction: {predictions['class']}")
        print(f"Confidence: {predictions['confidence']:.4f}")
        print("\nAll probabilities:")
        for cls, prob in predictions['all_probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
    
    print("\nInference completed!")


if __name__ == '__main__':
    main()
