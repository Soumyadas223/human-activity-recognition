"""
Data loading and preprocessing for Human Activity Recognition
"""

import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image


class VideoActivityDataset(Dataset):
    """
    Dataset class for loading video frames and labels
    
    Expected directory structure:
    dataset_root/
    ├── activity1/
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   └── ...
    ├── activity2/
    │   ├── video1.mp4
    │   └── ...
    └── ...
    """
    
    def __init__(self, root_dir, num_frames=16, frame_size=(224, 224), 
                 transform=None, sampling_strategy='uniform'):
        """
        Initialize dataset
        
        Args:
            root_dir: Root directory containing activity folders
            num_frames: Number of frames to extract from each video
            frame_size: (height, width) for resizing frames
            transform: PyTorch transforms to apply
            sampling_strategy: 'uniform' or 'random' frame sampling
        """
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        self.sampling_strategy = sampling_strategy
        
        # Get activity classes and create label mapping
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all video paths
        self.video_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.root_dir / class_name
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            
            for ext in video_extensions:
                video_files = list(class_dir.glob(f'*{ext}'))
                for video_file in video_files:
                    self.video_paths.append(str(video_file))
                    self.labels.append(class_idx)
        
        if len(self.video_paths) == 0:
            raise ValueError(f"No video files found in {root_dir}")
        
        print(f"Dataset initialized with {len(self.video_paths)} videos")
        print(f"Classes: {self.classes}")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _get_class_distribution(self):
        """Get class distribution"""
        distribution = {}
        for label in self.labels:
            class_name = self.classes[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def _extract_frames(self, video_path):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
        
        Returns:
            frames: List of extracted frames (num_frames, height, width, 3)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Could not read frames from {video_path}")
        
        # Determine frame indices to extract
        if self.sampling_strategy == 'uniform':
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:  # random
            indices = sorted(random.sample(range(total_frames), 
                                          min(self.num_frames, total_frames)))
            # Pad with repeated frames if not enough frames
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
        
        # Extract frames
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame in indices:
                # Resize frame
                frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            current_frame += 1
        
        cap.release()
        
        # Ensure we have exactly num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        frames = frames[:self.num_frames]
        
        return frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Get item from dataset
        
        Returns:
            frames: Tensor of shape (num_frames, 3, height, width)
            label: Class label
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames
        try:
            frames = self._extract_frames(video_path)
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return black frames as fallback
            frames = [np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8) 
                     for _ in range(self.num_frames)]
        
        # Convert to numpy array
        frames = np.array(frames, dtype=np.float32)  # (num_frames, height, width, 3)
        
        # Normalize to [0, 1]
        frames = frames / 255.0
        
        # Apply transforms if provided
        if self.transform:
            frames_list = [Image.fromarray((f * 255).astype(np.uint8)) for f in frames]
            frames_list = [np.array(self.transform(f)) for f in frames_list]
            frames = np.array(frames_list)
        
        # Convert to tensor and permute to (num_frames, 3, height, width)
        frames = torch.from_numpy(frames)
        frames = frames.permute(0, 3, 1, 2)  # (num_frames, 3, height, width)
        
        return frames, label


class FrameFeatureDataset(Dataset):
    """
    Dataset for pre-extracted frame features
    Useful when you have pre-computed CNN features
    """
    
    def __init__(self, features_dir, num_frames=16, transform=None):
        """
        Initialize feature dataset
        
        Args:
            features_dir: Directory containing feature files
            num_frames: Number of frames per sample
            transform: Optional transforms
        """
        self.features_dir = Path(features_dir)
        self.num_frames = num_frames
        self.transform = transform
        
        # Get all feature files
        self.feature_files = sorted(self.features_dir.glob('*.npy'))
        self.labels = []
        
        if len(self.feature_files) == 0:
            raise ValueError(f"No feature files found in {features_dir}")
        
        print(f"Loaded {len(self.feature_files)} feature files")
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        feature_file = self.feature_files[idx]
        
        # Load features (assuming format: features.npy)
        features = np.load(feature_file)  # (num_frames, feature_dim)
        
        # Convert to tensor
        features = torch.from_numpy(features).float()
        
        # Extract label from filename (assuming format: label_videoid.npy)
        parts = feature_file.stem.split('_')
        label = int(parts[0])
        
        return features, label


def create_data_loaders(root_dir, train_split=0.7, val_split=0.2, test_split=0.1,
                       batch_size=32, num_workers=4, num_frames=16,
                       frame_size=(224, 224), transform=None, seed=42):
    """
    Create train, validation, and test data loaders
    
    Args:
        root_dir: Root directory with video data
        train_split: Training split ratio
        val_split: Validation split ratio
        test_split: Test split ratio
        batch_size: Batch size
        num_workers: Number of workers for data loading
        num_frames: Number of frames to extract
        frame_size: Frame size (height, width)
        transform: PyTorch transforms
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create dataset
    dataset = VideoActivityDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        frame_size=frame_size,
        transform=transform
    )
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Dataset info
    dataset_info = {
        'num_classes': len(dataset.classes),
        'class_names': dataset.classes,
        'class_to_idx': dataset.class_to_idx,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'class_distribution': dataset._get_class_distribution()
    }
    
    return train_loader, val_loader, test_loader, dataset_info


def create_single_video_loader(video_path, num_frames=16, frame_size=(224, 224), 
                              batch_size=1, transform=None):
    """
    Create data loader for a single video
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        frame_size: Frame size (height, width)
        batch_size: Batch size
        transform: PyTorch transforms
    
    Returns:
        loader: DataLoader for video
    """
    
    class SingleVideoDataset(Dataset):
        def __init__(self, video_path, num_frames, frame_size, transform):
            self.video_path = video_path
            self.num_frames = num_frames
            self.frame_size = frame_size
            self.transform = transform
        
        def __len__(self):
            return 1
        
        def __getitem__(self, idx):
            cap = cv2.VideoCapture(self.video_path)
            frames = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            current_frame = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if current_frame in indices:
                    frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                
                current_frame += 1
            
            cap.release()
            
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
            frames = frames[:self.num_frames]
            
            frames = np.array(frames, dtype=np.float32) / 255.0
            
            if self.transform:
                frames_list = [Image.fromarray((f * 255).astype(np.uint8)) for f in frames]
                frames_list = [np.array(self.transform(f)) for f in frames_list]
                frames = np.array(frames_list)
            
            frames = torch.from_numpy(frames)
            frames = frames.permute(0, 3, 1, 2)
            
            return frames, 0
    
    dataset = SingleVideoDataset(video_path, num_frames, frame_size, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader
