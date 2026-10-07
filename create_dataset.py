"""
Dataset creation script for Human Activity Recognition
Creates synthetic data or organizes real videos
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import random


class SyntheticDatasetGenerator:
    """Generate synthetic video data for activities"""
    
    def __init__(self, output_dir, num_videos_per_class=10, video_length=60, fps=30, frame_size=(224, 224)):
        """
        Initialize synthetic dataset generator
        
        Args:
            output_dir: Output directory
            num_videos_per_class: Number of videos per activity class
            video_length: Video length in frames
            fps: Frames per second
            frame_size: (height, width)
        """
        self.output_dir = Path(output_dir)
        self.num_videos_per_class = num_videos_per_class
        self.video_length = video_length
        self.fps = fps
        self.frame_size = frame_size
        
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_walking(self, video_path):
        """Generate walking activity video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, 
                                (self.frame_size[1], self.frame_size[0]))
        
        for frame_idx in range(self.video_length):
            frame = np.ones((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8) * 255
            
            # Draw a person (stick figure) walking
            h, w = frame.shape[:2]
            progress = (frame_idx / self.video_length) * 2 * np.pi
            
            # Head position
            head_x = int(w / 2 + 50 * np.sin(progress))
            head_y = int(h / 2 - 80)
            
            # Body parts
            cv2.circle(frame, (head_x, head_y), 20, (0, 0, 0), 2)
            cv2.line(frame, (head_x, head_y + 20), (head_x, head_y + 60), (0, 0, 0), 2)
            
            # Legs with movement
            leg_offset = 30 * np.sin(progress)
            cv2.line(frame, (head_x, head_y + 60), 
                    (int(head_x - leg_offset), int(head_y + 110)), (0, 0, 0), 2)
            cv2.line(frame, (head_x, head_y + 60), 
                    (int(head_x + leg_offset), int(head_y + 110)), (0, 0, 0), 2)
            
            # Add activity label
            cv2.putText(frame, 'Walking', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 0), 2)
            
            writer.write(frame)
        
        writer.release()
    
    def generate_running(self, video_path):
        """Generate running activity video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps,
                                (self.frame_size[1], self.frame_size[0]))
        
        for frame_idx in range(self.video_length):
            frame = np.ones((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8) * 200
            
            h, w = frame.shape[:2]
            progress = (frame_idx / self.video_length) * 4 * np.pi  # Faster than walking
            
            # Head position
            head_x = int(w / 2 + 80 * np.sin(progress))
            head_y = int(h / 2 - 80)
            
            cv2.circle(frame, (head_x, head_y), 20, (0, 0, 0), 2)
            cv2.line(frame, (head_x, head_y + 20), (head_x, head_y + 50), (0, 0, 0), 2)
            
            # Legs with faster movement
            leg_offset = 40 * np.sin(progress)
            cv2.line(frame, (head_x, head_y + 50),
                    (int(head_x - leg_offset), int(head_y + 100)), (0, 0, 0), 2)
            cv2.line(frame, (head_x, head_y + 50),
                    (int(head_x + leg_offset), int(head_y + 100)), (0, 0, 0), 2)
            
            cv2.putText(frame, 'Running', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 0), 2)
            
            writer.write(frame)
        
        writer.release()
    
    def generate_sitting(self, video_path):
        """Generate sitting activity video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps,
                                (self.frame_size[1], self.frame_size[0]))
        
        for frame_idx in range(self.video_length):
            frame = np.ones((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8) * 240
            
            h, w = frame.shape[:2]
            
            # Head position (sitting)
            head_x = int(w / 2)
            head_y = int(h / 2 - 40)
            
            cv2.circle(frame, (head_x, head_y), 20, (0, 0, 0), 2)
            cv2.line(frame, (head_x, head_y + 20), (head_x, head_y + 50), (0, 0, 0), 2)
            
            # Bent legs
            cv2.line(frame, (head_x, head_y + 50), (head_x - 30, head_y + 80), (0, 0, 0), 2)
            cv2.line(frame, (head_x, head_y + 50), (head_x + 30, head_y + 80), (0, 0, 0), 2)
            
            # Add chair
            cv2.rectangle(frame, (head_x - 50, head_y + 80), (head_x + 50, head_y + 120),
                         (100, 100, 100), 2)
            
            cv2.putText(frame, 'Sitting', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 0), 2)
            
            writer.write(frame)
        
        writer.release()
    
    def generate_standing(self, video_path):
        """Generate standing activity video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps,
                                (self.frame_size[1], self.frame_size[0]))
        
        for frame_idx in range(self.video_length):
            frame = np.ones((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8) * 230
            
            h, w = frame.shape[:2]
            
            # Head position (standing)
            head_x = int(w / 2)
            head_y = int(h / 2 - 100)
            
            cv2.circle(frame, (head_x, head_y), 20, (0, 0, 0), 2)
            cv2.line(frame, (head_x, head_y + 20), (head_x, head_y + 70), (0, 0, 0), 2)
            
            # Straight legs
            cv2.line(frame, (head_x, head_y + 70), (head_x - 20, head_y + 130), (0, 0, 0), 2)
            cv2.line(frame, (head_x, head_y + 70), (head_x + 20, head_y + 130), (0, 0, 0), 2)
            
            cv2.putText(frame, 'Standing', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 0), 2)
            
            writer.write(frame)
        
        writer.release()
    
    def generate_jumping(self, video_path):
        """Generate jumping activity video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps,
                                (self.frame_size[1], self.frame_size[0]))
        
        for frame_idx in range(self.video_length):
            frame = np.ones((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8) * 220
            
            h, w = frame.shape[:2]
            
            # Head position (jumping)
            progress = (frame_idx % 20) / 20.0  # Periodic jump
            vertical_offset = int(60 * np.sin(progress * np.pi))
            
            head_x = int(w / 2)
            head_y = int(h / 2 - 100 - vertical_offset)
            
            cv2.circle(frame, (head_x, head_y), 20, (0, 0, 0), 2)
            cv2.line(frame, (head_x, head_y + 20), (head_x, head_y + 60), (0, 0, 0), 2)
            
            # Legs bent during jump
            cv2.line(frame, (head_x, head_y + 60), (head_x - 20, head_y + 100), (0, 0, 0), 2)
            cv2.line(frame, (head_x, head_y + 60), (head_x + 20, head_y + 100), (0, 0, 0), 2)
            
            cv2.putText(frame, 'Jumping', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 0), 2)
            
            writer.write(frame)
        
        writer.release()
    
    def generate_falling(self, video_path):
        """Generate falling activity video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, self.fps,
                                (self.frame_size[1], self.frame_size[0]))
        
        for frame_idx in range(self.video_length):
            frame = np.ones((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8) * 210
            
            h, w = frame.shape[:2]
            
            # Gradually falling down
            fall_progress = min(1.0, frame_idx / 30)
            head_x = int(w / 2)
            head_y = int(h / 2 - 100 + 100 * fall_progress)
            rotation = int(90 * fall_progress)
            
            cv2.circle(frame, (head_x, head_y), 20, (0, 0, 0), 2)
            
            if fall_progress < 1.0:
                cv2.line(frame, (head_x, head_y + 20), (head_x, head_y + 70), (0, 0, 0), 2)
                cv2.line(frame, (head_x, head_y + 70), (head_x - 30, head_y + 120), (0, 0, 0), 2)
                cv2.line(frame, (head_x, head_y + 70), (head_x + 30, head_y + 120), (0, 0, 0), 2)
            else:
                # Lying down
                cv2.line(frame, (head_x, head_y), (head_x + 50, head_y), (0, 0, 0), 2)
                cv2.line(frame, (head_x + 50, head_y), (head_x + 50, head_y - 30), (0, 0, 0), 2)
            
            cv2.putText(frame, 'Falling', (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 0, 0), 2)
            
            writer.write(frame)
        
        writer.release()
    
    def generate_dataset(self):
        """Generate synthetic dataset"""
        activities = {
            'walking': self.generate_walking,
            'running': self.generate_running,
            'sitting': self.generate_sitting,
            'standing': self.generate_standing,
            'jumping': self.generate_jumping,
            'falling': self.generate_falling
        }
        
        print("Generating synthetic dataset...")
        print(f"Output directory: {self.output_dir}")
        print(f"Videos per class: {self.num_videos_per_class}")
        
        for activity_name, generate_func in activities.items():
            activity_dir = self.output_dir / activity_name
            activity_dir.mkdir(exist_ok=True)
            
            print(f"\nGenerating {activity_name}...")
            for video_idx in tqdm(range(self.num_videos_per_class)):
                video_path = activity_dir / f'{activity_name}_{video_idx:03d}.mp4'
                generate_func(str(video_path))
        
        print(f"\nDataset generation completed!")
        print(f"Total videos: {len(activities) * self.num_videos_per_class}")


class DatasetOrganizer:
    """Organize real video files into class directories"""
    
    def __init__(self, output_dir):
        """
        Initialize organizer
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_class_structure(self, classes):
        """
        Create directory structure for classes
        
        Args:
            classes: List of class names
        """
        print(f"Creating directory structure in {self.output_dir}")
        for class_name in classes:
            class_dir = self.output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            print(f"  Created: {class_dir}")
    
    def print_instructions(self):
        """Print instructions for organizing dataset"""
        print("\n" + "="*60)
        print("Dataset Organization Instructions")
        print("="*60)
        print("\nDirectory structure expected:")
        print("dataset_root/")
        print("├── walking/")
        print("│   ├── video1.mp4")
        print("│   ├── video2.mp4")
        print("│   └── ...")
        print("├── running/")
        print("├── sitting/")
        print("├── standing/")
        print("├── jumping/")
        print("└── falling/")
        print("\nPlace your video files in the corresponding class directories")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Create dataset for Human Activity Recognition')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for dataset')
    parser.add_argument('--mode', type=str, default='synthetic',
                       choices=['synthetic', 'organize'],
                       help='Mode: synthetic (generate synthetic data) or organize (setup directories)')
    parser.add_argument('--num-videos', type=int, default=10,
                       help='Number of videos per class (for synthetic mode)')
    parser.add_argument('--video-length', type=int, default=60,
                       help='Length of each video in frames')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second')
    
    args = parser.parse_args()
    
    if args.mode == 'synthetic':
        generator = SyntheticDatasetGenerator(
            output_dir=args.output_dir,
            num_videos_per_class=args.num_videos,
            video_length=args.video_length,
            fps=args.fps
        )
        generator.generate_dataset()
    
    else:  # organize
        organizer = DatasetOrganizer(args.output_dir)
        classes = ['walking', 'running', 'sitting', 'standing', 'jumping', 'falling']
        organizer.create_class_structure(classes)
        organizer.print_instructions()


if __name__ == '__main__':
    main()
