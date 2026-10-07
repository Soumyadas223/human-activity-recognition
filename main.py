"""
Main script for Human Activity Recognition System
Complete end-to-end pipeline
"""

import argparse
import subprocess
import sys
from pathlib import Path


class PipelineManager:
    """Manage the complete HAR pipeline"""
    
    def __init__(self):
        self.steps = {
            '1': ('dataset', 'Create or organize dataset'),
            '2': ('train', 'Train the model'),
            '3': ('evaluate', 'Evaluate the model'),
            '4': ('inference', 'Run inference on video')
        }
    
    def print_menu(self):
        """Print main menu"""
        print("\n" + "="*60)
        print("Human Activity Recognition System - Main Menu")
        print("="*60)
        for key, (_, description) in self.steps.items():
            print(f"{key}. {description}")
        print("5. Run complete pipeline (1 -> 2 -> 3 -> 4)")
        print("0. Exit")
        print("="*60)
    
    def create_dataset(self):
        """Create or organize dataset"""
        print("\n" + "="*60)
        print("Dataset Creation")
        print("="*60)
        
        mode = input("Select mode (1: synthetic, 2: organize existing): ").strip()
        output_dir = input("Enter output directory path: ").strip()
        
        if mode == '1':
            num_videos = input("Number of videos per class (default: 10): ").strip() or '10'
            cmd = [
                'python', 'create_dataset.py',
                '--output-dir', output_dir,
                '--mode', 'synthetic',
                '--num-videos', num_videos
            ]
        else:
            cmd = [
                'python', 'create_dataset.py',
                '--output-dir', output_dir,
                '--mode', 'organize'
            ]
        
        subprocess.run(cmd, check=True)
    
    def train_model(self):
        """Train the model"""
        print("\n" + "="*60)
        print("Model Training")
        print("="*60)
        
        data_dir = input("Enter dataset directory path: ").strip()
        model_type = input("Model type (cnnlstm/cnn3d/simplelstm, default: cnnlstm): ").strip() or 'cnnlstm'
        batch_size = input("Batch size (default: 32): ").strip() or '32'
        num_epochs = input("Number of epochs (default: 50): ").strip() or '50'
        
        cmd = [
            'python', 'train.py',
            '--data-dir', data_dir,
            '--model-type', model_type,
            '--batch-size', batch_size,
            '--num-epochs', num_epochs
        ]
        
        subprocess.run(cmd, check=True)
    
    def evaluate_model(self):
        """Evaluate the model"""
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        data_dir = input("Enter dataset directory path: ").strip()
        checkpoint = input("Enter checkpoint path: ").strip()
        model_type = input("Model type (cnnlstm/cnn3d/simplelstm, default: cnnlstm): ").strip() or 'cnnlstm'
        output_dir = input("Output directory for results (default: evaluation_results): ").strip() or 'evaluation_results'
        
        cmd = [
            'python', 'evaluate.py',
            '--data-dir', data_dir,
            '--checkpoint', checkpoint,
            '--model-type', model_type,
            '--output-dir', output_dir
        ]
        
        subprocess.run(cmd, check=True)
    
    def run_inference(self):
        """Run inference on video"""
        print("\n" + "="*60)
        print("Video Inference")
        print("="*60)
        
        video_path = input("Enter video path: ").strip()
        checkpoint = input("Enter checkpoint path: ").strip()
        classes_str = input("Enter class names (comma-separated, e.g., walking,running,sitting,standing): ").strip()
        model_type = input("Model type (cnnlstm/cnn3d/simplelstm, default: cnnlstm): ").strip() or 'cnnlstm'
        output_video = input("Output video path (optional, press Enter to skip): ").strip() or None
        
        cmd = [
            'python', 'inference.py',
            '--video', video_path,
            '--checkpoint', checkpoint,
            '--class-names', classes_str,
            '--model-type', model_type
        ]
        
        if output_video:
            cmd.extend(['--output-video', output_video])
        
        subprocess.run(cmd, check=True)
    
    def run_full_pipeline(self):
        """Run complete pipeline"""
        print("\n" + "="*60)
        print("Running Complete Pipeline")
        print("="*60)
        
        # Step 1: Dataset
        print("\nStep 1: Dataset Creation")
        data_dir = input("Enter dataset directory path: ").strip()
        
        if not Path(data_dir).exists():
            print("Dataset directory doesn't exist. Creating...")
            num_videos = input("Number of videos per class (default: 5): ").strip() or '5'
            cmd = [
                'python', 'create_dataset.py',
                '--output-dir', data_dir,
                '--mode', 'synthetic',
                '--num-videos', num_videos
            ]
            subprocess.run(cmd, check=True)
        
        # Step 2: Training
        print("\nStep 2: Model Training")
        model_type = input("Model type (cnnlstm/cnn3d/simplelstm, default: cnnlstm): ").strip() or 'cnnlstm'
        batch_size = input("Batch size (default: 16): ").strip() or '16'
        num_epochs = input("Number of epochs (default: 30): ").strip() or '30'
        
        cmd = [
            'python', 'train.py',
            '--data-dir', data_dir,
            '--model-type', model_type,
            '--batch-size', batch_size,
            '--num-epochs', num_epochs
        ]
        subprocess.run(cmd, check=True)
        
        # Step 3: Evaluation
        print("\nStep 3: Model Evaluation")
        checkpoint = Path('checkpoints') / 'best_model.pth'
        cmd = [
            'python', 'evaluate.py',
            '--data-dir', data_dir,
            '--checkpoint', str(checkpoint),
            '--model-type', model_type
        ]
        subprocess.run(cmd, check=True)
        
        # Step 4: Inference
        print("\nStep 4: Video Inference")
        test_video = input("Enter test video path (or press Enter to skip): ").strip()
        
        if test_video:
            output_video = input("Output video path (press Enter to skip): ").strip() or None
            cmd = [
                'python', 'inference.py',
                '--video', test_video,
                '--checkpoint', str(checkpoint),
                '--class-names', 'walking,running,sitting,standing,jumping,falling',
                '--model-type', model_type
            ]
            if output_video:
                cmd.extend(['--output-video', output_video])
            subprocess.run(cmd, check=True)
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
    
    def run(self):
        """Run the menu"""
        while True:
            self.print_menu()
            choice = input("Select option: ").strip()
            
            try:
                if choice == '0':
                    print("Exiting...")
                    break
                elif choice == '1':
                    self.create_dataset()
                elif choice == '2':
                    self.train_model()
                elif choice == '3':
                    self.evaluate_model()
                elif choice == '4':
                    self.run_inference()
                elif choice == '5':
                    self.run_full_pipeline()
                else:
                    print("Invalid option. Please try again.")
            
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Human Activity Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py
  
  # Create synthetic dataset
  python main.py --create-dataset --output-dir ./data --num-videos 10
  
  # Train model
  python main.py --train --data-dir ./data --model cnnlstm --epochs 50
  
  # Evaluate model
  python main.py --evaluate --data-dir ./data --checkpoint checkpoints/best_model.pth
  
  # Run inference
  python main.py --infer --video test_video.mp4 --checkpoint checkpoints/best_model.pth
        """
    )
    
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'dataset', 'train', 'evaluate', 'infer'],
                       help='Mode to run')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory')
    parser.add_argument('--data-dir', type=str,
                       help='Dataset directory')
    parser.add_argument('--checkpoint', type=str,
                       help='Model checkpoint path')
    parser.add_argument('--video', type=str,
                       help='Video file path')
    parser.add_argument('--model', type=str, default='cnnlstm',
                       help='Model type')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-videos', type=int, default=10,
                       help='Number of videos per class')
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        manager = PipelineManager()
        manager.run()
    
    elif args.mode == 'dataset':
        if not args.output_dir:
            print("Error: --output-dir required for dataset mode")
            sys.exit(1)
        
        cmd = [
            'python', 'create_dataset.py',
            '--output-dir', args.output_dir,
            '--mode', 'synthetic',
            '--num-videos', str(args.num_videos)
        ]
        subprocess.run(cmd, check=True)
    
    elif args.mode == 'train':
        if not args.data_dir:
            print("Error: --data-dir required for train mode")
            sys.exit(1)
        
        cmd = [
            'python', 'train.py',
            '--data-dir', args.data_dir,
            '--model-type', args.model,
            '--batch-size', str(args.batch_size),
            '--num-epochs', str(args.epochs)
        ]
        subprocess.run(cmd, check=True)
    
    elif args.mode == 'evaluate':
        if not args.data_dir or not args.checkpoint:
            print("Error: --data-dir and --checkpoint required for evaluate mode")
            sys.exit(1)
        
        cmd = [
            'python', 'evaluate.py',
            '--data-dir', args.data_dir,
            '--checkpoint', args.checkpoint,
            '--model-type', args.model
        ]
        subprocess.run(cmd, check=True)
    
    elif args.mode == 'infer':
        if not args.video or not args.checkpoint:
            print("Error: --video and --checkpoint required for infer mode")
            sys.exit(1)
        
        cmd = [
            'python', 'inference.py',
            '--video', args.video,
            '--checkpoint', args.checkpoint,
            '--class-names', 'walking,running,sitting,standing,jumping,falling',
            '--model-type', args.model
        ]
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
