# Human Activity Recognition System (HARS)

A complete end-to-end PyTorch implementation for human activity recognition from video using CNN-LSTM and 3D CNN architectures.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The Human Activity Recognition System is designed to automatically identify human activities from video input. It processes video sequences, extracts spatial and temporal features, and classifies activities into predefined categories.

**Activities Recognized:**
- Walking
- Running
- Sitting
- Standing
- Jumping
- Falling

## ✨ Features

- **Multiple Model Architectures:**
  - CNN-LSTM (spatial-temporal modeling)
  - 3D CNN (end-to-end spatio-temporal learning)
  - Simple LSTM (feature sequence modeling)

- **Complete Pipeline:**
  - Video data loading and preprocessing
  - Frame extraction and normalization
  - Model training with validation
  - Comprehensive evaluation metrics
  - Real-time inference on new videos

- **Data Management:**
  - Synthetic dataset generation
  - Automatic train/validation/test split
  - Dataset balance tracking
  - Frame sampling strategies

- **Evaluation & Visualization:**
  - Confusion matrices
  - Precision, recall, F1-score
  - Per-class performance analysis
  - Prediction visualization on video

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- NumPy, Matplotlib, Scikit-learn
- CUDA support (optional, for GPU acceleration)

## 🚀 Installation

### 1. Clone or download the repository

```bash
cd human_activity_recognition
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
human_activity_recognition/
├── model.py                 # Model architectures (CNN-LSTM, 3D CNN, etc.)
├── data_loader.py          # Data loading and preprocessing
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── inference.py            # Inference on new videos
├── utils.py                # Utility functions
├── create_dataset.py       # Dataset creation and organization
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── README.md               # This file
│
├── checkpoints/            # Saved model checkpoints
├── evaluation_results/     # Evaluation metrics and plots
└── datasets/               # Sample datasets
```

## 📊 Dataset Preparation

### Option 1: Create Synthetic Dataset

Generate synthetic video data for quick testing:

```bash
python create_dataset.py --output-dir ./datasets/synthetic --mode synthetic --num-videos 10
```

This creates a directory structure with synthetic videos:
```
datasets/synthetic/
├── walking/
├── running/
├── sitting/
├── standing/
├── jumping/
└── falling/
```

### Option 2: Use Real Videos

Organize your real videos in the following structure:

```
datasets/my_dataset/
├── walking/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── running/
│   ├── video1.mp4
│   └── ...
├── sitting/
├── standing/
├── jumping/
└── falling/
```

**Requirements for video files:**
- Format: MP4, AVI, MOV, MKV
- Resolution: Any (will be resized to 224x224)
- Frame rate: 24-60 FPS recommended
- Duration: 1-5 seconds per clip

## 🎮 Usage

### Interactive Mode (Recommended)

Run the interactive menu:

```bash
python main.py
```

This launches an interactive menu with options to:
1. Create or organize dataset
2. Train model
3. Evaluate model
4. Run inference
5. Complete pipeline

### Command-Line Mode

#### 1. Create Dataset

**Synthetic:**
```bash
python create_dataset.py \
  --output-dir ./datasets/synthetic \
  --mode synthetic \
  --num-videos 10
```

**Organize existing:**
```bash
python create_dataset.py \
  --output-dir ./datasets/my_videos \
  --mode organize
```

#### 2. Train Model

```bash
python train.py \
  --data-dir ./datasets/synthetic \
  --model-type cnnlstm \
  --batch-size 32 \
  --num-epochs 50 \
  --lr 0.001
```

**Arguments:**
- `--data-dir`: Path to dataset
- `--model-type`: cnnlstm, cnn3d, or simplelstm
- `--batch-size`: Batch size (default: 32)
- `--num-epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--checkpoint-dir`: Directory to save checkpoints (default: checkpoints)
- `--device`: cuda or cpu (auto-detect)

#### 3. Evaluate Model

```bash
python evaluate.py \
  --data-dir ./datasets/synthetic \
  --checkpoint ./checkpoints/best_model.pth \
  --model-type cnnlstm \
  --output-dir ./evaluation_results
```

This generates:
- Confusion matrix visualization
- Metrics plot (accuracy, precision, recall, F1)
- Per-class performance analysis

#### 4. Run Inference

On a single video:

```bash
python inference.py \
  --video ./test_video.mp4 \
  --checkpoint ./checkpoints/best_model.pth \
  --class-names "walking,running,sitting,standing,jumping,falling" \
  --model-type cnnlstm \
  --output-video ./output_video.mp4
```

**Arguments:**
- `--video`: Input video path
- `--checkpoint`: Model checkpoint
- `--class-names`: Comma-separated class names
- `--model-type`: Model type
- `--output-video`: Optional path to save annotated video

## 🧠 Model Architecture

### CNN-LSTM (Recommended)

```
Video (batch_size, num_frames, 3, 224, 224)
    ↓
ResNet18 CNN (extracts spatial features per frame)
    ↓
Features (batch_size, num_frames, 512)
    ↓
Bidirectional LSTM (models temporal patterns)
    ↓
Attention Layer (weights frames)
    ↓
Fully Connected Layers
    ↓
Activity Prediction (batch_size, num_classes)
```

**Advantages:**
- Captures both spatial and temporal information
- Attention mechanism highlights important frames
- Good balance between accuracy and efficiency

### 3D CNN

```
Video (batch_size, 3, num_frames, 224, 224)
    ↓
3D Convolutional Blocks (learn spatio-temporal features)
    ↓
Adaptive Average Pooling
    ↓
Fully Connected Layers
    ↓
Activity Prediction
```

**Advantages:**
- Direct spatio-temporal feature learning
- Fewer parameters than multi-stream approaches
- Good for appearance-heavy activities

## 📈 Results

### Expected Performance

On synthetic dataset (6 classes):
- Accuracy: 85-95%
- Per-class F1-score: 0.80-0.95

On real-world data:
- Accuracy: 70-85% (depends on video quality, lighting, backgrounds)
- Performance varies significantly with activity similarity

### Evaluation Metrics

The system provides:

1. **Accuracy**: Proportion of correct predictions
2. **Precision**: Correct positive predictions / all positive predictions
3. **Recall**: Correct positive predictions / all actual positives
4. **F1-Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: Shows which activities are confused

### Per-Class Analysis

The evaluation reports per-class metrics to identify:
- Which activities are easiest to recognize
- Which activities are commonly confused
- Which classes need more training data

## 🔧 Advanced Usage

### Custom Model Parameters

Edit model creation in train.py:

```python
model = create_model(
    model_type='cnnlstm',
    num_classes=6,
    num_frames=16,        # Frames per video
    hidden_dim=256,       # LSTM hidden dimension
    num_layers=2,         # LSTM layers
    dropout=0.5           # Dropout rate
)
```

### Data Augmentation

Extend preprocessing in data_loader.py:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3),
])

loader = create_data_loaders(
    root_dir='./datasets',
    transform=transform
)
```

### Custom Activities

To recognize different activities:

1. Create videos in class directories
2. Update class names in training scripts
3. Adjust training duration if needed
4. Update inference class names

## 🐛 Troubleshooting

### CUDA Out of Memory

**Solution:** Reduce batch size or frame count

```bash
python train.py --data-dir ./data --batch-size 8 --num-frames 8
```

### Video Loading Errors

**Causes & Solutions:**
- Unsupported codec: Convert to H.264 MP4
  ```bash
  ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4
  ```
- Corrupted file: Skip or regenerate
- Missing frames: Handled automatically with padding

### Poor Model Performance

**Check these:**
1. Dataset quality and balance
2. Class definitions are clear
3. Training data variation (lighting, angles, backgrounds)
4. Sufficient training epochs
5. Appropriate frame rate and resolution

**Solutions:**
- Collect more diverse data
- Increase training duration
- Adjust model architecture
- Use data augmentation
- Verify labels are correct

### GPU Not Detected

Verify CUDA installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, install CUDA-compatible PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Literature & References

**Key Concepts:**
- Simonyan & Zisserman: Two-Stream ConvNets for Action Recognition
- Karpathy et al.: Large-scale Video Classification with Convolutional Neural Networks
- Ng et al.: Beyond Short Snippets: Deep Video Understanding

**Related Datasets:**
- UCF101
- HMDB51
- Kinetics
- ActivityNet

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Suggestions for improvement:
- Multi-person tracking
- Real-time processing
- Mobile deployment
- Additional activity classes
- Improved model architectures

## ❓ FAQ

**Q: Can I use this for real-time applications?**
A: Yes, with GPU acceleration. For CPU, expect 2-5 FPS. Optimize using model compression or edge devices.

**Q: What's the minimum video length?**
A: At least 1 second (depending on FPS). The system extracts 16 frames uniformly.

**Q: How many videos do I need?**
A: Start with 20-50 per class for reasonable performance. More data improves accuracy.

**Q: Can I add new activity classes?**
A: Yes! Add class directory, collect videos, retrain model with updated class count.

**Q: What's the difference between the models?**
A: CNN-LSTM is balanced, 3D CNN is faster, SimpleLSTM is smallest. Try CNN-LSTM first.

---

**For questions or issues, refer to the troubleshooting section or check the code comments.**

Happy Activity Recognition! 🎥🤖
