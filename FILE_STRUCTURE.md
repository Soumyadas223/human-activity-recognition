# File Structure & Documentation

Complete guide to all files in the Human Activity Recognition System.

## 📁 Project Files Overview

### Core Application Files

#### 1. **model.py** (525 lines)
**Purpose:** Model architecture definitions

**Contains:**
- `ActivityRecognitionCNNLSTM`: CNN-LSTM architecture for spatial-temporal learning
- `ActivityRecognitionCNN3D`: 3D CNN architecture for direct spatio-temporal feature learning
- `ActivityRecognitionSimpleLSTM`: Simple LSTM for feature sequences
- `create_model()`: Factory function to instantiate models
- Attention mechanism implementation
- Weight initialization utilities

**Key Components:**
```python
# CNN-LSTM: ResNet18 + Bidirectional LSTM + Attention
model = ActivityRecognitionCNNLSTM(
    num_classes=6,
    num_frames=16,
    hidden_dim=256,
    num_layers=2,
    dropout=0.5
)

# 3D CNN: Direct spatio-temporal convolutions
model = ActivityRecognitionCNN3D(
    num_classes=6,
    dropout=0.5
)

# Simple LSTM: Feature-based temporal modeling
model = ActivityRecognitionSimpleLSTM(
    num_classes=6,
    input_size=512,
    hidden_dim=256
)
```

---

#### 2. **data_loader.py** (530 lines)
**Purpose:** Video data loading and preprocessing

**Main Classes:**
- `VideoActivityDataset`: Loads videos and extracts frames
- `FrameFeatureDataset`: For pre-extracted features
- `Utility Functions`: Frame extraction, splitting, loader creation

**Key Features:**
```python
# Load dataset with automatic preprocessing
dataset = VideoActivityDataset(
    root_dir='./datasets',
    num_frames=16,
    frame_size=(224, 224),
    sampling_strategy='uniform'
)

# Create train/val/test loaders
train_loader, val_loader, test_loader, info = create_data_loaders(
    root_dir='./datasets',
    batch_size=32,
    num_frames=16
)

# Load single video for inference
loader = create_single_video_loader(
    video_path='video.mp4',
    num_frames=16
)
```

**Supported Formats:**
- MP4, AVI, MOV, MKV video files
- Automatic frame extraction and resizing
- Flexible sampling strategies (uniform, random)

---

#### 3. **train.py** (365 lines)
**Purpose:** Model training pipeline

**Main Class:**
- `Trainer`: Handles training, validation, checkpointing

**Features:**
```python
# Initialize trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    dataset_info=info,
    device=device
)

# Train model
history = trainer.train(num_epochs=50)
```

**Command-line Usage:**
```bash
python train.py \
  --data-dir ./datasets \
  --model-type cnnlstm \
  --batch-size 32 \
  --num-epochs 50 \
  --lr 0.001
```

**Outputs:**
- Checkpoints saved to `./checkpoints/best_model.pth`
- Training history (loss, accuracy curves)
- Early stopping support
- Learning rate scheduling

---

#### 4. **evaluate.py** (265 lines)
**Purpose:** Model evaluation and performance analysis

**Main Class:**
- `Evaluator`: Comprehensive model evaluation

**Features:**
```python
# Evaluate model
evaluator = Evaluator(
    model=model,
    test_loader=test_loader,
    dataset_info=info,
    device=device
)

metrics, cm, probs = evaluator.evaluate()
evaluator.analyze_per_class()
```

**Generates:**
- Confusion matrix visualization
- Accuracy, precision, recall, F1-score metrics
- Per-class performance breakdown
- Prediction probability distributions

**Command-line Usage:**
```bash
python evaluate.py \
  --data-dir ./datasets \
  --checkpoint ./checkpoints/best_model.pth \
  --output-dir ./evaluation_results
```

---

#### 5. **inference.py** (330 lines)
**Purpose:** Prediction on new videos

**Main Class:**
- `ActivityRecognizer`: Video-based activity prediction engine

**Features:**
```python
# Create recognizer
recognizer = ActivityRecognizer(
    model=model,
    class_names=['walking', 'running', ...],
    device=device
)

# Predict on video
predictions = recognizer.predict_video('video.mp4')

# Get predictions with annotated frames
frames, predictions = recognizer.predict_video_with_frames('video.mp4')
```

**Command-line Usage:**
```bash
python inference.py \
  --video test.mp4 \
  --checkpoint ./checkpoints/best_model.pth \
  --class-names "walking,running,sitting,standing,jumping,falling" \
  --output-video output.mp4
```

**Output Format:**
```python
{
    'class': 'walking',
    'class_idx': 0,
    'confidence': 0.9523,
    'all_probabilities': {
        'walking': 0.9523,
        'running': 0.0234,
        ...
    }
}
```

---

#### 6. **utils.py** (370 lines)
**Purpose:** Utility functions and helpers

**Main Classes:**
- `MetricsCalculator`: Compute metrics (accuracy, precision, recall, F1)

**Key Functions:**
```python
# Metrics
metrics = MetricsCalculator(num_classes, class_names)
metrics.update(predictions, targets)
scores = metrics.get_metrics()  # accuracy, precision, recall, f1
cm = metrics.get_confusion_matrix()

# Visualization
plot_confusion_matrix(cm, class_names, save_path='cm.png')
plot_metrics(metrics_dict, save_path='metrics.png')

# Video processing
frame = draw_activity_label(frame, 'walking', 0.95)
frame = draw_bounding_box(frame, [x1, y1, x2, y2])
save_video(frames, 'output.mp4', fps=30)

# Checkpointing
save_checkpoint(model, optimizer, epoch, loss, 'model.pth')
model, info = load_checkpoint(model, 'model.pth', device)

# Reporting
print_metrics_report(metrics, cm, class_names)
```

---

#### 7. **create_dataset.py** (420 lines)
**Purpose:** Dataset creation and organization

**Main Classes:**
- `SyntheticDatasetGenerator`: Generate synthetic training data
- `DatasetOrganizer`: Organize existing video files

**Features:**
```python
# Generate synthetic dataset
generator = SyntheticDatasetGenerator(
    output_dir='./datasets',
    num_videos_per_class=10,
    video_length=60,
    fps=30
)
generator.generate_dataset()

# Organize real videos
organizer = DatasetOrganizer('./datasets')
organizer.create_class_structure(['walking', 'running', ...])
organizer.print_instructions()
```

**Command-line Usage:**
```bash
# Synthetic
python create_dataset.py \
  --output-dir ./datasets \
  --mode synthetic \
  --num-videos 10

# Organize existing
python create_dataset.py \
  --output-dir ./datasets \
  --mode organize
```

**Generated Activities:**
- Walking (person moving with legs)
- Running (faster movement)
- Sitting (bent posture)
- Standing (upright posture)
- Jumping (upward displacement)
- Falling (rapid descent)

---

#### 8. **main.py** (380 lines)
**Purpose:** Entry point with interactive menu

**Main Class:**
- `PipelineManager`: Manages complete workflow

**Features:**
- Interactive menu system
- Command-line interface
- Complete pipeline orchestration
- User-friendly workflow

**Usage:**
```bash
# Interactive mode
python main.py

# Command-line mode
python main.py --mode train --data-dir ./data --epochs 50
python main.py --mode infer --video test.mp4 --checkpoint model.pth
```

---

### Documentation Files

#### 9. **README.md**
Comprehensive documentation covering:
- Project overview and features
- Installation instructions
- Dataset preparation
- Usage guide (interactive and CLI)
- Model architecture explanations
- Troubleshooting guide
- FAQ section

#### 10. **QUICKSTART.md**
Quick start guide with:
- 5-minute setup instructions
- 4 complete examples
- Performance tips
- Common issues and solutions
- Command reference
- Best practices

#### 11. **FILE_STRUCTURE.md** (This File)
Complete file structure documentation

#### 12. **config.yaml**
Configuration template for:
- Dataset parameters
- Model architecture
- Training hyperparameters
- Evaluation settings
- Data augmentation
- Experiment metadata

#### 13. **requirements.txt**
Python dependencies:
```
torch==2.0.0
torchvision==0.15.0
opencv-python==4.8.0.74
opencv-contrib-python==4.8.0.74
numpy==1.24.3
matplotlib==3.7.2
scikit-learn==1.3.0
tqdm==4.65.0
Pillow==10.0.0
pandas==2.0.3
seaborn==0.12.2
```

---

## 📊 Data Directory Structure

After setup, you'll have:

```
project_root/
├── datasets/
│   ├── sample/
│   │   ├── walking/
│   │   ├── running/
│   │   ├── sitting/
│   │   ├── standing/
│   │   ├── jumping/
│   │   └── falling/
│   └── my_dataset/
│       └── (your videos here)
│
├── checkpoints/
│   └── best_model.pth
│
├── evaluation_results/
│   ├── confusion_matrix.png
│   └── metrics.png
│
└── inference_results/
    ├── output_video.mp4
    └── predictions.json
```

---

## 🔄 Workflow Overview

### 1. Dataset Creation
```
create_dataset.py
    ↓
datasets/
├── activity1/
├── activity2/
└── ...
```

### 2. Data Preparation
```
data_loader.py
    ↓
VideoActivityDataset
    ↓
DataLoaders (train/val/test)
```

### 3. Model Training
```
train.py
    ↓
Trainer class
    ↓
checkpoints/best_model.pth
```

### 4. Evaluation
```
evaluate.py
    ↓
Evaluator class
    ↓
evaluation_results/
├── confusion_matrix.png
└── metrics.png
```

### 5. Inference
```
inference.py
    ↓
ActivityRecognizer
    ↓
inference_results/output.mp4
```

---

## 🎯 File Dependencies

```
main.py
├── train.py
│   ├── model.py
│   ├── data_loader.py
│   └── utils.py
├── evaluate.py
│   ├── model.py
│   ├── data_loader.py
│   └── utils.py
├── inference.py
│   ├── model.py
│   ├── data_loader.py
│   └── utils.py
├── create_dataset.py
│   └── (standalone)
└── config.yaml (optional)
```

---

## 📝 File Sizes & Complexity

| File | Lines | Complexity | Purpose |
|------|-------|-----------|---------|
| model.py | 525 | High | Model architectures |
| data_loader.py | 530 | High | Data pipeline |
| train.py | 365 | Medium | Training loop |
| evaluate.py | 265 | Medium | Evaluation metrics |
| inference.py | 330 | Medium | Prediction engine |
| utils.py | 370 | Medium | Helper functions |
| create_dataset.py | 420 | Medium | Dataset generation |
| main.py | 380 | Medium | CLI & menu |

**Total: ~3,200 lines of code**

---

## 🚀 Getting Started

### Minimal Setup
1. Copy all .py files to your project folder
2. Run `pip install -r requirements.txt`
3. Run `python main.py` and follow prompts

### With Custom Videos
1. Organize videos in `datasets/mydata/activity/` structure
2. Run `python train.py --data-dir ./datasets/mydata`
3. Run `python inference.py --video test.mp4 --checkpoint ./checkpoints/best_model.pth`

### Development
1. Modify model architectures in `model.py`
2. Adjust training in `train.py`
3. Extend evaluation in `evaluate.py`
4. Add features in `utils.py`

---

## 🔧 Customization Points

### Easy to Modify
- Number of activity classes
- Model hyperparameters
- Training duration
- Batch size
- Learning rate

### Medium Complexity
- Model architecture
- Data preprocessing
- Loss function
- Optimizer type

### Advanced
- Training strategies
- Custom layers
- Multi-task learning
- Real-time processing

---

## 📚 Code Organization Principles

1. **Modularity**: Each file has single responsibility
2. **Reusability**: Functions work independently
3. **Clarity**: Clear variable and function names
4. **Documentation**: Docstrings and comments throughout
5. **Flexibility**: Configuration parameters instead of hardcoding

---

## 🎓 Learning Path

1. **Start**: Read QUICKSTART.md
2. **Basics**: Run Example 1 (synthetic dataset)
3. **Understanding**: Study model.py and data_loader.py
4. **Advanced**: Modify parameters in train.py
5. **Deep**: Customize model architectures
6. **Production**: Deploy with inference.py

---

**For detailed information on each file, see the inline comments and docstrings in the code.**
