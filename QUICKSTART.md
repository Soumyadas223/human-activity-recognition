# Quick Start Guide

Get up and running with Human Activity Recognition in 5 minutes!

## 🎯 5-Minute Setup

### Step 1: Install Dependencies (1 min)

```bash
pip install -r requirements.txt
```

### Step 2: Create Synthetic Dataset (1 min)

```bash
python create_dataset.py \
  --output-dir ./datasets/sample \
  --mode synthetic \
  --num-videos 5
```

### Step 3: Train Model (2 min)

```bash
python train.py \
  --data-dir ./datasets/sample \
  --batch-size 4 \
  --num-epochs 5
```

### Step 4: Test on Your Video (1 min)

```bash
python inference.py \
  --video ./datasets/sample/walking/walking_000.mp4 \
  --checkpoint ./checkpoints/best_model.pth \
  --class-names "walking,running,sitting,standing,jumping,falling" \
  --output-video ./output.mp4
```

**Done!** Check `output.mp4` for predictions.

---

## 📋 Complete Examples

### Example 1: Minimal Setup (Testing)

Perfect for validating your installation:

```bash
# Create tiny dataset
python create_dataset.py \
  --output-dir ./test_data \
  --mode synthetic \
  --num-videos 2

# Quick training
python train.py \
  --data-dir ./test_data \
  --batch-size 2 \
  --num-epochs 3 \
  --num-frames 8

# Evaluate
python evaluate.py \
  --data-dir ./test_data \
  --checkpoint ./checkpoints/best_model.pth
```

### Example 2: Production Dataset (Better Results)

```bash
# Create substantial dataset
python create_dataset.py \
  --output-dir ./production_data \
  --mode synthetic \
  --num-videos 20 \
  --video-length 120 \
  --fps 30

# Train with better parameters
python train.py \
  --data-dir ./production_data \
  --model-type cnnlstm \
  --batch-size 32 \
  --num-epochs 100 \
  --lr 0.001 \
  --checkpoint-dir ./checkpoints_prod
```

### Example 3: Real Video Dataset

```bash
# Organize your videos
python create_dataset.py \
  --output-dir ./my_videos \
  --mode organize

# Now place your MP4 files in:
# my_videos/walking/
# my_videos/running/
# my_videos/sitting/
# etc.

# Train on real data
python train.py \
  --data-dir ./my_videos \
  --model-type cnnlstm \
  --batch-size 16 \
  --num-epochs 50

# Full evaluation
python evaluate.py \
  --data-dir ./my_videos \
  --checkpoint ./checkpoints/best_model.pth \
  --output-dir ./results
```

### Example 4: Different Model Architectures

**Try CNN-LSTM (Recommended):**
```bash
python train.py \
  --data-dir ./datasets/sample \
  --model-type cnnlstm
```

**Try 3D CNN:**
```bash
python train.py \
  --data-dir ./datasets/sample \
  --model-type cnn3d
```

**Try Simple LSTM:**
```bash
python train.py \
  --data-dir ./datasets/sample \
  --model-type simplelstm
```

---

## 🎬 Using With Your Own Videos

### Step 1: Prepare Videos

Create directory structure:
```
my_activity_data/
├── walking/
│   ├── person1_walking.mp4
│   ├── person2_walking.mp4
│   └── ...
├── running/
│   ├── person1_running.mp4
│   └── ...
└── ... (other activities)
```

**Tips:**
- MP4 format recommended
- 1-5 seconds per video
- Multiple angles and people for better accuracy
- ~20-50 videos per activity

### Step 2: Train

```bash
python train.py \
  --data-dir ./my_activity_data \
  --model-type cnnlstm \
  --batch-size 16 \
  --num-epochs 50
```

### Step 3: Test

```bash
python inference.py \
  --video ./my_activity_data/walking/test_video.mp4 \
  --checkpoint ./checkpoints/best_model.pth \
  --class-names "walking,running,sitting,standing,jumping,falling" \
  --output-video ./output_with_predictions.mp4
```

---

## 🖥️ Using Interactive Mode

Simply run:
```bash
python main.py
```

Then follow the menu prompts:
```
==============================================================
Human Activity Recognition System - Main Menu
==============================================================
1. Create or organize dataset
2. Train the model
3. Evaluate the model
4. Run inference on video
5. Run complete pipeline (1 -> 2 -> 3 -> 4)
0. Exit
==============================================================
Select option: 
```

---

## 📊 Performance Tips

### For Faster Training:
```bash
python train.py \
  --data-dir ./datasets/sample \
  --batch-size 64 \
  --num-frames 8 \
  --num-epochs 20
```

### For Better Accuracy:
```bash
python train.py \
  --data-dir ./datasets/sample \
  --batch-size 16 \
  --num-frames 16 \
  --num-epochs 100 \
  --lr 0.001
```

### For GPU Memory Constraints:
```bash
python train.py \
  --data-dir ./datasets/sample \
  --batch-size 8 \
  --num-frames 8 \
  --device cpu
```

---

## 🔍 Understanding Output

### Training Output
```
Epoch 1/50
Training: 45%|████▌     | 45/100 [00:15<00:20, 2.75it/s, loss=0.85]
Train Loss: 0.8524 | Train Acc: 0.6234
Val Loss: 0.7834 | Val Acc: 0.6892
Saved best model with val loss: 0.7834
```

### Evaluation Output
```
==================================================
EVALUATION METRICS REPORT
==================================================
ACCURACY          : 0.8524
PRECISION         : 0.8612
RECALL            : 0.8524
F1_SCORE          : 0.8528

-------------------------------------------------
PER-CLASS ANALYSIS
-------------------------------------------------
walking:
  Precision: 0.9234
  Recall:    0.9123
  F1-Score:  0.9178
```

### Inference Output
```
Prediction: walking
Confidence: 0.9523

All probabilities:
  walking: 0.9523
  running: 0.0234
  sitting: 0.0145
  standing: 0.0089
  jumping: 0.0006
  falling: 0.0003
```

---

## 🐛 Common Issues & Quick Fixes

| Issue | Solution |
|-------|----------|
| "No module named torch" | `pip install -r requirements.txt` |
| CUDA out of memory | Reduce `--batch-size` or `--num-frames` |
| Video loading error | Convert to MP4: `ffmpeg -i input.avi output.mp4` |
| Poor accuracy | Get more diverse training videos |
| Slow training | Use `--num-frames 8` instead of 16 |
| GPU not detected | Check CUDA installation or use `--device cpu` |

---

## 📈 Next Steps

After running the quick start:

1. **Improve Accuracy:**
   - Collect real videos instead of synthetic
   - Add more videos per class
   - Increase training epochs
   - Try different model architectures

2. **Deploy to Production:**
   - Optimize model with quantization
   - Convert to ONNX format
   - Deploy on edge devices

3. **Extend Functionality:**
   - Add more activity classes
   - Implement multi-person tracking
   - Add real-time webcam processing
   - Create web/mobile interface

4. **Experiment:**
   - Try different batch sizes
   - Adjust learning rates
   - Implement custom data augmentation
   - Compare model architectures

---

## 📚 Useful Commands Reference

```bash
# Create dataset
python create_dataset.py --output-dir ./data --mode synthetic --num-videos 10

# Train model
python train.py --data-dir ./data --batch-size 32 --num-epochs 50

# Evaluate model
python evaluate.py --data-dir ./data --checkpoint ./checkpoints/best_model.pth

# Run inference
python inference.py --video video.mp4 --checkpoint ./checkpoints/best_model.pth \
  --class-names "walking,running,sitting,standing,jumping,falling"

# Interactive mode
python main.py

# Full pipeline
python main.py --mode train --data-dir ./data --epochs 50
```

---

## 💡 Tips for Best Results

✅ **Do:**
- Use multiple videos per activity (20+ recommended)
- Include variation: different people, angles, lighting
- Use clear, high-quality videos
- Balance classes with similar number of videos
- Train for sufficient epochs (50-100)

❌ **Don't:**
- Use very short videos (<1 second)
- Train on single video clip
- Use low-resolution videos
- Train only 2-3 epochs
- Use imbalanced datasets

---

## 🎓 Learning More

1. Read the main [README.md](README.md)
2. Check code comments in source files
3. Experiment with different parameters
4. Review evaluation metrics closely
5. Analyze confusion matrix to find weak areas

---

**Now you're ready! Start with Example 1 and explore from there.** 🚀
