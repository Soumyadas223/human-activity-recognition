# Installation & Setup Guide

Complete step-by-step installation instructions for the Human Activity Recognition System.

## 🖥️ System Requirements

### Minimum Requirements
- **CPU**: Intel Core i5 or equivalent
- **RAM**: 8GB
- **Storage**: 10GB free space (for datasets and models)
- **OS**: Windows, macOS, or Linux

### Recommended for GPU Acceleration
- **GPU**: NVIDIA GPU with CUDA capability
- **RAM**: 16GB
- **CUDA**: 11.8 or higher
- **cuDNN**: 8.6 or higher

## 📋 Pre-Installation Checklist

- [ ] Python 3.8+ installed
- [ ] pip package manager working
- [ ] Git installed (optional, for cloning)
- [ ] Internet connection for downloading packages
- [ ] 20GB free disk space

## 🚀 Installation Steps

### Step 1: Verify Python Installation

```bash
python --version
pip --version
```

**Expected output:**
```
Python 3.8.0 (or higher)
pip 21.0 (or higher)
```

If not installed, download from [python.org](https://www.python.org/downloads/)

### Step 2: Create Project Directory

```bash
# Create and navigate to project folder
mkdir human_activity_recognition
cd human_activity_recognition
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

**You should see `(venv)` in your terminal prompt**

### Step 4: Copy Project Files

Copy these 13 files to your project directory:
```
1. model.py
2. data_loader.py
3. train.py
4. evaluate.py
5. inference.py
6. utils.py
7. create_dataset.py
8. main.py
9. requirements.txt
10. README.md
11. QUICKSTART.md
12. FILE_STRUCTURE.md
13. config.yaml
```

### Step 5: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all packages
pip install -r requirements.txt
```

**Installation takes 5-15 minutes depending on internet speed**

### Step 6: Verify Installation

```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Test OpenCV
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# Test CUDA (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test all imports
python -c "from model import create_model; from data_loader import create_data_loaders; print('✓ All imports successful')"
```

**Expected output:**
```
PyTorch version: 2.0.0
OpenCV version: 4.8.0
CUDA available: True (or False)
✓ All imports successful
```

## 🖼️ GPU Setup (Optional)

### NVIDIA GPU with CUDA

If you have NVIDIA GPU and want to use it:

#### 1. Check if NVIDIA GPU is available
```bash
# Windows
wmic logicaldisk get name

# macOS/Linux
nvidia-smi
```

#### 2. Install CUDA (if not installed)
- Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Follow installation instructions for your OS

#### 3. Install cuDNN
- Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
- Extract and add to system PATH

#### 4. Install PyTorch with CUDA support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 5. Verify CUDA setup
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 🔧 Troubleshooting Installation

### Issue: "No module named 'torch'"

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall PyTorch
pip install torch==2.0.0
```

### Issue: "pip: command not found"

**Solution:**
```bash
# Use python -m pip instead
python -m pip install -r requirements.txt
```

### Issue: CUDA not detected

**Solution:**
```bash
# Use CPU version instead
python -c "import torch; torch.device('cpu')"

# Run training with CPU
python train.py --data-dir ./data --device cpu
```

### Issue: Permission denied

**Solution:**
```bash
# Windows: Run command prompt as Administrator
# macOS/Linux: Use sudo or check folder permissions
chmod -R 755 human_activity_recognition
```

### Issue: Disk space error during installation

**Solution:**
```bash
# Clear pip cache
pip cache purge

# Install packages one by one
pip install torch
pip install torchvision
pip install opencv-python
# etc.
```

## ✅ Post-Installation Verification

### Quick Test

```bash
# Test 1: Create synthetic dataset
python create_dataset.py \
  --output-dir ./test_data \
  --mode synthetic \
  --num-videos 2

# Test 2: Train on test data (should complete in <5 minutes)
python train.py \
  --data-dir ./test_data \
  --batch-size 2 \
  --num-epochs 2

# Test 3: Evaluate
python evaluate.py \
  --data-dir ./test_data \
  --checkpoint ./checkpoints/best_model.pth

# Test 4: Run inference
python inference.py \
  --video ./test_data/walking/walking_000.mp4 \
  --checkpoint ./checkpoints/best_model.pth \
  --class-names "walking,running,sitting,standing,jumping,falling"
```

**If all tests complete successfully, installation is correct!** ✅

## 📁 Project Structure After Setup

```
human_activity_recognition/
├── venv/                          # Virtual environment
├── model.py                       # Model architectures
├── data_loader.py                # Data loading
├── train.py                      # Training script
├── evaluate.py                   # Evaluation
├── inference.py                  # Inference
├── utils.py                      # Utilities
├── create_dataset.py             # Dataset creation
├── main.py                       # Main entry point
├── requirements.txt              # Dependencies
├── README.md                     # Documentation
├── QUICKSTART.md                 # Quick start guide
├── FILE_STRUCTURE.md             # File documentation
├── config.yaml                   # Configuration
├── INSTALLATION_GUIDE.md         # This file
│
├── datasets/                      # Data directory (created later)
├── checkpoints/                  # Model checkpoints (created later)
├── evaluation_results/           # Eval results (created later)
└── logs/                         # Logs (created later)
```

## 🎯 Next Steps After Installation

1. **Read Quick Start**: `cat QUICKSTART.md`
2. **Run Example**: `python main.py`
3. **Create Dataset**: `python create_dataset.py --output-dir ./datasets/demo --mode synthetic --num-videos 5`
4. **Train Model**: `python train.py --data-dir ./datasets/demo --num-epochs 10`
5. **Test Inference**: `python inference.py --video ./datasets/demo/walking/walking_000.mp4 --checkpoint ./checkpoints/best_model.pth --class-names "walking,running,sitting,standing,jumping,falling"`

## 🔄 Updating Installation

### Update Single Package

```bash
pip install --upgrade torch
```

### Update All Packages

```bash
pip install --upgrade -r requirements.txt
```

### Clean Installation

```bash
# Remove virtual environment
rm -rf venv  # macOS/Linux
rmdir /s venv  # Windows

# Recreate from scratch
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate
pip install -r requirements.txt
```

## 🆘 Getting Help

### Check Installation
```bash
python -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
import cv2
print(f'OpenCV: {cv2.__version__}')
import numpy as np
print(f'NumPy: {np.__version__}')
"
```

### Check Specific Module
```bash
python -c "from model import create_model; print('✓ model.py OK')"
python -c "from data_loader import create_data_loaders; print('✓ data_loader.py OK')"
python -c "from train import Trainer; print('✓ train.py OK')"
```

### Common Solutions
1. Always use virtual environment
2. Make sure venv is activated (see `(venv)` in prompt)
3. Use Python 3.8 or higher
4. Install exact versions from requirements.txt
5. Check disk space (20GB minimum)

## 📞 Support Resources

- **PyTorch Issues**: https://github.com/pytorch/pytorch/issues
- **OpenCV Issues**: https://github.com/opencv/opencv/issues
- **Python Issues**: https://www.python.org/community/

## ✨ Successful Installation Indicators

You'll know installation is complete when you can:
- ✅ Import all modules without errors
- ✅ Create synthetic dataset
- ✅ Train model (at least for 1 epoch)
- ✅ Evaluate on test data
- ✅ Run inference on video

---

**Installation complete! Ready to use the Human Activity Recognition System.** 🚀
