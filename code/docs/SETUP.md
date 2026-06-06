# Setup Guide

## Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/zebrafish-heart-imaging.git
cd zebrafish-heart-imaging
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
pytest tests/ -v
```

## Troubleshooting

### CUDA/GPU Support

If you want GPU support with PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### macOS (Apple Silicon)

For M1/M2 Macs, use the Metal Performance Shaders backend:

```bash
pip install torch::nightly -c pytorch-nightly
```

And set device to "mps" in configs.

### Virtual Environment Issues

If `source venv/bin/activate` doesn't work, try:

```bash
# Using conda instead
conda create -n zebrafish-heart python=3.10
conda activate zebrafish-heart
pip install -r requirements.txt
```

## Project Structure Overview

See `README.md` for detailed directory structure explanation.

## Next Steps

1. Review the data organization in `data/raw/`
2. Check out example notebooks in `notebooks/`
3. Configure preprocessing in `configs/preprocessing.yaml`
4. Run preprocessing: `python src/preprocessing/process_images.py`
5. Train a model: `python src/train.py --config configs/training.yaml`

## Getting Help

- Check the README.md for common commands
- Review documentation in `docs/`
- Look at test examples in `tests/`
- Check logs in `logs/` for detailed error messages
