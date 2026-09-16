# Mastitis Deep Learning ViT Pipeline

This repository contains the Python conversion of the original Google Colab notebook for mastitis image classification with CNN and Vision Transformer models.

## Project files

- `mastitis_pipeline.py`: converted training, evaluation, Grad-CAM, cross-validation, statistical analysis, and inference pipeline.
- `requirements.txt`: Python dependencies.
- `.gitignore`: excludes local datasets, checkpoints, generated reports, and secrets.

## Setup

Use Python 3.10 or newer. Create a virtual environment, install dependencies, then place the dataset under `data/` or set `MASTITIS_DATA_ROOT` to its location.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MASTITIS_DATA_ROOT = "$PWD\data"
py mastitis_pipeline.py
```

The dataset is expected to contain the class folders used by the notebook, including `Healthy` and `Mastitic`. Training is GPU-intensive and can take a long time on CPU.

The original notebook is intentionally not included in this Python-only project. Large datasets, model checkpoints, and generated result files are also excluded from Git.
