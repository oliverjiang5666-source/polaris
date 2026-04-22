#!/bin/bash
# Setup environment on cloud GPU (AutoDL / any Ubuntu+CUDA machine)
# Run once after uploading and extracting the tarball

set -e

echo "=== Setting up GPU environment ==="

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || \
pip install torch torchvision

pip install lightgbm loguru scikit-learn pandas pyarrow numpy scipy

# Create needed directories
mkdir -p models
mkdir -p data/china/processed

# Ensure __init__.py files exist
touch forecast/__init__.py
touch oracle/__init__.py
touch data/__init__.py
touch data/china/__init__.py

# Verify CUDA
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo "=== Setup complete ==="
