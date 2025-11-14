#!/bin/bash
# Fix CUDA compatibility issues for CUDA 13.0 / 12.x systems

set -e

echo "=================================="
echo "CUDA Compatibility Fix Script"
echo "=================================="
echo ""

# Detect system CUDA version
if command -v nvcc &> /dev/null; then
    SYSTEM_CUDA=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "System CUDA version: $SYSTEM_CUDA"
else
    echo "⚠️  nvcc not found, assuming CUDA 12.1+"
    SYSTEM_CUDA="12.1"
fi

# Check current PyTorch CUDA version
PYTORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "not_installed")
echo "Current PyTorch CUDA version: $PYTORCH_CUDA"
echo ""

# Determine which PyTorch version to install
if [[ "$SYSTEM_CUDA" == 13.* ]]; then
    echo "✓ Detected CUDA 13.x"
    echo "  Installing PyTorch with CUDA 12.1 (compatible with CUDA 13.x)"
    CUDA_VERSION="cu121"
elif [[ "$SYSTEM_CUDA" == 12.* ]]; then
    echo "✓ Detected CUDA 12.x"
    echo "  Installing PyTorch with CUDA 12.1"
    CUDA_VERSION="cu121"
else
    echo "⚠️  Detected CUDA $SYSTEM_CUDA"
    echo "  Installing PyTorch with CUDA 11.8 (most compatible)"
    CUDA_VERSION="cu118"
fi

echo ""
echo "=================================="
echo "Step 1: Uninstall old PyTorch"
echo "=================================="
pip uninstall -y torch torchvision torchaudio || true

echo ""
echo "=================================="
echo "Step 2: Install PyTorch with $CUDA_VERSION"
echo "=================================="

if [[ "$CUDA_VERSION" == "cu121" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "cu118" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

echo ""
echo "=================================="
echo "Step 3: Reinstall dependencies"
echo "=================================="
pip install transformers accelerate torch-geometric --upgrade

echo ""
echo "=================================="
echo "Step 4: Verify installation"
echo "=================================="
python check_cuda.py

echo ""
echo "=================================="
echo "Fix Complete!"
echo "=================================="
echo ""
echo "If you still see errors, try:"
echo "1. export CUDA_LAUNCH_BLOCKING=1"
echo "2. pip install torch --force-reinstall --no-cache-dir --index-url https://download.pytorch.org/whl/cu121"
echo ""
