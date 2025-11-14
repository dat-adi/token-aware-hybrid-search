# IMMEDIATE FIX for "no kernel image is available" Error

## Your Issue

You're on **CUDA 13.0** but PyTorch was compiled with an older CUDA version.

---

## Quick Fix (2 minutes)

### Option 1: Automatic (Recommended)

```bash
# Run the auto-fix script
cd /path/to/pgts_qwen
chmod +x fix_cuda.sh check_cuda.py
./fix_cuda.sh
```

This will:
1. Detect your CUDA version
2. Uninstall old PyTorch
3. Install compatible PyTorch with CUDA 12.1 (works with CUDA 13.0)
4. Verify the installation

### Option 2: Manual (if auto-fix fails)

```bash
# Step 1: Uninstall old PyTorch
pip uninstall torch torchvision torchaudio -y

# Step 2: Install PyTorch with CUDA 12.1 (compatible with CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 3: Reinstall dependencies
pip install transformers accelerate torch-geometric --upgrade

# Step 4: Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); x = torch.randn(10,10).cuda(); print('✓ Works!')"
```

---

## Then Resume Training

After fixing CUDA:

```bash
# Add these environment variables (helps with stability)
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Resume training
./train_2gpu.sh
```

---

## If Still Not Working

### Check your setup
```bash
python check_cuda.py
```

This will tell you:
- Your system CUDA version
- Your PyTorch CUDA version
- Your GPU compute capability
- What's wrong

### Common issues:

**Issue**: Still seeing the error after reinstall
**Fix**: Clear pip cache
```bash
pip cache purge
pip install torch --force-reinstall --no-cache-dir --index-url https://download.pytorch.org/whl/cu121
```

**Issue**: "ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS'"
**Fix**: Upgrade transformers
```bash
pip install transformers --upgrade
```

**Issue**: torch-geometric errors
**Fix**: Reinstall PyG
```bash
pip uninstall torch-geometric torch-scatter torch-sparse -y
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

---

## Full Documentation

See `CUDA_TROUBLESHOOTING.md` for comprehensive troubleshooting guide.

---

## Why This Happens

| Component | Version | Issue |
|-----------|---------|-------|
| Your server CUDA | 13.0 | ✓ Latest |
| PyTorch (old install) | Compiled for CUDA 11.8 | ❌ Too old |
| PyTorch (after fix) | Compiled for CUDA 12.1 | ✓ Compatible! |

**CUDA 12.1 is forward-compatible with CUDA 13.0**, so PyTorch with cu121 will work!

---

## Quick Test After Fix

```python
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')

# Test CUDA operation
x = torch.randn(100, 100).cuda()
y = x @ x
print('✓✓✓ CUDA is working! ✓✓✓')
"
```

If this runs without errors, you're good to go!
