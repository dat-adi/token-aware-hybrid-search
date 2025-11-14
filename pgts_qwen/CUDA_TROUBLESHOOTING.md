# CUDA Troubleshooting Guide

## Error: "no kernel image is available for execution on the device"

This error occurs when your PyTorch installation doesn't have CUDA kernels compiled for your specific GPU or CUDA version.

### Quick Fix (Most Common Solution)

```bash
# 1. Run diagnostic
python check_cuda.py

# 2. Run auto-fix
chmod +x fix_cuda.sh
./fix_cuda.sh

# 3. Test the fix
python check_cuda.py
```

---

## Understanding the Problem

### CUDA Version Mismatch

Your system has **CUDA 13.0**, but PyTorch might be compiled with an older CUDA version (like 11.8 or 12.1).

**The good news**: PyTorch compiled with CUDA 12.1 works with CUDA 13.0 due to forward compatibility!

### Common Scenarios

| Your CUDA | PyTorch CUDA | Compatible? | Solution |
|-----------|--------------|-------------|----------|
| 13.0 | 11.8 | ❌ No | Reinstall with cu121 |
| 13.0 | 12.1 | ✅ Yes | Should work |
| 12.x | 11.8 | ⚠️ Maybe | Reinstall with cu121 |
| 12.x | 12.1 | ✅ Yes | Should work |

---

## Manual Fix Instructions

### Step 1: Check Your Setup

```bash
# Check system CUDA version
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Reinstall PyTorch

**For CUDA 12.1+ / 13.0:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Reinstall Other Dependencies

```bash
# Use the CUDA-specific requirements file
pip install -r requirements_cuda121.txt
```

### Step 4: Verify Fix

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')

# Test operation
x = torch.randn(10, 10).cuda()
y = x @ x
print('✓ CUDA operations working!')
"
```

---

## Alternative Solutions

### Option 1: Use PyTorch Nightly (Most Cutting-Edge)

```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

### Option 2: Build PyTorch from Source (Advanced)

Only if pre-built wheels don't work:

```bash
# This takes hours and requires CUDA toolkit
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"
python setup.py install
```

### Option 3: Use Docker with Pre-configured Environment

```bash
# Use NVIDIA's PyTorch container
docker pull nvcr.io/nvidia/pytorch:24.01-py3
docker run --gpus all -it nvcr.io/nvidia/pytorch:24.01-py3
```

---

## Environment Variable Fixes

Sometimes setting these helps:

```bash
# Add to your ~/.bashrc or training script

# More detailed error messages
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Disable tokenizer parallelism (can cause issues)
export TOKENIZERS_PARALLELISM=false

# Force CUDA device
export CUDA_VISIBLE_DEVICES=0,1
```

Then source it:
```bash
source ~/.bashrc
# or add to your training script
```

---

## GPU Compute Capability Issues

Your error might also be caused by GPU compute capability mismatch.

### Check Your GPU Compute Capability

```bash
python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {props.name}')
    print(f'  Compute Capability: {props.major}.{props.minor}')
"
```

### Common GPUs and Their Compute Capabilities

| GPU | Compute Capability | Supported? |
|-----|-------------------|------------|
| RTX 4090 | 8.9 | ✅ Yes (cu121) |
| RTX 4080 | 8.9 | ✅ Yes (cu121) |
| RTX 3090 | 8.6 | ✅ Yes |
| RTX 3080 | 8.6 | ✅ Yes |
| A100 | 8.0 | ✅ Yes |
| V100 | 7.0 | ✅ Yes |
| T4 | 7.5 | ✅ Yes |
| Older GPUs | < 7.0 | ⚠️ Limited |

### If Your GPU Has Compute Capability < 7.0

You'll need to build PyTorch from source or use an older PyTorch version:

```bash
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

---

## Transformers-Specific Issues

### Issue: Works with PyTorch, fails with Transformers

The Transformers library might have its own CUDA kernel compilation.

**Fix:**
```bash
# Reinstall transformers from source
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers.git

# Or use a specific version
pip install transformers==4.40.0 --force-reinstall --no-cache-dir
```

### Issue: Flash Attention Errors

Flash Attention requires specific CUDA versions.

**Fix:**
```bash
# Uninstall
pip uninstall flash-attn -y

# Reinstall for your CUDA version
pip install flash-attn --no-build-isolation

# Or disable flash attention in your config
# In model_config_2gpu.yaml, set:
# use_flash_attention: false
```

---

## Testing the Fix

After applying any fix, run this comprehensive test:

```bash
python check_cuda.py
```

If that passes, test with actual model:

```python
python -c "
from transformers import AutoModel, AutoTokenizer
import torch

print('Loading model...')
model = AutoModel.from_pretrained('prajjwal1/bert-tiny').cuda()
tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')

print('Testing inference...')
inputs = tokenizer('Hello world', return_tensors='pt')
inputs = {k: v.cuda() for k, v in inputs.items()}
outputs = model(**inputs)

print('✓ Everything works!')
print(f'Output shape: {outputs.last_hidden_state.shape}')
"
```

---

## Still Not Working?

### Debug Mode

Run your training with full debug output:

```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TORCH_SHOW_CPP_STACKTRACES=1

python train_optimized.py --model_config config/model_config_2gpu.yaml \
                          --train_config config/training_config_2gpu.yaml \
                          --output_dir outputs/debug
```

### Check torch-geometric

torch-geometric can also cause CUDA issues:

```bash
# Uninstall
pip uninstall torch-geometric torch-scatter torch-sparse -y

# Reinstall for your PyTorch/CUDA version
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Last Resort: CPU-Only Testing

To verify your code works (but slowly):

```python
# Edit model_config_2gpu.yaml
qwen3:
  device_map: "cpu"  # Force CPU

reward_model:
  device: "cpu"
```

Then run training. If it works on CPU, it's definitely a CUDA issue.

---

## Prevention for Next Time

Create a fresh conda environment with specific versions:

```bash
conda create -n pgts python=3.10
conda activate pgts

# Install PyTorch first with correct CUDA
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Then install other requirements
pip install -r requirements_cuda121.txt
```

---

## Quick Reference

### Diagnose
```bash
python check_cuda.py
```

### Auto-fix
```bash
./fix_cuda.sh
```

### Manual fix for CUDA 13.0/12.x
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_cuda121.txt
```

### Test
```bash
python -c "import torch; x = torch.randn(10,10).cuda(); print('✓ Works!')"
```

---

## Contact / Issues

If none of these solutions work, please provide:

1. Output of `python check_cuda.py`
2. Output of `nvcc --version`
3. Output of `nvidia-smi`
4. Full error traceback
5. Your GPU model

This helps diagnose exotic issues!
