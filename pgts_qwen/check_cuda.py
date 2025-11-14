#!/usr/bin/env python3
"""
Diagnostic script to check CUDA compatibility issues.
"""
import torch
import sys

print("=" * 60)
print("CUDA Compatibility Check")
print("=" * 60)

# Check CUDA availability
print(f"\n1. PyTorch CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("   ❌ CUDA is not available to PyTorch!")
    sys.exit(1)

# PyTorch and CUDA versions
print(f"\n2. PyTorch version: {torch.__version__}")
print(f"   CUDA version (PyTorch compiled with): {torch.version.cuda}")

# System CUDA version
try:
    import subprocess
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        nvcc_output = result.stdout
        for line in nvcc_output.split('\n'):
            if 'release' in line.lower():
                print(f"   CUDA version (system): {line.strip()}")
    else:
        print("   ⚠️  Could not find nvcc (CUDA compiler)")
except FileNotFoundError:
    print("   ⚠️  nvcc not found in PATH")

# GPU information
print(f"\n3. Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    compute_capability = f"{props.major}.{props.minor}"
    print(f"   GPU {i}: {props.name}")
    print(f"      - Compute Capability: {compute_capability}")
    print(f"      - Total Memory: {props.total_memory / 1e9:.2f} GB")

# Check if compute capability is supported
print(f"\n4. PyTorch CUDA architectures: {torch.cuda.get_arch_list()}")

# Test tensor operations
print("\n5. Testing basic CUDA operations...")
try:
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = x @ y
    print("   ✓ Basic tensor operations work!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test transformers-specific operations
print("\n6. Testing transformer model loading...")
try:
    from transformers import AutoModel
    print("   ✓ Transformers library available")

    # Try to load a small model
    print("   Testing model load to CUDA...")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model = model.cuda()
    print("   ✓ Model loaded to CUDA successfully!")

    # Try forward pass
    input_ids = torch.randint(0, 1000, (1, 10)).cuda()
    outputs = model(input_ids)
    print("   ✓ Forward pass successful!")

except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("Diagnosis Complete")
print("=" * 60)

# Recommendations
print("\n📋 RECOMMENDATIONS:")
print("-" * 60)

# Check PyTorch CUDA version
pytorch_cuda_version = torch.version.cuda
if pytorch_cuda_version:
    major_version = int(pytorch_cuda_version.split('.')[0])
    if major_version < 12:
        print(f"⚠️  Your PyTorch is compiled with CUDA {pytorch_cuda_version}")
        print(f"   Your system has CUDA 13.0")
        print(f"   → Reinstall PyTorch with CUDA 12.1+ support")
        print(f"\n   Run this command:")
        print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    else:
        print(f"✓  PyTorch CUDA version ({pytorch_cuda_version}) is recent")

# Check compute capability
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    compute_cap = f"{props.major}.{props.minor}"
    arch_list = torch.cuda.get_arch_list()

    # Check if this compute capability is in the list
    if f"sm_{props.major}{props.minor}" not in arch_list:
        print(f"\n⚠️  GPU {i} ({props.name}) has compute capability {compute_cap}")
        print(f"   This may not be supported by your PyTorch installation")
        print(f"   Supported architectures: {arch_list}")

print("\n" + "=" * 60)
