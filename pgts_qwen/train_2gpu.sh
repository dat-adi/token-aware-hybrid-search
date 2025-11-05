#!/bin/bash
# PGTS Training Script for 2x24GB GPUs (48GB total)
# Optimized for memory efficiency and throughput

set -e

# Configuration
OUTPUT_DIR="outputs/training_2gpu_$(date +%Y%m%d_%H%M%S)"
MODEL_CONFIG="config/model_config_fast.yaml"
TRAIN_CONFIG="config/training_config_fast.yaml"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log system info
echo "=== System Information ==="
nvidia-smi
echo ""
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Set memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Run optimized training script
echo "=== Starting PGTS Training ==="
echo "Model config: $MODEL_CONFIG"
echo "Training config: $TRAIN_CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo ""

python train_optimized.py \
    --model_config "$MODEL_CONFIG" \
    --train_config "$TRAIN_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "=== Training Complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To resume training from a checkpoint:"
echo "python train_optimized.py \\"
echo "    --skip_reward_training \\"
echo "    --reward_model_path $OUTPUT_DIR/reward_model_final \\"
echo "    --output_dir outputs/training_resume"
