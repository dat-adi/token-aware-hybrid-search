#!/bin/bash
# PGTS Training Script for 2x24GB GPUs (48GB total)
# Optimized for memory efficiency and throughput

set -e
echo "MWAHAHAHA"
# Configuration
REWARD_MODEL_PATH="/iacl/pg23/prahlad/rl/tahs/pgts_qwen/outputs/training_2gpu_20251511/reward_model_final"
POLICY_OUTPUT_DIR="outputs/policy_training_$(date +%Y%m%d_%H%M%S)"
MODEL_CONFIG="config/model_config_fast.yaml"
TRAIN_CONFIG="config/training_config_fast.yaml"

# Create output directory for policy training
mkdir -p "$POLICY_OUTPUT_DIR"

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
echo "=== Starting PGTS Policy Training (Stage 2) ==="
echo "Model config: $MODEL_CONFIG"
echo "Training config: $TRAIN_CONFIG"
echo "Reward Model Path: $REWARD_MODEL_PATH"
echo "Policy Output Dir: $POLICY_OUTPUT_DIR"
echo ""

python train_optimized.py \
    --model_config "$MODEL_CONFIG" \
    --train_config "$TRAIN_CONFIG" \
    --output_dir "$POLICY_OUTPUT_DIR" \
    --skip_reward_training \
    --reward_model_path "$REWARD_MODEL_PATH" \
    2>&1 | tee "$POLICY_OUTPUT_DIR/training.log"

echo "=== Training Complete ==="
echo "Policy training results saved to: $POLICY_OUTPUT_DIR"
echo ""
echo "Logs available at:"
echo "  - Console log: $POLICY_OUTPUT_DIR/training.log"
echo "  - Detailed log: $POLICY_OUTPUT_DIR/training_detailed.log"
echo ""
echo "To resume training from a checkpoint:"
echo "python train_optimized.py \\"
echo "    --skip_reward_training \\"
echo "    --reward_model_path \"$REWARD_MODEL_PATH\" \\"
echo "    --output_dir \"$POLICY_OUTPUT_DIR\""
