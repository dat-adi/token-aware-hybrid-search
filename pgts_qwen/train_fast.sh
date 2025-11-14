#!/bin/bash
# FAST TRAINING script for rapid experimentation
# Uses: Qwen 3B reasoning + Qwen 1.5B PRM
# Training time: 3.5-5.5 hours (vs 11-15 hours with larger models)

set -e

# Configuration
OUTPUT_DIR="outputs/training_fast_$(date +%Y%m%d_%H%M%S)"
MODEL_CONFIG="config/model_config_fast.yaml"
TRAIN_CONFIG="config/training_config_fast.yaml"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "FAST TRAINING MODE - Quick Experimentation"
echo "=============================================="
echo ""
echo "Models:"
echo "  - Reasoning: Qwen 3B (~6GB VRAM)"
echo "  - Reward: Qwen 1.5B (~3GB VRAM)"
echo "  - Policy: GPS (~2GB VRAM)"
echo ""
echo "Expected time: 3.5-5.5 hours"
echo "Expected accuracy: 60-75% on GSM8K"
echo ""
echo "Perfect for:"
echo "  - Method comparison"
echo "  - Rapid iteration"
echo "  - Ablation studies"
echo "  - Hyperparameter tuning"
echo "=============================================="
echo ""

# Log system info
echo "=== System Information ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Set memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Run training
echo "=== Starting Fast Training ==="
echo "Output: $OUTPUT_DIR"
echo ""

python train_optimized.py \
    --model_config "$MODEL_CONFIG" \
    --train_config "$TRAIN_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To scale up to larger models:"
echo "  ./train_2gpu.sh"
echo ""
echo "To evaluate:"
echo "  python main_eval.py \\"
echo "      --policy_checkpoint $OUTPUT_DIR/policy_final.pt \\"
echo "      --reward_model_path $OUTPUT_DIR/reward_model_final"
