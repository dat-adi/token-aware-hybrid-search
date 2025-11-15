#!/bin/bash
# Simplified diagnostics that work around initialization issues

set -e

echo "=========================================================================="
echo "QUICK PGTS DIAGNOSTICS"
echo "=========================================================================="
echo ""

# Find reward model
REWARD_MODEL_PATH=""
if [ -d "outputs/training_2gpu_20251104_215652/reward_model_final" ]; then
    REWARD_MODEL_PATH="outputs/training_2gpu_20251104_215652/reward_model_final"
elif [ -d "outputs/reward_model_final" ]; then
    REWARD_MODEL_PATH="outputs/reward_model_final"
fi

echo "Found reward model: ${REWARD_MODEL_PATH:-NOT FOUND}"
echo ""

# Test 1: Answer extraction (simple version)
echo "=========================================================================="
echo "TEST 1: Answer Extraction Logic"
echo "=========================================================================="
python test_answer_extraction_simple.py
echo ""
read -p "Press Enter to continue..."
echo ""

# Test 2: Reward model quality
if [ -n "$REWARD_MODEL_PATH" ]; then
    echo "=========================================================================="
    echo "TEST 2: Reward Model Quality"
    echo "=========================================================================="
    python diagnose_reward_model.py "$REWARD_MODEL_PATH"
else
    echo "Skipping reward model test (not found)"
fi

echo ""
echo "=========================================================================="
echo "DIAGNOSIS COMPLETE"
echo "=========================================================================="
