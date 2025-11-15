#!/bin/bash
# Run all diagnostic scripts to identify training issues

set -e  # Exit on error

echo "=========================================================================="
echo "PGTS TRAINING DIAGNOSTICS - COMPREHENSIVE ANALYSIS"
echo "=========================================================================="
echo ""
echo "This script will run all diagnostic tests to identify why training"
echo "is showing 0% accuracy and very short trajectories."
echo ""

# Determine paths
REWARD_MODEL_PATH=""
POLICY_PATH=""
CONFIG_PATH=""

# Try to find reward model
if [ -d "outputs/reward_model_final" ]; then
    REWARD_MODEL_PATH="outputs/reward_model_final"
elif [ -d "outputs/training_2gpu/reward_model_final" ]; then
    REWARD_MODEL_PATH="outputs/training_2gpu/reward_model_final"
elif [ -d "outputs/training_fast/reward_model_final" ]; then
    REWARD_MODEL_PATH="outputs/training_fast/reward_model_final"
else
    echo "⚠️  WARNING: Could not find reward model in outputs/"
    echo "   Please specify path manually:"
    echo "   export REWARD_MODEL_PATH=/path/to/reward_model"
    echo ""
    if [ -z "$REWARD_MODEL_PATH" ]; then
        echo "Skipping reward model tests..."
    fi
fi

# Try to find policy checkpoint
if [ -f "outputs/training_2gpu/policy_checkpoint_latest.pt" ]; then
    POLICY_PATH="outputs/training_2gpu/policy_checkpoint_latest.pt"
elif [ -f "outputs/training_fast/policy_checkpoint_latest.pt" ]; then
    POLICY_PATH="outputs/training_fast/policy_checkpoint_latest.pt"
fi

# Try to find config
if [ -f "config/model_config_2gpu.yaml" ]; then
    CONFIG_PATH="config/model_config_2gpu.yaml"
elif [ -f "config/model_config_fast.yaml" ]; then
    CONFIG_PATH="config/model_config_fast.yaml"
fi

echo "Found resources:"
echo "  Reward model: ${REWARD_MODEL_PATH:-Not found}"
echo "  Policy checkpoint: ${POLICY_PATH:-Not found (will use random policy)}"
echo "  Config: ${CONFIG_PATH:-Not found (will use defaults)}"
echo ""

# ========================================================================
# TEST 1: Answer Extraction
# ========================================================================
echo "=========================================================================="
echo "TEST 1/3: ANSWER EXTRACTION"
echo "=========================================================================="
echo "Testing if the LLM generates answers in extractable format..."
echo ""

python diagnose_answer_extraction.py

echo ""
read -p "Press Enter to continue to next test..."
echo ""

# ========================================================================
# TEST 2: Reward Model Quality
# ========================================================================
if [ -n "$REWARD_MODEL_PATH" ] && [ -d "$REWARD_MODEL_PATH" ]; then
    echo "=========================================================================="
    echo "TEST 2/3: REWARD MODEL QUALITY"
    echo "=========================================================================="
    echo "Testing if reward model can distinguish correct from incorrect reasoning..."
    echo ""

    python diagnose_reward_model.py "$REWARD_MODEL_PATH"

    echo ""
    read -p "Press Enter to continue to next test..."
    echo ""
else
    echo "=========================================================================="
    echo "TEST 2/3: REWARD MODEL QUALITY - SKIPPED"
    echo "=========================================================================="
    echo "⚠️  Reward model not found. Skipping this test."
    echo ""
fi

# ========================================================================
# TEST 3: Trajectory Analysis
# ========================================================================
echo "=========================================================================="
echo "TEST 3/3: TRAJECTORY ANALYSIS"
echo "=========================================================================="
echo "Running full search to see what's happening during trajectory collection..."
echo ""

CMD="python diagnose_trajectory.py"

if [ -n "$REWARD_MODEL_PATH" ]; then
    CMD="$CMD --reward_model $REWARD_MODEL_PATH"
fi

if [ -n "$POLICY_PATH" ]; then
    CMD="$CMD --policy $POLICY_PATH"
fi

if [ -n "$CONFIG_PATH" ]; then
    CMD="$CMD --config $CONFIG_PATH"
fi

echo "Running: $CMD"
echo ""

$CMD

# ========================================================================
# SUMMARY
# ========================================================================
echo ""
echo "=========================================================================="
echo "DIAGNOSTIC SUMMARY"
echo "=========================================================================="
echo ""
echo "All diagnostics completed. Review the results above to identify:"
echo ""
echo "1. Answer Extraction Issues:"
echo "   - Does the LLM generate '####' or 'answer is' format?"
echo "   - What percentage of generations are extractable?"
echo ""
echo "2. Reward Model Issues:"
echo "   - Can it distinguish correct from incorrect reasoning?"
echo "   - Are rewards meaningful or close to 0.5 (random)?"
echo ""
echo "3. Trajectory Issues:"
echo "   - What actions is the policy taking?"
echo "   - How many reasoning steps are generated?"
echo "   - What rewards are assigned?"
echo ""
echo "Common issues and fixes:"
echo ""
echo "  Issue: 0% accuracy, short trajectories (2-5 steps)"
echo "  Likely cause: Answer extraction failing OR reward model untrained"
echo "  Fix: See recommendations in diagnostic outputs above"
echo ""
echo "  Issue: Reward model gives scores near 0.5"
echo "  Likely cause: Reward model not properly trained"
echo "  Fix: Re-run Phase 1 (reward model training)"
echo ""
echo "  Issue: LLM doesn't generate '####' format"
echo "  Likely cause: Prompt doesn't specify answer format"
echo "  Fix: Modify generation prompt in qwen3_wrapper.py"
echo ""
echo "=========================================================================="
