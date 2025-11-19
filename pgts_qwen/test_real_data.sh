#!/bin/bash
#
# Test the new GSM8K-Gen-Stepwise data pipeline
#
# This script verifies the integration of real incorrect reasoning data
# before running full training.
#

set -e  # Exit on error

echo "=========================================="
echo "GSM8K-Gen-Stepwise Data Pipeline Test"
echo "=========================================="
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "feature/real-incorrect-reasoning-data" ]; then
    echo "⚠️  Warning: You're on branch '$CURRENT_BRANCH'"
    echo "   This feature is on branch 'feature/real-incorrect-reasoning-data'"
    echo ""
fi

# Run the test
echo "Running data pipeline test..."
echo ""

python test_real_data_pipeline.py

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
