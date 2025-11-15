"""
Test script for GSM8K-Gen-Stepwise formatter.
Quick validation before running full dataset formatting.
"""
import logging
from data.gsm8k_stepwise_formatter import GSM8KStepwiseFormatter, format_gsm8k_stepwise

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_formatter():
    """Test formatter with small sample."""
    logger.info("=" * 60)
    logger.info("Testing GSM8K-Gen-Stepwise Formatter")
    logger.info("=" * 60)

    formatter = GSM8KStepwiseFormatter()

    # Test with just 10 examples to verify functionality
    logger.info("\n1. Loading small sample (10 examples)...")
    examples = formatter.load_and_format(
        dataset_name="ebony59/gsm8k-gen-stepwise",
        max_examples=10,
        balance_dataset=True
    )

    logger.info(f"\n✓ Generated {len(examples)} step-wise examples from 10 problems")

    # Show statistics
    correct = sum(1 for ex in examples if ex['label'] == 1)
    incorrect = sum(1 for ex in examples if ex['label'] == 0)

    logger.info(f"\nDataset statistics:")
    logger.info(f"  - Total examples: {len(examples)}")
    logger.info(f"  - Correct: {correct} ({100*correct/len(examples):.1f}%)")
    logger.info(f"  - Incorrect: {incorrect} ({100*incorrect/len(examples):.1f}%)")

    # Show a few examples
    logger.info("\n" + "=" * 60)
    logger.info("Sample Examples:")
    logger.info("=" * 60)

    for i, example in enumerate(examples[:3]):
        logger.info(f"\nExample {i+1}:")
        logger.info(f"  Problem: {example['problem'][:100]}...")
        logger.info(f"  Step ({example['step_index']+1}/{example['total_steps']}): {example['step'][:80]}...")
        logger.info(f"  Label: {'✓ CORRECT' if example['label'] == 1 else '✗ INCORRECT'}")
        logger.info(f"  Previous steps: {len(example['previous_steps'])}")

    # Test train/val split
    logger.info("\n" + "=" * 60)
    logger.info("Testing Train/Val Split")
    logger.info("=" * 60)

    train, val = formatter.create_train_val_split(examples, val_ratio=0.2)
    logger.info(f"Train: {len(train)} examples")
    logger.info(f"Val: {len(val)} examples")
    logger.info(f"Split ratio: {len(val)/len(examples):.2f}")

    logger.info("\n✓ All tests passed!")
    logger.info("\nYou can now run full dataset formatting with:")
    logger.info("  python data/gsm8k_stepwise_formatter.py --output_dir data/prm_formatted")

    return examples


if __name__ == "__main__":
    test_formatter()
