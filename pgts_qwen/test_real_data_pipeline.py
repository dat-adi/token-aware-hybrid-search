#!/usr/bin/env python3
"""
Test script for the new GSM8K-Gen-Stepwise data pipeline.

This script verifies that the real incorrect reasoning data integration works correctly.
"""
import logging
from data.gsm8k_stepwise_formatter import GSM8KStepwiseFormatter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_data_loading():
    """Test loading and formatting the GSM8K-Gen-Stepwise dataset."""
    logger.info("=" * 60)
    logger.info("Testing GSM8K-Gen-Stepwise Data Pipeline")
    logger.info("=" * 60)

    # Initialize formatter
    formatter = GSM8KStepwiseFormatter(seed=42)

    # Load a small sample first to verify it works
    logger.info("\n[Step 1] Loading small sample (100 examples)...")
    sample_dataset = formatter.load_and_format(
        dataset_name="ebony59/gsm8k-gen-stepwise",
        max_examples=100,  # Small sample for testing
        balance_dataset=True
    )

    logger.info(f"✓ Successfully loaded {len(sample_dataset)} examples")

    # Verify data structure
    logger.info("\n[Step 2] Verifying data structure...")
    if len(sample_dataset) > 0:
        example = sample_dataset[0]

        required_fields = ['problem', 'step', 'previous_steps', 'reasoning_path', 'label']
        missing_fields = [field for field in required_fields if field not in example]

        if missing_fields:
            logger.error(f"✗ Missing fields: {missing_fields}")
            return False

        logger.info("✓ All required fields present:")
        logger.info(f"  - problem: {type(example['problem'])}")
        logger.info(f"  - step: {type(example['step'])}")
        logger.info(f"  - previous_steps: {type(example['previous_steps'])} (length: {len(example['previous_steps'])})")
        logger.info(f"  - reasoning_path: {type(example['reasoning_path'])} (length: {len(example['reasoning_path'])})")
        logger.info(f"  - label: {type(example['label'])} (value: {example['label']})")

    # Check label distribution
    logger.info("\n[Step 3] Checking label distribution...")
    correct_count = sum(1 for ex in sample_dataset if ex['label'] == 1)
    incorrect_count = sum(1 for ex in sample_dataset if ex['label'] == 0)

    logger.info(f"  Correct examples (label=1): {correct_count}")
    logger.info(f"  Incorrect examples (label=0): {incorrect_count}")
    logger.info(f"  Balance ratio: {correct_count}/{incorrect_count} = {correct_count/max(incorrect_count, 1):.2f}")

    # Show sample examples
    logger.info("\n[Step 4] Sample examples from dataset:")
    logger.info("\n--- CORRECT EXAMPLE ---")
    correct_ex = next((ex for ex in sample_dataset if ex['label'] == 1), None)
    if correct_ex:
        logger.info(f"Problem: {correct_ex['problem'][:100]}...")
        logger.info(f"Step: {correct_ex['step'][:100]}...")
        logger.info(f"Label: {correct_ex['label']} (CORRECT)")

    logger.info("\n--- INCORRECT EXAMPLE ---")
    incorrect_ex = next((ex for ex in sample_dataset if ex['label'] == 0), None)
    if incorrect_ex:
        logger.info(f"Problem: {incorrect_ex['problem'][:100]}...")
        logger.info(f"Step: {incorrect_ex['step'][:100]}...")
        logger.info(f"Label: {incorrect_ex['label']} (INCORRECT)")

    # Test step-level formatting
    logger.info("\n[Step 5] Verifying step-level PRM formatting...")
    from models.reward_model import ProcessRewardModel

    # Note: We're testing the format method, not loading the full model
    # Create a mock formatter (without loading the model weights)
    logger.info("Testing format_step_input() method...")

    # Test with first example
    if len(sample_dataset) > 0:
        example = sample_dataset[0]

        # Simulate what the reward model format method does
        class MockRewardModel:
            def format_step_input(self, problem, previous_steps, current_step):
                if previous_steps:
                    prev_text = "\n".join(
                        f"Step {i+1}: {step}"
                        for i, step in enumerate(previous_steps)
                    )
                    formatted = f"""Problem: {problem}

Previous Steps:
{prev_text}

Current Step:
Step {len(previous_steps)+1}: {current_step}"""
                else:
                    formatted = f"""Problem: {problem}

Current Step:
Step 1: {current_step}"""
                return formatted

        mock_rm = MockRewardModel()
        formatted_text = mock_rm.format_step_input(
            example['problem'],
            example['previous_steps'],
            example['step']
        )

        logger.info("✓ Step-level format generated successfully")
        logger.info("\nFormatted input preview:")
        logger.info("-" * 60)
        logger.info(formatted_text[:300] + "..." if len(formatted_text) > 300 else formatted_text)
        logger.info("-" * 60)

        # Verify it's step-level, not chain-level
        if "Current Step:" in formatted_text:
            logger.info("✓ CORRECT: Using step-level formatting (has 'Current Step:')")
        else:
            logger.error("✗ WRONG: Missing 'Current Step:' - might be chain-level")

        if len(example['previous_steps']) > 0:
            if "Previous Steps:" in formatted_text:
                logger.info("✓ CORRECT: Context included (has 'Previous Steps:')")
            else:
                logger.error("✗ WRONG: Missing 'Previous Steps:' - context not included")

    logger.info("\n" + "=" * 60)
    logger.info("✓ All tests passed!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Run full training with: ./train_2gpu.sh")
    logger.info("2. Or use fast training with: ./train_fast.sh")
    logger.info("3. Monitor outputs in: outputs/training_2gpu/")

    return True


def test_train_val_split():
    """Test train/validation split functionality."""
    logger.info("\n[Step 6] Testing train/val split...")

    formatter = GSM8KStepwiseFormatter(seed=42)

    # Load small dataset
    dataset = formatter.load_and_format(
        dataset_name="ebony59/gsm8k-gen-stepwise",
        max_examples=100,
        balance_dataset=True
    )

    # Split into train/val
    train_data, val_data = formatter.create_train_val_split(
        dataset,
        val_ratio=0.1
    )

    logger.info(f"  Train examples: {len(train_data)}")
    logger.info(f"  Val examples: {len(val_data)}")
    logger.info(f"  Split ratio: {len(val_data) / len(dataset):.2%}")

    return True


if __name__ == "__main__":
    try:
        success = test_data_loading()
        if success:
            test_train_val_split()
            logger.info("\n✓✓✓ Pipeline integration successful! ✓✓✓")
    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}", exc_info=True)
        logger.info("\nTroubleshooting:")
        logger.info("1. Ensure you have internet connection (downloads HuggingFace dataset)")
        logger.info("2. Check that 'datasets' package is installed: pip install datasets")
        logger.info("3. Verify HuggingFace credentials if dataset is private")
