"""
Test PRM generalization from MATH to GSM8K.
Check if 85% MATH accuracy translates to GSM8K.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.reward_model import ProcessRewardModel
from data.gsm8k_loader import load_gsm8k
import torch

def test_prm_generalization():
    """Test PRM trained on MATH dataset on GSM8K examples."""

    print("="*80)
    print("Testing PRM Generalization: MATH → GSM8K")
    print("="*80)

    # Load PRM
    prm_path = "outputs/training_2gpu_20251104_215652/reward_model_final"
    print(f"\nLoading PRM from: {prm_path}")

    try:
        prm = ProcessRewardModel.from_pretrained(prm_path, device="cuda")
        print("✓ PRM loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load PRM: {e}")
        return

    # Load GSM8K examples
    train_data, _, _ = load_gsm8k()

    # Test on examples where we know the answer
    test_cases = [
        {
            'problem': train_data[0]['problem'],
            'correct_path': train_data[0]['steps'][:3] if 'steps' in train_data[0] else [
                "Calculate the total number of students: 5 + 8 + 4 = 17 students",
                "Calculate apples needed: 17 students × 2 apples = 34 apples",
                "Therefore, the teacher needs 34 apples total."
            ],
            'incorrect_path': [
                "Calculate the total number of students: 5 + 8 + 4 = 16 students",  # Wrong
                "Calculate apples needed: 16 students × 2 apples = 32 apples",
                "Therefore, the teacher needs 32 apples total."
            ]
        }
    ]

    # Manual test cases
    manual_test = {
        'problem': "If John has 5 apples and buys 3 more, how many does he have?",
        'correct_path': [
            "Start with 5 apples",
            "Add 3 more apples: 5 + 3 = 8",
            "Answer: 8 apples"
        ],
        'incorrect_path': [
            "Start with 5 apples",
            "Add 3 more apples: 5 + 3 = 7",  # Wrong calculation
            "Answer: 7 apples"
        ]
    }
    test_cases.append(manual_test)

    print("\n" + "="*80)
    print("Testing PRM on Correct vs Incorrect Reasoning")
    print("="*80)

    for idx, test in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"Test Case {idx + 1}")
        print(f"{'='*80}")
        print(f"Problem: {test['problem'][:100]}...")

        # Evaluate correct path
        print("\n📗 CORRECT Path:")
        for i, step in enumerate(test['correct_path']):
            print(f"  Step {i+1}: {step[:80]}...")

        correct_rewards = prm.compute_step_rewards(test['problem'], test['correct_path'])
        avg_correct = sum(correct_rewards) / len(correct_rewards)
        print(f"\n  PRM Rewards: {[f'{r:.3f}' for r in correct_rewards]}")
        print(f"  Average: {avg_correct:.3f}")
        prm_says_correct = avg_correct > 0.5
        print(f"  PRM Prediction: {'✓ Correct' if prm_says_correct else '✗ Incorrect'}")

        # Evaluate incorrect path
        print("\n📕 INCORRECT Path:")
        for i, step in enumerate(test['incorrect_path']):
            print(f"  Step {i+1}: {step[:80]}...")

        incorrect_rewards = prm.compute_step_rewards(test['problem'], test['incorrect_path'])
        avg_incorrect = sum(incorrect_rewards) / len(incorrect_rewards)
        print(f"\n  PRM Rewards: {[f'{r:.3f}' for r in incorrect_rewards]}")
        print(f"  Average: {avg_incorrect:.3f}")
        prm_says_incorrect = avg_incorrect <= 0.5
        print(f"  PRM Prediction: {'✓ Incorrect' if prm_says_incorrect else '✗ Correct (WRONG!)'}")

        # Check discrimination
        print(f"\n📊 Discrimination Analysis:")
        print(f"  Correct avg:   {avg_correct:.3f}")
        print(f"  Incorrect avg: {avg_incorrect:.3f}")
        print(f"  Gap:           {avg_correct - avg_incorrect:.3f}")

        if avg_correct > avg_incorrect:
            print(f"  ✓ PRM correctly ranks correct > incorrect")
        else:
            print(f"  ✗ PRM FAILS: ranks incorrect ≥ correct!")

        if abs(avg_correct - avg_incorrect) < 0.1:
            print(f"  ⚠️  WARNING: Gap is very small ({abs(avg_correct - avg_incorrect):.3f})")
            print(f"     PRM cannot reliably distinguish correct from incorrect!")

        print()

    # Summary
    print("="*80)
    print("Summary")
    print("="*80)
    print("\nIf you see:")
    print("  • High rewards for both correct AND incorrect → PRM not generalizing")
    print("  • Small gap (< 0.1) → PRM cannot discriminate")
    print("  • Incorrect ranked higher → PRM completely broken for GSM8K")
    print("\nPossible solutions:")
    print("  1. Fine-tune PRM on GSM8K examples")
    print("  2. Use outcome-only rewards (ignore PRM)")
    print("  3. Collect PRM800K-style annotations for GSM8K")
    print("  4. Use a different PRM trained on GSM8K")

if __name__ == "__main__":
    test_prm_generalization()
