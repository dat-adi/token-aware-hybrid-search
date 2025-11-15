#!/usr/bin/env python3
"""
Diagnostic script to test if the reward model is functioning correctly.

This will check:
1. Can the reward model distinguish correct vs incorrect reasoning?
2. Are rewards meaningful or random noise?
3. Is the model loaded correctly?
"""

import torch
import sys
from models.reward_model import ProcessRewardModel

def test_reward_model(model_path: str):
    """Test reward model quality."""

    print("=" * 80)
    print("REWARD MODEL DIAGNOSTIC")
    print("=" * 80)

    # Load model
    print(f"\n1. Loading reward model from: {model_path}")
    try:
        prm = ProcessRewardModel.from_pretrained(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ FAILED to load model: {e}")
        return

    # Test problem
    problem = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    # Correct reasoning
    correct_reasoning = [
        "Janet's ducks lay 16 eggs per day.",
        "She eats 3 eggs for breakfast.",
        "She uses 4 eggs for baking muffins.",
        "So she uses 3 + 4 = 7 eggs total.",
        "That means she has 16 - 7 = 9 eggs left to sell.",
        "She sells them for $2 each.",
        "So she makes 9 * 2 = 18 dollars per day."
    ]

    # Incorrect reasoning (wrong math)
    incorrect_reasoning = [
        "Janet's ducks lay 16 eggs per day.",
        "She eats 3 eggs for breakfast.",
        "She uses 4 eggs for baking muffins.",
        "So she uses 3 + 4 = 8 eggs total.",  # WRONG: 3+4=8
        "That means she has 16 - 8 = 8 eggs left to sell.",  # Wrong
        "She sells them for $2 each.",
        "So she makes 8 * 2 = 16 dollars per day."  # Wrong answer
    ]

    # Nonsense reasoning
    nonsense_reasoning = [
        "Let's think about quantum physics first.",
        "The duck is a bird that can fly.",
        "Muffins are delicious breakfast food.",
        "Therefore the answer must be 42."
    ]

    print("\n2. Testing reward model discrimination")
    print("-" * 80)

    # Compute rewards
    print("\nCORRECT REASONING:")
    correct_rewards = []
    for i, step in enumerate(correct_reasoning):
        reward = prm.compute_step_reward(problem, correct_reasoning[:i+1], i)
        correct_rewards.append(reward)
        print(f"  Step {i+1}: {reward:.4f} - {step[:60]}...")

    print("\nINCORRECT REASONING (wrong math):")
    incorrect_rewards = []
    for i, step in enumerate(incorrect_reasoning):
        reward = prm.compute_step_reward(problem, incorrect_reasoning[:i+1], i)
        incorrect_rewards.append(reward)
        print(f"  Step {i+1}: {reward:.4f} - {step[:60]}...")

    print("\nNONSENSE REASONING:")
    nonsense_rewards = []
    for i, step in enumerate(nonsense_reasoning):
        reward = prm.compute_step_reward(problem, nonsense_reasoning[:i+1], i)
        nonsense_rewards.append(reward)
        print(f"  Step {i+1}: {reward:.4f} - {step[:60]}...")

    print("\n" + "=" * 80)
    print("3. ANALYSIS")
    print("=" * 80)

    # Calculate statistics
    correct_mean = sum(correct_rewards) / len(correct_rewards)
    incorrect_mean = sum(incorrect_rewards) / len(incorrect_rewards)
    nonsense_mean = sum(nonsense_rewards) / len(nonsense_rewards)

    correct_final = correct_rewards[-1]
    incorrect_final = incorrect_rewards[-1]
    nonsense_final = nonsense_rewards[-1]

    print(f"\nMean Rewards:")
    print(f"  Correct:   {correct_mean:.4f}")
    print(f"  Incorrect: {incorrect_mean:.4f}")
    print(f"  Nonsense:  {nonsense_mean:.4f}")

    print(f"\nFinal Step Rewards:")
    print(f"  Correct:   {correct_final:.4f}")
    print(f"  Incorrect: {incorrect_final:.4f}")
    print(f"  Nonsense:  {nonsense_final:.4f}")

    # Diagnostics
    print("\n" + "=" * 80)
    print("4. DIAGNOSTIC RESULTS")
    print("=" * 80)

    all_rewards = correct_rewards + incorrect_rewards + nonsense_rewards
    reward_range = max(all_rewards) - min(all_rewards)

    print(f"\nReward range: {min(all_rewards):.4f} to {max(all_rewards):.4f} (span: {reward_range:.4f})")

    # Check if model is discriminating
    issues = []

    # Test 1: Is correct better than incorrect?
    if correct_mean > incorrect_mean:
        print("✓ PASS: Correct reasoning scores higher than incorrect")
    else:
        print("✗ FAIL: Correct reasoning does NOT score higher than incorrect")
        issues.append("Model cannot distinguish correct from incorrect reasoning")

    # Test 2: Is nonsense scored low?
    if nonsense_mean < correct_mean:
        print("✓ PASS: Nonsense reasoning scores lower than correct")
    else:
        print("✗ FAIL: Nonsense reasoning scores as high as correct")
        issues.append("Model cannot detect nonsense reasoning")

    # Test 3: Is there meaningful variance?
    if reward_range > 0.1:
        print(f"✓ PASS: Rewards show meaningful variance (range: {reward_range:.4f})")
    else:
        print(f"✗ FAIL: Rewards are too similar (range: {reward_range:.4f})")
        issues.append("Rewards have very low variance - model may not be trained")

    # Test 4: Are all rewards near 0.5 (random)?
    mean_all = sum(all_rewards) / len(all_rewards)
    if abs(mean_all - 0.5) > 0.1:
        print(f"✓ PASS: Mean reward {mean_all:.4f} is away from 0.5 (not random)")
    else:
        print(f"✗ FAIL: Mean reward {mean_all:.4f} is close to 0.5 (may be random/untrained)")
        issues.append("Rewards cluster around 0.5 - suggests untrained model")

    # Test 5: Does correct final answer score high?
    if correct_final > 0.7:
        print(f"✓ PASS: Correct final answer scores high ({correct_final:.4f})")
    else:
        print(f"✗ FAIL: Correct final answer scores low ({correct_final:.4f})")
        issues.append("Correct solutions not recognized with high confidence")

    print("\n" + "=" * 80)
    print("5. CONCLUSION")
    print("=" * 80)

    if len(issues) == 0:
        print("\n✓✓✓ REWARD MODEL APPEARS TO BE WORKING CORRECTLY ✓✓✓")
        print("\nThe reward model can distinguish correct from incorrect reasoning.")
        print("It should provide useful training signal for the policy network.")
    else:
        print("\n✗✗✗ REWARD MODEL HAS ISSUES ✗✗✗")
        print("\nProblems detected:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n⚠️  RECOMMENDATION: The reward model may not have been properly trained.")
        print("    This would cause the policy training to fail with 0% accuracy.")
        print("    Consider re-training the reward model (Phase 1) before policy training.")

    print("\n" + "=" * 80)

    return len(issues) == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_reward_model.py <path_to_reward_model>")
        print("\nExample:")
        print("  python diagnose_reward_model.py outputs/reward_model_final")
        print("  python diagnose_reward_model.py outputs/training_2gpu/reward_model_final")
        sys.exit(1)

    model_path = sys.argv[1]
    success = test_reward_model(model_path)
    sys.exit(0 if success else 1)
