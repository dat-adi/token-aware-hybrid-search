"""
Test the quality of base reasoning model on GSM8K.
Diagnose why it's generating poor reasoning.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.qwen3_wrapper import Qwen3ReasoningGenerator
from data.gsm8k_loader import load_gsm8k
import torch

def test_reasoning_quality():
    """Test base model reasoning on GSM8K examples."""

    print("="*80)
    print("Testing Base Reasoning Model Quality")
    print("="*80)

    # Load a few GSM8K examples
    train_data, _, _ = load_gsm8k()
    test_problems = train_data[:5]

    # Initialize reasoning generator
    print("\nLoading Qwen3 model...")
    gen = Qwen3ReasoningGenerator(
        model_name="Qwen/Qwen2.5-Math-1.5B",  # Math-tuned version
        device="cuda",
        max_new_tokens=300,
        temperature=0.7
    )

    print("\n" + "="*80)
    print("Testing on 5 GSM8K problems")
    print("="*80)

    for idx, example in enumerate(test_problems):
        problem = example['problem']
        ground_truth = example['answer']

        print(f"\n{'='*80}")
        print(f"Problem {idx + 1}")
        print(f"{'='*80}")
        print(f"Question: {problem[:150]}...")
        print(f"Ground Truth: {ground_truth}")

        # Generate first step
        print("\nGenerating Step 1...")
        step1, _ = gen.generate_step(problem, [])
        print(f"Step 1: {step1[:200]}...")

        # Generate second step
        print("\nGenerating Step 2...")
        step2, _ = gen.generate_step(problem, [step1])
        print(f"Step 2: {step2[:200]}...")

        # Generate third step
        print("\nGenerating Step 3...")
        step3, _ = gen.generate_step(problem, [step1, step2])
        print(f"Step 3: {step3[:200]}...")

        # Check if reasoning makes sense
        print("\n📊 Quality Check:")

        # Does it contain numbers?
        has_numbers = any(char.isdigit() for char in step1 + step2 + step3)
        print(f"  ✓ Contains numbers: {has_numbers}")

        # Does it contain arithmetic operations?
        has_ops = any(op in step1 + step2 + step3 for op in ['+', '-', '*', '/', '='])
        print(f"  ✓ Contains operations: {has_ops}")

        # Length check
        total_len = len(step1) + len(step2) + len(step3)
        print(f"  ✓ Total reasoning length: {total_len} chars")

        if total_len < 50:
            print("  ⚠️  WARNING: Reasoning is very short!")

        if not has_numbers:
            print("  ⚠️  WARNING: No numbers in reasoning!")

        if not has_ops:
            print("  ⚠️  WARNING: No arithmetic operations!")

        print()

if __name__ == "__main__":
    test_reasoning_quality()
