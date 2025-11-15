#!/usr/bin/env python3
"""
Test answer extraction to see if it's working correctly.

This checks:
1. Does the LLM generate answers in the expected format?
2. Is the extraction regex working?
3. What percentage of generations are extractable?
"""

import torch
import sys
from models.qwen3_wrapper import Qwen3ReasoningGenerator
from tree_search.search_algorithm import PGTSSearch, SearchConfig


def test_answer_extraction():
    """Test answer extraction from generated text."""

    print("=" * 80)
    print("ANSWER EXTRACTION DIAGNOSTIC")
    print("=" * 80)

    # Test extraction function
    search = PGTSSearch(None, None, None, SearchConfig())

    # Test cases
    test_cases = [
        ("The answer is 42", "42 or answer extraction from 'answer is'"),
        ("Therefore, the final answer is 18", "18"),
        ("So the result is 18 dollars", "Likely fails - no marker"),
        ("The answer is 18 dollars per day", "18"),
        ("#### 18", "18 (GSM8K format)"),
        ("After solving, we get #### 345", "345"),
        ("Let me calculate... the answer is $15.50", "Likely $15.50 or just 15"),
        ("The solution gives us 42.", "Likely fails"),
        ("18", "Likely fails - no marker"),
    ]

    print("\n1. Testing extraction function on sample texts:")
    print("=" * 80)

    for text, expected in test_cases:
        extracted = search.extract_answer(text)
        status = "✓" if extracted is not None else "✗"
        print(f"\n{status} Input: \"{text}\"")
        print(f"  Extracted: {extracted}")
        print(f"  Expected: {expected}")

    # Test with actual generation
    print("\n\n2. Testing with actual LLM generation:")
    print("=" * 80)

    print("\nLoading Qwen3 model...")
    generator = Qwen3ReasoningGenerator(
        model_name="Qwen/Qwen2.5-3B",
        device="cuda:0",
        torch_dtype=torch.bfloat16
    )
    print("✓ Model loaded")

    # Generate answers for test problems
    test_problems = [
        ("What is 15 * 23?", "345"),
        ("If a shirt costs $25 and is on sale for 20% off, what is the sale price?", "20"),
        ("A train travels 120 miles in 2 hours. What is its average speed in miles per hour?", "60"),
    ]

    print("\n3. Generating answers and testing extraction:")
    print("=" * 80)

    extraction_success = 0
    total_tests = len(test_problems)

    for problem, correct_answer in test_problems:
        print(f"\n{'='*80}")
        print(f"Problem: {problem}")
        print(f"Expected answer: {correct_answer}")
        print("-" * 80)

        # Generate reasoning
        reasoning_chain = []
        max_steps = 5

        for step_num in range(max_steps):
            try:
                generated_text, _ = generator.generate_step(
                    problem,
                    reasoning_chain,
                    return_hidden_states=True,
                    max_length=150
                )

                reasoning_chain.append(generated_text)
                print(f"\nStep {step_num + 1}: {generated_text}")

                # Try to extract answer
                extracted = search.extract_answer(generated_text)
                if extracted is not None:
                    print(f"\n✓ Answer extracted: {extracted}")
                    extraction_success += 1
                    break

            except Exception as e:
                print(f"\n✗ Generation failed: {e}")
                break

        # Check final step
        if len(reasoning_chain) > 0:
            final_step = reasoning_chain[-1]
            extracted_final = search.extract_answer(final_step)

            if extracted_final is None:
                print(f"\n✗ Final step did NOT contain extractable answer:")
                print(f"   Last step: {final_step}")
                print(f"\n   ISSUE: The model doesn't naturally generate '####' or 'answer is'")
                print(f"          You need to either:")
                print(f"          1. Modify the prompt to force answer format")
                print(f"          2. Improve answer extraction regex")
                print(f"          3. Add a final 'answer generation' step")
            else:
                print(f"\n✓ Successfully extracted: {extracted_final}")
        else:
            print("\n✗ No reasoning generated")

    # Report results
    print("\n\n" + "=" * 80)
    print("4. RESULTS")
    print("=" * 80)

    success_rate = 100 * extraction_success / total_tests
    print(f"\nExtraction success rate: {extraction_success}/{total_tests} ({success_rate:.1f}%)")

    if extraction_success == 0:
        print("\n✗✗✗ CRITICAL ISSUE ✗✗✗")
        print("\nAnswer extraction is FAILING completely!")
        print("This explains why training accuracy is 0%.")
        print("\nRECOMMENDED FIXES:")
        print("  1. Modify prompt to force answer format (easiest)")
        print("  2. Add final answer generation step before TERMINATE")
        print("  3. Use better extraction (regex for any number)")
        print("\nExample prompt modification:")
        print("  'Solve step by step. End with: The answer is #### <number>'")

    elif extraction_success < total_tests:
        print("\n⚠️  PARTIAL SUCCESS")
        print(f"\nAnswer extraction works {success_rate:.1f}% of the time.")
        print("This may contribute to low training accuracy.")
        print("\nConsider improving extraction or prompting.")

    else:
        print("\n✓✓✓ EXTRACTION WORKING ✓✓✓")
        print("\nAnswer extraction appears to work correctly.")
        print("If training accuracy is still 0%, the issue is elsewhere.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_answer_extraction()
