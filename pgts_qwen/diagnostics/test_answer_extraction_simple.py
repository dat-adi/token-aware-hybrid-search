#!/usr/bin/env python3
"""
Simple answer extraction test without dependencies.
"""

import re
from typing import Optional


def extract_answer(text: str) -> Optional[str]:
    """
    Extract final answer from reasoning step.
    (Copy of search_algorithm.py extract_answer method)
    """
    # Look for answer markers
    if "####" in text:
        answer = text.split("####")[-1].strip()
        return answer
    elif "answer is" in text.lower():
        parts = text.lower().split("answer is")
        if len(parts) > 1:
            answer = parts[-1].strip().split()[0]
            return answer

    return None


def test_extraction():
    """Test extraction on sample cases."""

    print("=" * 80)
    print("ANSWER EXTRACTION TEST")
    print("=" * 80)

    test_cases = [
        ("The answer is 42", "42"),
        ("Therefore, the final answer is 18", "18"),
        ("So the result is 18 dollars", None),
        ("The answer is 18 dollars per day", "18"),
        ("#### 18", "18"),
        ("After solving, we get #### 345", "345"),
        ("Let me calculate... the answer is $15.50", "$15.50 or 15"),
        ("The solution gives us 42.", None),
        ("18", None),
    ]

    print("\n1. Testing extraction function on sample texts:")
    print("=" * 80)

    success = 0
    for text, expected in test_cases:
        extracted = extract_answer(text)
        if extracted is not None:
            success += 1
            status = "✓"
        else:
            status = "✗"

        print(f"\n{status} Input: \"{text}\"")
        print(f"  Extracted: {extracted}")
        print(f"  Expected: {expected}")

    print("\n" + "=" * 80)
    print(f"Extraction success: {success}/{len(test_cases)} ({100*success/len(test_cases):.1f}%)")

    if success < len(test_cases) / 2:
        print("\n⚠️  WARNING: Extraction working on less than 50% of test cases")
        print("This suggests the extraction logic is too strict")

    return success > 0


if __name__ == "__main__":
    test_extraction()
