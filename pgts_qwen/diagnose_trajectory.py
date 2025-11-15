#!/usr/bin/env python3
"""
Diagnostic script to analyze what's happening during trajectory collection.

This will check:
1. What actions is the policy taking?
2. What reasoning steps are being generated?
3. Is answer extraction working?
4. What rewards are being assigned?
5. Why are trajectories so short?
"""

import torch
import sys
import yaml
from pathlib import Path

from models.policy_network import GPSPolicyNetwork
from models.qwen3_wrapper import Qwen3ReasoningGenerator
from models.reward_model import ProcessRewardModel
from tree_search.search_algorithm import PGTSSearch, SearchConfig


def diagnose_trajectory(policy_path=None, reward_model_path=None, config_path=None):
    """Run a single search and diagnose what's happening."""

    print("=" * 80)
    print("TRAJECTORY DIAGNOSTIC")
    print("=" * 80)

    # Load configuration
    if config_path and Path(config_path).exists():
        print(f"\nLoading config from: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_config = config.get('model', {})
        qwen_model = model_config.get('qwen3', {}).get('model_name', 'Qwen/Qwen2.5-3B')
    else:
        qwen_model = 'Qwen/Qwen2.5-3B'
        print(f"\nUsing default model: {qwen_model}")

    # Load models
    print("\n1. Loading models...")

    print(f"   - Loading reasoning generator: {qwen_model}")
    generator = Qwen3ReasoningGenerator(
        model_name=qwen_model,
        device="cuda:0",
        torch_dtype=torch.bfloat16
    )
    print("   ✓ Generator loaded")

    if reward_model_path:
        print(f"   - Loading reward model: {reward_model_path}")
        reward_model = ProcessRewardModel.from_pretrained(
            reward_model_path,
            device="cuda:1"
        )
        print("   ✓ Reward model loaded")
    else:
        print("   ✗ No reward model path provided - skipping")
        reward_model = None

    if policy_path:
        print(f"   - Loading policy network: {policy_path}")
        policy = GPSPolicyNetwork()
        checkpoint = torch.load(policy_path, map_location="cuda:1")
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy = policy.to("cuda:1")
        policy.eval()
        print("   ✓ Policy loaded")
    else:
        print("   - Using random policy (no checkpoint provided)")
        policy = GPSPolicyNetwork().to("cuda:1")

    # Create search algorithm
    search_config = SearchConfig(
        device="cuda:1",
        use_policy=True,
        temperature=1.0,
        max_nodes=50,
        max_depth=20
    )

    search = PGTSSearch(
        reasoning_generator=generator,
        reward_model=reward_model,
        policy_network=policy,
        config=search_config
    )

    # Test problem (GSM8K example)
    problem = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    ground_truth = "18"

    print("\n" + "=" * 80)
    print("2. Running search on test problem")
    print("=" * 80)
    print(f"\nProblem: {problem}")
    print(f"Ground truth: ${ground_truth}")

    # Run search
    print("\n3. Search execution:")
    print("-" * 80)

    tree, trajectory = search.search(problem, collect_trajectory=True)

    print("\n4. Search completed")
    print("-" * 80)

    # Analyze trajectory
    print(f"\nTrajectory length: {len(trajectory.states)} steps")
    print(f"Final answer: {trajectory.final_answer}")

    # Evaluate correctness
    is_correct = search.evaluate_solution(trajectory, ground_truth)
    print(f"Correct: {is_correct}")

    # Print detailed trajectory
    print("\n5. Detailed trajectory breakdown:")
    print("=" * 80)

    action_names = ["EXPAND", "BRANCH", "BACKTRACK", "TERMINATE"]

    for i, (action, reward) in enumerate(zip(trajectory.actions, trajectory.rewards)):
        action_name = action_names[action]
        print(f"\nStep {i+1}:")
        print(f"  Action: {action_name}")
        print(f"  Reward: {reward:.4f}")

        if i < len(trajectory.log_probs):
            log_prob = trajectory.log_probs[i]
            if isinstance(log_prob, torch.Tensor):
                log_prob = log_prob.item()
            print(f"  Log prob: {log_prob:.4f}")

        if i < len(trajectory.values):
            value = trajectory.values[i]
            if isinstance(value, torch.Tensor):
                value = value.item()
            print(f"  Value estimate: {value:.4f}")

    # Print reasoning steps
    print("\n6. Generated reasoning steps:")
    print("=" * 80)

    path = tree.get_best_leaf().get_path_from_root()[1:]  # Exclude root

    if len(path) == 0:
        print("✗ NO REASONING STEPS GENERATED!")
    else:
        for i, node in enumerate(path):
            print(f"\nStep {i+1} (depth={node.depth}, reward={node.reward:.4f}):")
            print(f"  {node.content}")

    # Answer extraction test
    print("\n7. Answer extraction test:")
    print("=" * 80)

    if len(path) > 0:
        last_step = path[-1].content
        print(f"\nLast step text: {last_step}")
        extracted = search.extract_answer(last_step)
        print(f"Extracted answer: {extracted}")

        if extracted is None:
            print("✗ FAILED to extract answer from last step!")
            print("  This is why accuracy is 0%")
            print("\n  The last step needs to contain:")
            print("    - '####' followed by the answer (GSM8K format), OR")
            print("    - 'answer is' followed by the answer")
    else:
        print("✗ No reasoning steps generated - cannot extract answer")

    # Diagnose issues
    print("\n8. DIAGNOSTIC RESULTS:")
    print("=" * 80)

    issues = []

    if len(trajectory.states) < 5:
        issues.append(f"Trajectory too short ({len(trajectory.states)} steps) - policy may be learning to terminate early")

    if trajectory.final_answer is None:
        issues.append("Answer extraction failed - model doesn't generate answer in expected format")

    if not is_correct and trajectory.final_answer is not None:
        issues.append(f"Answer extracted ({trajectory.final_answer}) but incorrect (expected {ground_truth})")

    if len(path) == 0:
        issues.append("No reasoning steps generated - generator may be failing")

    # Check reward quality
    if len(trajectory.rewards) > 0:
        avg_reward = sum(trajectory.rewards) / len(trajectory.rewards)
        if abs(avg_reward - 0.5) < 0.1:
            issues.append(f"Average reward {avg_reward:.4f} is close to 0.5 - reward model may be untrained")

    # Check action distribution
    if len(trajectory.actions) > 0:
        action_counts = [trajectory.actions.count(i) for i in range(4)]
        print(f"\nAction distribution:")
        for i, count in enumerate(action_counts):
            pct = 100 * count / len(trajectory.actions)
            print(f"  {action_names[i]}: {count} ({pct:.1f}%)")

        terminate_count = action_counts[3]
        if len(trajectory.actions) > 0 and trajectory.actions[-1] == 3:
            print("  (Terminated normally)")
        elif len(trajectory.actions) > 0:
            issues.append(f"Did not terminate - hit max_nodes limit")

    if len(issues) == 0:
        print("\n✓ No obvious issues detected")
        print("  The search appears to be working correctly.")
        print("  If training accuracy is still 0%, the issue may be:")
        print("    - Reasoning generator not capable enough")
        print("    - Problems too difficult")
        print("    - Need more exploration/training iterations")
    else:
        print("\n✗ Issues detected:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose trajectory collection")
    parser.add_argument("--policy", type=str, help="Path to policy checkpoint")
    parser.add_argument("--reward_model", type=str, help="Path to reward model")
    parser.add_argument("--config", type=str, help="Path to model config YAML")

    args = parser.parse_args()

    print("\nUsage examples:")
    print("  python diagnose_trajectory.py --reward_model outputs/reward_model_final")
    print("  python diagnose_trajectory.py --policy outputs/policy_iter_10.pt --reward_model outputs/reward_model_final")
    print()

    diagnose_trajectory(
        policy_path=args.policy,
        reward_model_path=args.reward_model,
        config_path=args.config
    )
