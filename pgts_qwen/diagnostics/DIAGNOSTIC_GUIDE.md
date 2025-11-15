# Training Diagnostic Guide

## Problem: 0% Accuracy, Short Trajectories (2-7 steps)

You're experiencing poor training performance with these symptoms:
- **Accuracy: 0.0%** across all iterations
- **Very short trajectories**: 2-7 steps (should be 8-15 for GSM8K)
- **High entropy**: ~1.26-1.30 (not decreasing)
- **Unstable value loss**: 0.5-4.8 (wild fluctuations)

## Quick Diagnosis

Run the comprehensive diagnostic suite:

```bash
cd /home/datadi/Code/token-aware-hybrid-search/pgts_qwen
chmod +x run_all_diagnostics.sh
./run_all_diagnostics.sh
```

This will run 3 tests:
1. **Answer Extraction Test** - Does the LLM generate extractable answers?
2. **Reward Model Quality Test** - Is the PRM giving meaningful signals?
3. **Trajectory Analysis** - What happens during a search?

## Individual Diagnostics

### 1. Test Reward Model Quality

**Most likely root cause** - Run this first!

```bash
python diagnose_reward_model.py outputs/reward_model_final
```

**What it tests:**
- Can the PRM distinguish correct from incorrect reasoning?
- Are rewards meaningful or random (close to 0.5)?
- Does the model have sufficient variance?

**Expected output if working:**
- ✓ Correct reasoning scores higher than incorrect (e.g., 0.75 vs 0.45)
- ✓ Nonsense reasoning scores low (< 0.4)
- ✓ Reward range > 0.1
- ✓ Mean reward NOT close to 0.5

**Expected output if broken:**
- ✗ All rewards cluster around 0.5 ± 0.05
- ✗ No discrimination between correct/incorrect
- ✗ Model may not have been trained

**If broken:** Re-run Phase 1 (reward model training) before continuing policy training.

---

### 2. Test Answer Extraction

```bash
python diagnose_answer_extraction.py
```

**What it tests:**
- Does Qwen3 generate answers in the expected format?
- Does the extraction function work on real generations?
- What percentage of answers are extractable?

**Expected output if working:**
- LLM generates "The answer is 42" or "#### 42"
- Extraction success rate > 80%

**Expected output if broken:**
- LLM generates "Therefore: 42" or just "42" (no marker)
- Extraction success rate 0-30%
- This explains 0% accuracy!

**If broken:** See fixes below.

---

### 3. Analyze Full Trajectory

```bash
# With reward model only
python diagnose_trajectory.py --reward_model outputs/reward_model_final

# With policy checkpoint
python diagnose_trajectory.py \
    --reward_model outputs/reward_model_final \
    --policy outputs/training_2gpu/policy_checkpoint_latest.pt
```

**What it tests:**
- What actions does the policy take?
- How many reasoning steps are generated?
- What rewards are assigned?
- Is the final answer extracted?

**Look for:**
- Trajectory length < 5 → Policy learned to terminate early
- Final answer: None → Answer extraction failing
- Average reward ~0.5 → Reward model broken
- Action distribution: 80%+ TERMINATE → Policy collapsed

---

## Common Issues and Fixes

### Issue #1: Reward Model Giving Random Scores

**Symptoms:**
- `diagnose_reward_model.py` shows rewards clustered around 0.5
- No discrimination between correct/incorrect
- Mean reward 0.45-0.55

**Root Cause:**
The reward model was never properly trained in Phase 1, or training failed silently.

**Fix:**
```bash
# Re-run reward model training
python train_optimized.py \
    --model_config config/model_config_2gpu.yaml \
    --train_config config/training_config_2gpu.yaml \
    --output_dir outputs/reward_retrain \
    --skip_policy_training

# Then resume policy training with new reward model
python train_optimized.py \
    --model_config config/model_config_2gpu.yaml \
    --train_config config/training_config_2gpu.yaml \
    --output_dir outputs/policy_retrain \
    --skip_reward_training \
    --reward_model_path outputs/reward_retrain/reward_model_final
```

---

### Issue #2: Answer Extraction Failing

**Symptoms:**
- `diagnose_answer_extraction.py` shows 0% extraction success
- LLM generates valid answers but not in "####" or "answer is" format
- Final answer always None in trajectories

**Root Cause:**
The LLM wasn't prompted to generate answers in the expected GSM8K format.

**Fix Option A: Modify Prompt (Easiest)**

Edit `pgts_qwen/models/qwen3_wrapper.py`:

```python
def generate_step(self, problem: str, reasoning_chain: List[str], ...):
    # Add instruction to the system prompt
    system_prompt = """Solve the math problem step by step.
    When you reach the final answer, write it as:
    #### <answer>

    For example: #### 42"""

    # ... rest of generation code
```

**Fix Option B: Force Final Answer Generation**

Edit `pgts_qwen/tree_search/search_algorithm.py`:

```python
# In execute_action(), when action == TERMINATE:
def action_terminate(self, tree, problem):
    # Generate final answer explicitly
    current_chain = [n.content for n in tree.get_current_path()[1:]]

    final_prompt = f"""Based on the reasoning above:
    {chr(10).join(current_chain)}

    What is the final numerical answer? Respond with just: #### <number>"""

    answer_text, _ = self.reasoning_generator.generate_step(
        problem, current_chain, max_length=50
    )

    # Extract and store
    final_answer = self.extract_answer(answer_text)
    return True, 0.0, final_answer
```

**Fix Option C: Better Extraction Regex**

Edit `pgts_qwen/tree_search/search_algorithm.py`:

```python
import re

def extract_answer(self, text: str) -> Optional[str]:
    # Try GSM8K format
    if "####" in text:
        return text.split("####")[-1].strip()

    # Try "answer is" format
    if "answer is" in text.lower():
        parts = text.lower().split("answer is")
        # Extract full number (not just first word)
        match = re.search(r'-?\d+\.?\d*', parts[-1])
        if match:
            return match.group(0)

    # Try to extract ANY number from the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]  # Return last number found

    return None
```

---

### Issue #3: Policy Learning to Terminate Early

**Symptoms:**
- Trajectory length decreasing over iterations (7 → 5 → 3 → 2)
- 0% accuracy persists
- Policy always chooses TERMINATE

**Root Cause:**
The reward structure incentivizes early termination:
- Wrong answer: -5.0
- Step penalty: -0.1 per step
- Longer trajectories: More negative reward
- Policy learns: "Terminate immediately to minimize penalty"

**Fix: Change Reward Shaping**

Edit `pgts_qwen/training/ppo_trainer.py`:

```python
def compute_advantages(self, trajectory):
    # OLD (BAD):
    # final_reward = -5.0 if wrong
    # step_penalty = -0.1 * num_steps
    # Total: -5.0 - 1.0 = -6.0 for 10 steps

    # NEW (GOOD):
    if trajectory.is_correct:
        final_reward = 10.0
        step_penalty = 0.0  # No penalty for correct solutions
    else:
        final_reward = 0.0  # Neutral, not -5.0
        step_penalty = -0.05 * num_steps  # Smaller penalty

    # Alternative: Only give reward for correct, ignore wrong
    # This is called "sparse rewards" and works better for exploration
```

---

### Issue #4: No Exploration (All Trajectories Wrong)

**Symptoms:**
- 0% accuracy for many iterations
- Policy never sees a correct example
- Bootstrap problem: Can't learn without positive examples

**Fix: Add Curriculum Learning**

```python
# Start with EASIER problems
easy_problems = [p for p in train_problems if count_numbers(p) <= 2]

# Or use temperature annealing
search_config.temperature = 2.0  # High exploration initially
# Decrease over time: temp = 2.0 * (0.95 ** iteration)

# Or add epsilon-greedy exploration
if random.random() < 0.2:  # 20% random actions
    action = random.choice(valid_actions)
else:
    action = policy.select_action(...)
```

---

## After Running Diagnostics

Based on the diagnostic results, you'll likely find one of these issues:

| Diagnostic Result | Issue | Fix |
|------------------|-------|-----|
| Reward model scores all ~0.5 | PRM not trained | Re-run Phase 1 |
| Answer extraction 0% success | Prompt doesn't force format | Modify prompt or extraction |
| Trajectory length < 3 steps | Early termination learned | Change reward shaping |
| All actions are TERMINATE | Policy collapsed | Reset policy, add exploration |
| Rewards are good but accuracy 0% | LLM not capable | Use larger model or easier problems |

---

## Emergency Reset

If training is completely broken:

```bash
# 1. Stop current training
pkill -f train_optimized.py

# 2. Re-train reward model from scratch
python train_optimized.py \
    --model_config config/model_config_fast.yaml \
    --train_config config/training_config_fast.yaml \
    --output_dir outputs/fresh_start \
    --skip_policy_training

# 3. Test reward model
python diagnose_reward_model.py outputs/fresh_start/reward_model_final

# 4. If reward model works, start policy training
python train_optimized.py \
    --model_config config/model_config_fast.yaml \
    --train_config config/training_config_fast.yaml \
    --output_dir outputs/fresh_start \
    --skip_reward_training \
    --reward_model_path outputs/fresh_start/reward_model_final
```

---

## Expected Behavior (Healthy Training)

For comparison, here's what **good** training should look like:

```
Iteration 1: {
    'accuracy': 0.03,  # 3% - at least SOME correct
    'avg_trajectory_length': 8.5,  # Reasonable length
    'entropy': 1.35,  # High exploration
    'value_loss': 2.1,  # Will be high initially
}

Iteration 10: {
    'accuracy': 0.12,  # 12% - improving
    'avg_trajectory_length': 9.2,  # Stable
    'entropy': 1.15,  # Decreasing (more confident)
    'value_loss': 1.3,  # Decreasing
}

Iteration 50: {
    'accuracy': 0.35,  # 35% - significant progress
    'avg_trajectory_length': 8.8,  # Stable
    'entropy': 0.85,  # Low (confident)
    'value_loss': 0.6,  # Low (good predictions)
}
```

If you're seeing **persistent 0% accuracy and decreasing trajectory length**, the training is fundamentally broken and needs intervention.
