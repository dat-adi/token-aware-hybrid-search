# PGTS Training Troubleshooting - Quick Reference

## Your Current Symptoms

```
Iteration 1-7: ALL showing:
- accuracy: 0.0%  ← CRITICAL ISSUE
- avg_trajectory_length: 2.5-7 steps  ← Too short (should be 8-15)
- entropy: 1.26-1.30  ← Not decreasing
- value_loss: Fluctuating wildly
```

## Diagnosis Steps (Run on Test Server)

### Step 1: Run All Diagnostics (5 minutes)

```bash
cd /home/datadi/Code/token-aware-hybrid-search/pgts_qwen
chmod +x run_all_diagnostics.sh
./run_all_diagnostics.sh
```

This will tell you EXACTLY what's broken.

### Step 2: Quick Individual Tests

```bash
# Test 1: Is reward model working? (MOST LIKELY ISSUE)
python diagnose_reward_model.py outputs/reward_model_final

# Test 2: Can we extract answers from generations?
python diagnose_answer_extraction.py

# Test 3: What happens during a trajectory?
python diagnose_trajectory.py --reward_model outputs/reward_model_final
```

---

## Most Likely Root Causes (In Order of Probability)

### 1. Reward Model Not Trained (80% probability)

**How to check:**
```bash
python diagnose_reward_model.py outputs/reward_model_final
```

**If broken, you'll see:**
- ✗ All rewards cluster around 0.5 ± 0.05
- ✗ No discrimination between correct/incorrect reasoning
- ✗ Reward range < 0.1

**Why this causes 0% accuracy:**
- PRM gives random scores → Policy gets no signal
- All trajectories get similar bad rewards → Advantages normalize to ~0
- Policy learns to terminate early (minimize step penalty)
- Never generates actual answers → 0% accuracy

**Fix:**
Re-run Phase 1 (reward model training):
```bash
python train_optimized.py \
    --model_config config/model_config_2gpu.yaml \
    --train_config config/training_config_2gpu.yaml \
    --output_dir outputs/reward_retrain \
    --skip_policy_training
```

---

### 2. Answer Extraction Failing (15% probability)

**How to check:**
```bash
python diagnose_answer_extraction.py
```

**If broken, you'll see:**
- ✗ Extraction success rate: 0-30%
- LLM generates valid reasoning but no "####" or "answer is" markers

**Why this causes 0% accuracy:**
- LLM solves problem correctly
- But doesn't format answer as expected
- `extract_answer()` returns None
- Evaluation always returns False → 0% accuracy

**Fix:**
See `DIAGNOSTIC_GUIDE.md` Issue #2 for three fix options:
1. Modify generation prompt (easiest)
2. Force final answer generation
3. Improve extraction regex

---

### 3. Premature Termination Learned (5% probability)

**How to check:**
```bash
python diagnose_trajectory.py --reward_model outputs/reward_model_final
```

**If this is the issue:**
- Trajectory length decreasing over iterations (7→5→3→2)
- Policy action distribution: >80% TERMINATE

**Why this happens:**
- Wrong answer penalty: -5.0
- Step penalty: -0.1 per step
- Policy learns: "Terminate immediately to avoid -5.0 - (steps * 0.1)"

**Fix:**
Modify reward shaping in `training/ppo_trainer.py:160-172`

---

## Quick Decision Tree

```
Run: python diagnose_reward_model.py outputs/reward_model_final

├─ Rewards all ~0.5, no discrimination?
│  └─ YES → Reward model broken → Re-run Phase 1
│
├─ Rewards look good (correct > incorrect)?
│  └─ Run: python diagnose_answer_extraction.py
│     │
│     ├─ Extraction success < 50%?
│     │  └─ YES → Fix answer extraction → See DIAGNOSTIC_GUIDE.md Issue #2
│     │
│     └─ Extraction works fine?
│        └─ Run: python diagnose_trajectory.py --reward_model outputs/reward_model_final
│           │
│           ├─ Trajectory length < 5?
│           │  └─ YES → Early termination → Fix reward shaping
│           │
│           └─ Everything looks normal?
│              └─ Model may be too weak → Try larger model or easier problems
```

---

## Expected Diagnostic Output (Healthy System)

### Reward Model (Good):
```
Mean Rewards:
  Correct:   0.7234
  Incorrect: 0.4521
  Nonsense:  0.2143

✓ PASS: Correct reasoning scores higher than incorrect
✓ PASS: Nonsense reasoning scores lower than correct
✓ PASS: Rewards show meaningful variance (range: 0.5091)
✓ PASS: Mean reward 0.4633 is away from 0.5
✓ PASS: Correct final answer scores high (0.7234)
```

### Answer Extraction (Good):
```
✓ Input: "The answer is 42"
  Extracted: 42

✓ Input: "#### 345"
  Extracted: 345

Extraction success rate: 3/3 (100.0%)
```

### Trajectory (Good):
```
Trajectory length: 9 steps
Final answer: 18
Correct: True

Action distribution:
  EXPAND: 7 (77.8%)
  BRANCH: 1 (11.1%)
  BACKTRACK: 0 (0.0%)
  TERMINATE: 1 (11.1%)
```

---

## Files Created for You

1. **`diagnose_reward_model.py`** - Tests if PRM gives meaningful signals
2. **`diagnose_answer_extraction.py`** - Tests if answers are extractable
3. **`diagnose_trajectory.py`** - Runs full search and analyzes what happens
4. **`run_all_diagnostics.sh`** - Runs all three tests automatically
5. **`DIAGNOSTIC_GUIDE.md`** - Detailed guide with fixes for each issue
6. **`TROUBLESHOOTING_SUMMARY.md`** - This file (quick reference)

---

## What to Do Right Now

1. **Pull these files to your test server:**
   ```bash
   cd /path/to/test/server
   git pull origin trial/adi
   cd pgts_qwen
   ```

2. **Run the comprehensive diagnostic:**
   ```bash
   chmod +x run_all_diagnostics.sh
   ./run_all_diagnostics.sh
   ```

3. **Read the output carefully** - It will tell you exactly what's wrong

4. **Apply the recommended fix** from `DIAGNOSTIC_GUIDE.md`

5. **Re-run diagnostics** to verify the fix worked

6. **Resume training** with the fixed component

---

## Contact Points

If diagnostics show:
- **Reward model broken** → Re-train Phase 1
- **Answer extraction broken** → Fix prompt or extraction function
- **Early termination** → Fix reward shaping
- **Everything looks good but still 0%** → Model may be too weak for task

The diagnostic scripts will give you specific recommendations for each case.

---

## Training Should Be Paused

Your current training is:
1. Wasting GPU time (0% accuracy won't improve without fixing root cause)
2. Potentially making it worse (policy learning bad behaviors)
3. Overwriting logs that could be useful

**Recommendation:** Stop training, run diagnostics, fix the issue, then restart.
