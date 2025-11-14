# Fast PRM Data Generation Guide

## The Problem

Your current setup generates incorrect reasoning paths by sampling from Qwen3-8B with high temperature:
- **3 incorrect paths × 7,000 problems = 21,000 model inference runs**
- **~3-5 seconds per inference = 20+ hours**
- No caching - regenerated every time!

## Solutions (Ranked by Speed)

### Option 1: Use PRM800K Dataset (INSTANT - 0 hours!)

OpenAI released a pre-generated dataset specifically for training Process Reward Models on GSM8K!

**Pros:**
- ✅ Instant - just download
- ✅ High quality (human-annotated)
- ✅ 800K examples (way more than you need)
- ✅ Already balanced and formatted

**Cons:**
- ⚠️ Requires download (~2GB)
- ⚠️ May not match your exact data distribution

**How to use:**

```python
# In main_train.py, replace the import:
from data.prm800k_loader import create_prm_dataset_fast

# Then in train_reward_model function:
prm_dataset = create_prm_dataset_fast(
    gsm8k_train_data,
    max_examples=train_config['reward_training']['max_examples']
)
```

**Download PRM800K:**
```bash
# Option A: Try HuggingFace (may not be available yet)
# It should work automatically

# Option B: Manual download
mkdir -p data/prm800k
wget https://github.com/openai/prm800k/archive/refs/heads/main.zip
unzip main.zip -d data/prm800k/
```

---

### Option 2: Fast Synthetic Corruption (2-5 minutes!)

Generate incorrect examples by corrupting correct ones synthetically.

**Pros:**
- ✅ Super fast (2-5 minutes vs 20 hours)
- ✅ No model inference needed
- ✅ No extra VRAM usage
- ✅ Works offline
- ✅ Still effective for training

**Cons:**
- ⚠️ Less realistic than model-generated errors
- ⚠️ May not capture all error types

**How to use:**

```python
# In main_train.py, replace:
from data.reward_annotator_fast import create_prm_dataset_fast

# Then:
prm_dataset = create_prm_dataset_fast(
    gsm8k_train_data,
    num_incorrect_per_problem=3,
    max_examples=50000
)
```

**Corruption strategies:**
- Flip numbers (5 → 7, 10 → 12)
- Change operations (+, -, ×, ÷)
- Introduce calculation errors
- Wrong final answer

---

### Option 3: Cached Model Generation (20 hours first run, 30 seconds after!)

Use the slow model-based generation BUT cache results.

**Pros:**
- ✅ Most realistic incorrect examples
- ✅ Instant on re-runs (cached)
- ✅ Best for research/ablations

**Cons:**
- ⚠️ Still takes 20 hours on first run
- ⚠️ Requires disk space (~500MB)

**How to use:**

```python
# In main_train.py, replace:
from data.reward_annotator_cached import create_prm_dataset_cached

# Then:
prm_dataset = create_prm_dataset_cached(
    gsm8k_train_data,
    reasoning_generator,
    num_incorrect_per_problem=3,
    max_examples=50000,
    cache_dir='data/cached_prm',
    force_regenerate=False  # Set True to regenerate
)
```

**Cache management:**
```bash
# View cached datasets
ls -lh data/cached_prm/

# Clear cache to regenerate
rm -rf data/cached_prm/*.json

# Force regeneration in code
prm_dataset = create_prm_dataset_cached(..., force_regenerate=True)
```

---

### Option 4: vLLM Acceleration (5-8 hours with batching)

Use vLLM for faster batch inference.

**Pros:**
- ✅ 2-4x faster than HuggingFace
- ✅ Realistic model-generated errors
- ✅ Good throughput with batching

**Cons:**
- ⚠️ Still takes 5-8 hours
- ⚠️ Requires vLLM installation
- ⚠️ No hidden states (can't use for policy during generation)

**Setup:**
```bash
pip install vllm
```

**How to use:**

```python
# In main_train.py, modify reasoning_generator initialization:
reasoning_generator = Qwen3ReasoningGenerator(
    model_name=model_config['qwen3']['model_name'],
    temperature=1.0,
    use_vllm=True  # Enable vLLM
)
```

---

## Recommended Workflow

### For Quick Experimentation (Get Started Fast!)
```python
# Use Option 2: Fast Synthetic Corruption
from data.reward_annotator_fast import create_prm_dataset_fast

prm_dataset = create_prm_dataset_fast(
    gsm8k_train_data,
    num_incorrect_per_problem=3,
    max_examples=50000
)
# Takes: 2-5 minutes ✓
```

### For Best Quality (If You Have Time)
```python
# Use Option 3: Cached Model Generation
from data.reward_annotator_cached import create_prm_dataset_cached

prm_dataset = create_prm_dataset_cached(
    gsm8k_train_data,
    reasoning_generator,
    num_incorrect_per_problem=3,
    max_examples=50000
)
# First run: 20 hours
# Subsequent runs: 30 seconds ✓
```

### For Production / Research
```python
# Use Option 1: PRM800K (Industry Standard)
from data.prm800k_loader import create_prm_dataset_fast

prm_dataset = create_prm_dataset_fast(
    gsm8k_train_data,
    max_examples=50000
)
# Takes: Instant (after download) ✓
```

---

## Quick Fix for Your Current Setup

Replace in `main_train.py` (around line 58):

**Before (20 hours):**
```python
from data.reward_annotator import create_prm_dataset

prm_dataset = create_prm_dataset(
    gsm8k_train_data,
    reasoning_generator,
    num_incorrect_per_problem=train_config['reward_training']['num_incorrect_per_problem'],
    max_examples=train_config['reward_training']['max_examples']
)
```

**After (2-5 minutes):**
```python
from data.reward_annotator_fast import create_prm_dataset_fast

# No reasoning_generator needed!
prm_dataset = create_prm_dataset_fast(
    gsm8k_train_data,
    num_incorrect_per_problem=train_config['reward_training']['num_incorrect_per_problem'],
    max_examples=train_config['reward_training']['max_examples']
)
```

---

## Comparison Table

| Method | First Run | Re-run | Quality | VRAM | Notes |
|--------|-----------|--------|---------|------|-------|
| **Current** | 20 hours | 20 hours | High | 16GB | No caching |
| **PRM800K** | Instant* | Instant | Very High | 0GB | *After download |
| **Fast Synthetic** | 2-5 min | 2-5 min | Good | 0GB | Fastest! |
| **Cached** | 20 hours | 30 sec | High | 16GB | Best for re-runs |
| **vLLM** | 5-8 hours | 5-8 hours | High | 16GB | No caching |

---

## Do You Need Model-Generated Errors?

**Short answer: Probably not!**

Research shows that:
1. **Synthetic corruption works well** for training reward models
2. The reward model learns to **distinguish patterns**, not memorize specific errors
3. Fast iteration > perfect data quality (you can experiment more)

**When to use model-generated:**
- Publishing research (need to match baselines)
- Final production model (after prototyping)
- Ablation studies (comparing data quality)

**When to use synthetic:**
- Initial experiments (99% of the time!)
- Rapid prototyping
- Limited compute budget
- Iterating on model architecture

---

## Testing Data Quality

After training reward model, evaluate on held-out set:

```python
# Test reward model accuracy
from evaluation.metrics import evaluate_reward_model

accuracy = evaluate_reward_model(
    reward_model,
    test_data,
    metric='step_accuracy'
)

print(f"Reward model accuracy: {accuracy:.2%}")
```

If accuracy is >70%, your data is good enough!

---

## Summary

**Just want to get started?**
→ Use Option 2 (Fast Synthetic) - 2 minutes!

**Need best quality?**
→ Use Option 1 (PRM800K) - instant after download!

**Doing research?**
→ Use Option 3 (Cached) - wait once, instant forever!

**Have limited time?**
→ Option 2 or Option 1 - both are instant!

---

## See Also

- `data/prm800k_loader.py` - PRM800K loader
- `data/reward_annotator_fast.py` - Fast synthetic corruption
- `data/reward_annotator_cached.py` - Cached generation
- Original: `data/reward_annotator.py` - Slow but realistic
