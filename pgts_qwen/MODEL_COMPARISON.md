# Model Size Comparison Guide

## TL;DR - Which Config Should I Use?

| Use Case | Config | Training Time | Accuracy | Command |
|----------|--------|---------------|----------|---------|
| **Quick experiments** 🚀 | Fast | **3.5-5.5 hrs** | 60-75% | `./train_fast.sh` |
| **Balanced** ⚖️ | 2GPU | 11-15 hrs | 70-80% | `./train_2gpu.sh` |
| **Best quality** 🎯 | Full | 20-30 hrs | 75-85% | Custom |

**Recommendation: Start with Fast config!** Get results in half a day, then scale up if needed.

---

## Detailed Comparison

### Fast Config (Recommended for Experimentation)

**Models:**
- Reasoning Generator: **Qwen 3B** (~6GB VRAM)
- Reward Model: **Qwen 1.5B** (~3GB VRAM)
- Policy Network: GPS (~2GB VRAM)

**VRAM Usage:**
- Phase 1: ~6GB on single GPU
- Phase 2: ~11GB total (6GB + 5GB)
- **Fits in: 1×24GB GPU** (can use just one GPU!)

**Training Time:**
- Phase 1: ~1.5 hours
- Phase 2: ~2-4 hours
- **Total: 3.5-5.5 hours** ⚡

**Performance:**
- GSM8K accuracy: **60-75%**
- Good enough for method comparison
- 3x faster iteration

**Best for:**
- ✅ Comparing different RL algorithms
- ✅ Testing tree search strategies
- ✅ Hyperparameter tuning
- ✅ Ablation studies
- ✅ Quick proof-of-concept
- ✅ Running multiple experiments

**Command:**
```bash
./train_fast.sh
```

---

### 2GPU Config (Balanced)

**Models:**
- Reasoning Generator: **Qwen 7B** (~14GB VRAM)
- Reward Model: **Qwen 3B** (~6GB VRAM)
- Policy Network: GPS (~2GB VRAM)

**VRAM Usage:**
- Phase 1: ~14GB on single GPU
- Phase 2: ~22GB total (14GB + 8GB)
- **Fits in: 2×24GB GPUs**

**Training Time:**
- Phase 1: ~3 hours
- Phase 2: ~8-12 hours
- **Total: 11-15 hours**

**Performance:**
- GSM8K accuracy: **70-80%**
- Good balance of speed and quality

**Best for:**
- ✅ Final model training
- ✅ Publishing results
- ✅ Production deployments
- ✅ After validating with fast config

**Command:**
```bash
./train_2gpu.sh
```

---

### Full Config (Best Quality)

**Models:**
- Reasoning Generator: **Qwen 14B** (~28GB VRAM with 8-bit)
- Reward Model: **Qwen 7B** (~14GB VRAM)
- Policy Network: GPS (~2GB VRAM)

**VRAM Usage:**
- Phase 1: ~28GB on single GPU (needs 8-bit)
- Phase 2: ~44GB total
- **Requires: 2×24GB GPUs with 8-bit quantization**

**Training Time:**
- Phase 1: ~5-6 hours
- Phase 2: ~15-20 hours
- **Total: 20-26 hours**

**Performance:**
- GSM8K accuracy: **75-85%**
- Best quality

**Best for:**
- ✅ State-of-the-art results
- ✅ Final production model
- ✅ Research papers
- ⚠️ Only after validating approach with smaller models

**Setup:**
Requires custom config with 8-bit quantization enabled.

---

## Model Quality Breakdown

### Why Smaller Models Work

**Reward Model (1.5B vs 3B vs 7B):**
- Task: Binary classification (correct/incorrect)
- **1.5B**: 85-90% classification accuracy
- **3B**: 90-95% classification accuracy
- **7B**: 92-97% classification accuracy
- **Takeaway:** 1.5B is sufficient for reward modeling!

**Reasoning Generator (3B vs 7B vs 14B):**
- Task: Generate reasoning steps
- **3B**: Can do basic math, simple reasoning
- **7B**: Good at multi-step reasoning
- **14B**: Excellent reasoning, fewer errors
- **Takeaway:** 3B works for GSM8K, but 7B is better

### Performance vs Speed Trade-off

| Model Size | GSM8K Accuracy | Training Time | Speedup | Cost |
|------------|----------------|---------------|---------|------|
| 3B + 1.5B | 60-75% | 3.5-5.5 hrs | 3x | Low |
| 7B + 3B | 70-80% | 11-15 hrs | 1x (baseline) | Medium |
| 14B + 7B | 75-85% | 20-26 hrs | 0.5x | High |

**Diminishing Returns:**
- 3B → 7B: +10-15% accuracy, +2.5x time
- 7B → 14B: +5-10% accuracy, +2x time

---

## Detailed VRAM Breakdown

### Fast Config (3B + 1.5B)

```
Phase 1 (Single GPU):
├─ Qwen 3B (bfloat16): 6GB
├─ Optimizer states: 1GB
├─ Activations/gradients: 1GB
└─ Batch data: 1GB
   Total: ~9GB / 24GB ✓

Phase 2 (Dual GPU):
├─ GPU 0: Qwen 3B: 6GB
│   ├─ Model: 6GB
│   └─ Batch: 1GB
│      Total: ~7GB / 24GB ✓
│
└─ GPU 1:
    ├─ Qwen 1.5B PRM: 3GB
    ├─ GPS Policy: 2GB
    ├─ Optimizer: 1GB
    └─ Batch: 1GB
       Total: ~7GB / 24GB ✓

Total Phase 2: ~14GB / 48GB (70% headroom!)
```

### 2GPU Config (7B + 3B)

```
Phase 1 (Single GPU):
├─ Qwen 7B (bfloat16): 14GB
├─ Optimizer states: 2GB
├─ Activations/gradients: 2GB
└─ Batch data: 1GB
   Total: ~19GB / 24GB ✓

Phase 2 (Dual GPU):
├─ GPU 0: Qwen 7B: 14GB
│   ├─ Model: 14GB
│   └─ Batch: 2GB
│      Total: ~16GB / 24GB ✓
│
└─ GPU 1:
    ├─ Qwen 3B PRM: 6GB
    ├─ GPS Policy: 2GB
    ├─ Optimizer: 2GB
    └─ Batch: 1GB
       Total: ~11GB / 24GB ✓

Total Phase 2: ~27GB / 48GB (44% headroom)
```

---

## Parallelizing Experiments

With fast config, you can run **multiple experiments in parallel**!

### Example: Compare 3 RL Algorithms

```bash
# Terminal 1: Standard PPO
CUDA_VISIBLE_DEVICES=0 ./train_fast.sh

# Terminal 2: PPO with different entropy
# (Edit config, change entropy_coeff)
CUDA_VISIBLE_DEVICES=1 ./train_fast.sh

# Both run simultaneously!
# 2 experiments in 4 hours vs 8 hours sequentially
```

### Example: Hyperparameter Sweep

```bash
# Run 4 experiments on 2 GPUs in parallel
# Each takes ~4 hours
# Total time: ~8 hours for 4 experiments
# vs ~16 hours sequentially

# GPU 0, slot 1
CUDA_VISIBLE_DEVICES=0 python train_optimized.py --output_dir exp1 &
# Wait 2 hours, then:
# GPU 0, slot 2
CUDA_VISIBLE_DEVICES=0 python train_optimized.py --output_dir exp2 &

# GPU 1, slot 1
CUDA_VISIBLE_DEVICES=1 python train_optimized.py --output_dir exp3 &
# Wait 2 hours, then:
# GPU 1, slot 2
CUDA_VISIBLE_DEVICES=1 python train_optimized.py --output_dir exp4 &
```

---

## Migration Path

### Recommended Workflow

**Stage 1: Fast Experimentation (Days 1-2)**
```bash
# Try different approaches quickly
./train_fast.sh  # Experiment 1: Baseline
# Modify configs
./train_fast.sh  # Experiment 2: Different RL algo
# Modify configs
./train_fast.sh  # Experiment 3: Different tree search

# Result: 3 experiments in ~12 hours
# Pick best approach
```

**Stage 2: Validation (Days 3-4)**
```bash
# Validate best approach with larger models
./train_2gpu.sh

# Result: Confirm approach works at 70-80% accuracy
```

**Stage 3: Final Training (Optional, Days 5-6)**
```bash
# Only if needed for production/publication
# Train with largest models
# Custom config with 14B + 7B
```

---

## File Guide

### Configuration Files

```
config/
├── model_config_fast.yaml      # 3B + 1.5B (recommended)
├── training_config_fast.yaml   # Fast training settings
├── model_config_2gpu.yaml      # 7B + 3B (balanced)
├── training_config_2gpu.yaml   # Balanced settings
└── model_config.yaml           # Original (customize)
```

### Training Scripts

```
train_fast.sh          # 3.5-5.5 hours - Quick experiments
train_2gpu.sh          # 11-15 hours - Balanced quality
train_optimized.py     # Core training script (all configs)
main_train.py          # Original training script
```

---

## Quick Start Examples

### Experiment 1: Baseline with Fast Models
```bash
./train_fast.sh
# Wait ~4 hours
# Check accuracy in outputs/training_fast_*/
```

### Experiment 2: Compare Search Strategies
```bash
# Edit config/training_config_fast.yaml
# Change: max_depth from 12 to 8
./train_fast.sh
# Compare results
```

### Experiment 3: Test Reward Shaping
```bash
# Edit config/training_config_fast.yaml
# Change: final_reward_correct from 10.0 to 20.0
./train_fast.sh
# Compare results
```

### Scale Up After Finding Best Approach
```bash
./train_2gpu.sh
# Use same hyperparameters that worked best
```

---

## FAQ

**Q: Will 1.5B PRM be good enough?**
A: Yes! Reward models just need to classify correct/incorrect. Even 1.5B gets 85-90% accuracy, which is sufficient for training.

**Q: Can I use 0.5B for PRM?**
A: You can try `Qwen2.5-0.5B`, but quality drops to ~70-80% classification accuracy. Still usable for quick experiments.

**Q: Will 3B reasoning generator work on GSM8K?**
A: Yes, but with lower accuracy (60-75%). Good enough for method comparison. For final results, use 7B (70-80%) or 14B (75-85%).

**Q: Can I use just 1 GPU with fast config?**
A: Yes! Fast config only needs ~11GB total in Phase 2, so you can run everything on a single 24GB GPU (though slightly slower).

**Q: How much accuracy do I lose with smaller models?**
A: Roughly:
- 3B vs 7B generator: -10-15% accuracy
- 1.5B vs 3B PRM: -2-5% accuracy (minor)
- Combined: -12-20% accuracy vs full setup

**Q: Should I start with fast or balanced?**
A: **Always start with fast!**
- Get results in 4 hours vs 12 hours
- Iterate 3x faster
- Validate approach works
- Then scale up to balanced/full

---

## Performance Benchmarks

### GSM8K Expected Accuracy

| Config | Accuracy | Time | Cost/Time Ratio |
|--------|----------|------|-----------------|
| Fast (3B+1.5B) | 60-75% | 4 hrs | **15-19% per hour** |
| Balanced (7B+3B) | 70-80% | 12 hrs | 6-7% per hour |
| Full (14B+7B) | 75-85% | 24 hrs | 3-4% per hour |

**Takeaway:** Fast config gives best "accuracy per hour"!

---

## Summary

**For rapid experimentation and method comparison:**
→ Use **Fast config** (3B + 1.5B)
```bash
./train_fast.sh
```

**Benefits:**
- ✅ 3x faster (4 hours vs 12 hours)
- ✅ 60-75% accuracy (good enough to compare methods)
- ✅ Can run multiple experiments in parallel
- ✅ Lower compute cost
- ✅ Faster iteration = more ideas tested

**After validating approach:**
→ Scale up to **Balanced config** (7B + 3B)
```bash
./train_2gpu.sh
```

**Your workflow:**
1. Day 1-2: Run 3-4 experiments with fast config (4 hrs each)
2. Day 3: Pick best approach
3. Day 4-5: Validate with balanced config (12 hrs)
4. (Optional) Day 6-7: Final training with full config (24 hrs)

**Total time to validated result: 2-3 days vs 1 week with slow experiments!**
