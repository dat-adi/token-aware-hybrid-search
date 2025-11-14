# PGTS Quick Start - 2x24GB GPUs

## TL;DR

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training (one command!)
./train_2gpu.sh

# 3. Wait 12-18 hours

# 4. Evaluate
python main_eval.py \
    --policy_checkpoint outputs/training_2gpu_*/policy_final.pt \
    --reward_model_path outputs/training_2gpu_*/reward_model_final
```

---

## What's Actually Happening

### NOT Student-Teacher! It's Reinforcement Learning:

```
┌─────────────────────────────────────────────────────┐
│  Phase 1: Train Reward Model (4-6 hours, GPU 0)    │
├─────────────────────────────────────────────────────┤
│  1. Qwen3-8B generates reasoning samples           │
│  2. Qwen3-4B learns to classify correctness        │
│  3. Save reward model ✓                            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Phase 2: Train Policy (8-12 hours, Both GPUs)     │
├─────────────────────────────────────────────────────┤
│  GPU 0: Qwen3-8B (generates steps)                 │
│  GPU 1: Qwen3-4B (evaluates) + Policy (learns)     │
│                                                     │
│  Loop:                                              │
│    Policy → action → Generator → step →            │
│    Reward → PPO update → better policy             │
└─────────────────────────────────────────────────────┘
```

### The Three Components

| Component | Model | Role | Trained? | VRAM |
|-----------|-------|------|----------|------|
| **Policy Network** | GPS Transformer | Learns to search | ✅ PPO | ~2GB |
| **Generator** | Qwen3-8B | Creates reasoning | ❌ Frozen | ~16GB |
| **Reward Model** | Qwen3-4B | Judges quality | ✅ Fine-tuned | ~8GB |

---

## Memory Budget

```
Phase 1: GPU 0 only
├─ Qwen3-8B: 16GB
└─ Headroom:  8GB ✓

Phase 2: Both GPUs
├─ GPU 0: Qwen3-8B (16GB)
└─ GPU 1: Qwen3-4B (8GB) + Policy (2GB) = 10GB ✓

Total: 26GB / 48GB available
```

---

## Files Created

### Configurations (optimized for 2x24GB)
- `config/model_config_2gpu.yaml` - Model selection
- `config/training_config_2gpu.yaml` - Training hyperparameters

### Scripts
- `train_optimized.py` - Smart GPU placement
- `train_2gpu.sh` - One-command training

### Documentation
- `TRAINING_GUIDE_2GPU.md` - Full guide
- `QUICK_START.md` - This file

---

## If You Run Out of Memory

### Quick Fix #1: Enable 8-bit
Edit `config/model_config_2gpu.yaml`:
```yaml
load_in_8bit: true  # For both qwen3 and reward_model
```
Saves ~40% VRAM, slightly slower.

### Quick Fix #2: Smaller Batch
Edit `config/training_config_2gpu.yaml`:
```yaml
policy_training:
  batch_size: 2  # Down from 4
```

### Quick Fix #3: Smaller Models
```yaml
qwen3:
  model_name: "Qwen/Qwen2.5-3B"  # Instead of 7B
reward_model:
  model_name: "Qwen/Qwen2.5-1.5B"  # Instead of 3B
```

---

## Monitor Training

```bash
# Terminal 1: GPU usage
watch -n 1 nvidia-smi

# Terminal 2: Training log
tail -f outputs/training_2gpu_*/training.log
```

Expected output:
```
Phase 1: ~4-6 hours
  Generating PRM data: 2-3 hours
  Training reward model: 2-3 hours

Phase 2: ~8-12 hours
  100 iterations × 16 problems
  ~5-7 minutes per iteration
  Checkpoints saved every 10 iterations
```

---

## Key Differences from Original

| Original | 2x24GB Version | Why? |
|----------|----------------|------|
| Batch size: 8 | Batch size: 4 | Memory |
| 32 problems/iter | 16 problems/iter | Speed |
| Max depth: 20 | Max depth: 15 | Memory |
| Max nodes: 100 | Max nodes: 80 | Memory |
| No 8-bit | Optional 8-bit | Flexibility |

---

## Common Questions

**Q: Why is Qwen3-4B the "teacher" if it's smaller than Qwen3-8B?**

A: It's NOT a teacher! It's a **reward evaluator**. Think of it like:
- Qwen3-8B = Writer (generates text)
- Qwen3-4B = Judge (says if it's correct)
- Policy = Strategist (decides what to write)

The judge doesn't need to be a better writer, just a good judge!

**Q: Do I need to pretrain on GSM8K?**

A: No! Qwen models already know math. The reward model gets fine-tuned automatically in Phase 1.

**Q: Can I pause and resume?**

A: Yes! After Phase 1:
```bash
python train_optimized.py \
    --skip_reward_training \
    --reward_model_path outputs/.../reward_model_final \
    --output_dir outputs/resume
```

**Q: How do I know if it's working?**

A: Watch for:
- Phase 1: Training loss decreasing
- Phase 2: Accuracy increasing, policy_loss/value_loss decreasing

---

## Next Steps After Training

1. **Evaluate**: Run `main_eval.py` on test set
2. **Tune**: Adjust hyperparameters based on results
3. **Scale**: Try larger models if you have more GPUs
4. **Adapt**: Use on other reasoning tasks

See `TRAINING_GUIDE_2GPU.md` for full details!
