# PGTS Training Guide for 2x24GB GPUs

## Understanding the PGTS Architecture

### It's NOT Student-Teacher Distillation!

This is a **Reinforcement Learning** system with three components:

#### 1. Policy Network (What's Being Trained)
- **GPS Policy Network**: A graph transformer that learns search strategies
- **Actions**: EXPAND, BRANCH, BACKTRACK, TERMINATE
- **Training**: PPO (Proximal Policy Optimization)
- **Memory**: ~2GB VRAM

#### 2. Reasoning Generator (The Environment)
- **Model**: Qwen3-8B (or Qwen2.5-7B)
- **Purpose**: Generates reasoning steps when policy chooses actions
- **Role**: Acts as an environment, NOT a teacher
- **Status**: Frozen (not trained)
- **Memory**: ~16GB VRAM in bfloat16

#### 3. Reward Model (The Evaluator)
- **Model**: Qwen3-4B (or Qwen2.5-3B)
- **Purpose**: Evaluates if reasoning steps are correct
- **Training**: Fine-tuned on GSM8K in Phase 1
- **Task**: Binary classification (correct/incorrect)
- **Memory**: ~8GB VRAM in bfloat16

### Why Is the Reward Model Smaller?

The reward model is INTENTIONALLY smaller because:
1. **Speed**: Called frequently during search, needs fast inference
2. **Simple Task**: Binary classification vs. text generation
3. **Memory**: Allows room for policy network on same GPU

### Do Models Need GSM8K Pretraining?

**No!** The base Qwen models already have reasoning capabilities:
- Qwen3-8B: General reasoning and math skills
- Qwen3-4B: General language understanding

The reward model gets fine-tuned on GSM8K during Phase 1 automatically.

---

## Training Pipeline

### Phase 1: Reward Model Training
1. Use Qwen3-8B to generate diverse reasoning samples (correct + incorrect)
2. Fine-tune Qwen3-4B to classify step correctness
3. Save reward model for Phase 2

**GPU Usage**: Only GPU 0 (~16GB)

### Phase 2: Policy Network Training (PPO)
1. Policy network selects actions
2. Qwen3-8B generates reasoning steps
3. Qwen3-4B evaluates step quality
4. PPO updates policy based on rewards

**GPU Usage**:
- GPU 0: Qwen3-8B (~16GB)
- GPU 1: Qwen3-4B (~8GB) + GPS Policy (~2GB)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Optional: Flash Attention for faster training
pip install flash-attn --no-build-isolation
```

### 2. Run Training

```bash
chmod +x train_2gpu.sh
./train_2gpu.sh
```

This will:
- Train reward model (Phase 1)
- Train policy network (Phase 2)
- Save checkpoints to `outputs/training_2gpu_<timestamp>/`

### 3. Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# View training log
tail -f outputs/training_2gpu_<timestamp>/training.log
```

---

## Memory Optimization

### Current Configuration (2x24GB)

**Phase 1** (Single GPU):
- Qwen3-8B for data generation: ~16GB
- Total: ~16GB (fits in 24GB with headroom)

**Phase 2** (Dual GPU):
- GPU 0: Qwen3-8B: ~16GB
- GPU 1: Qwen3-4B + Policy: ~10GB
- Total: ~26GB (fits in 48GB with headroom)

### If You Run Out of Memory

#### Option 1: Enable 8-bit Quantization
Edit `config/model_config_2gpu.yaml`:
```yaml
qwen3:
  load_in_8bit: true  # Reduces VRAM by ~40%

reward_model:
  load_in_8bit: true
```

#### Option 2: Reduce Batch Size
Edit `config/training_config_2gpu.yaml`:
```yaml
policy_training:
  batch_size: 2  # Reduce from 4
  problems_per_iteration: 8  # Reduce from 16
```

#### Option 3: Reduce Search Depth
```yaml
search:
  max_depth: 10  # Reduce from 15
  max_nodes: 50  # Reduce from 80
```

#### Option 4: Use Smaller Models
```yaml
qwen3:
  model_name: "Qwen/Qwen2.5-3B"  # Instead of 7B

reward_model:
  model_name: "Qwen/Qwen2.5-1.5B"  # Instead of 3B
```

---

## Training Configuration

### Files Created for 2x24GB Setup

1. **`config/model_config_2gpu.yaml`**
   - Optimized model selection
   - Memory-efficient settings

2. **`config/training_config_2gpu.yaml`**
   - Reduced batch sizes
   - Adjusted iteration counts
   - Memory optimization flags

3. **`train_optimized.py`**
   - Explicit GPU placement
   - Memory management
   - Better error handling

4. **`train_2gpu.sh`**
   - One-command training
   - Environment setup
   - Logging

### Key Hyperparameters

**Reward Model Training:**
- Batch size: 8 (effective 16 with gradient accumulation)
- Learning rate: 2e-5
- Epochs: 3
- Max examples: 50,000

**Policy Training (PPO):**
- Batch size: 4
- Learning rate: 1e-5
- Iterations: 100
- Problems per iteration: 16
- PPO epochs: 4
- Clip epsilon: 0.2

**Search Parameters:**
- Max depth: 15
- Max nodes: 80
- Temperature: 0.8

---

## Expected Training Time

On 2x24GB GPUs (e.g., RTX 3090, RTX 4090):

- **Phase 1 (Reward Model)**: 4-6 hours
  - Data generation: 2-3 hours
  - Model training: 2-3 hours

- **Phase 2 (Policy Network)**: 8-12 hours
  - 100 iterations × 16 problems
  - ~5-7 minutes per iteration

**Total**: 12-18 hours for full training

---

## Checkpoints and Resume Training

### Automatic Checkpointing

Checkpoints are saved every 10 iterations:
```
outputs/training_2gpu_<timestamp>/
├── reward_model_final/           # Trained reward model
├── policy_checkpoints/
│   ├── checkpoint_10.pt
│   ├── checkpoint_20.pt
│   └── ...
└── policy_final.pt               # Final policy
```

### Resume from Checkpoint

If training is interrupted, resume from saved reward model:

```bash
python train_optimized.py \
    --skip_reward_training \
    --reward_model_path outputs/training_2gpu_<timestamp>/reward_model_final \
    --output_dir outputs/training_resume
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Enable 8-bit quantization (see above)
2. Reduce batch size to 2
3. Reduce max_depth to 10
4. Use smaller base models

### Slow Training

**Solutions**:
1. Install Flash Attention: `pip install flash-attn`
2. Reduce `max_examples` in reward training
3. Use vLLM for inference (if you don't need hidden states)

### Poor Performance

**Symptoms**: Low accuracy after training

**Solutions**:
1. Increase `num_iterations` to 200
2. Increase `problems_per_iteration`
3. Tune reward shaping (`final_reward_correct`, `step_penalty`)
4. Use larger base models (Qwen3-14B if you have more memory)

### GPU Imbalance

**Symptoms**: One GPU full, other GPU idle

**Solutions**:
- This is normal! Different phases use different GPUs
- Phase 1: Only GPU 0
- Phase 2: Both GPUs (uneven load)

---

## Evaluation

After training, evaluate your policy:

```bash
python main_eval.py \
    --policy_checkpoint outputs/training_2gpu_<timestamp>/policy_final.pt \
    --reward_model_path outputs/training_2gpu_<timestamp>/reward_model_final \
    --output_file results.json
```

---

## Advanced: Using Better Models

If you have access to better base models:

### Qwen3 Series (Recommended)
```yaml
# Larger generator for better reasoning
qwen3:
  model_name: "Qwen/Qwen3-14B"  # Needs ~28GB, may need quantization

# Same reward model
reward_model:
  model_name: "Qwen/Qwen3-4B"
```

### With 8-bit Quantization
```yaml
qwen3:
  model_name: "Qwen/Qwen3-14B"
  load_in_8bit: true  # Fits in ~18GB
```

---

## FAQ

**Q: Can I use a single GPU?**
A: Yes, but you'll need to reduce model sizes or use 8-bit quantization. Both models will share the same GPU.

**Q: Can I use more than 2 GPUs?**
A: The current script uses 2 GPUs. For more GPUs, you'd need to implement model parallelism or data parallelism.

**Q: How much disk space do I need?**
A: ~50GB for models + checkpoints + dataset

**Q: Can I use different models for generator vs reward?**
A: Yes! The reward model doesn't need to be from the same family. You could use:
- Generator: Qwen3-8B
- Reward: Llama-3-8B or any other model

**Q: Do I need to train on GSM8K specifically?**
A: No! You can adapt this to any reasoning task. Just modify the data loader.

---

## Citation

If you use this code, please cite the PGTS paper:
```
@article{pgts2024,
  title={Policy Gradient Token Search for Mathematical Reasoning},
  author={...},
  year={2024}
}
```
