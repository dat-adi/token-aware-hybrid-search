# GSM8K-Gen-Stepwise Dataset - Quick Start

Train your Process Reward Model on high-quality, step-wise labeled data from the `ebony59/gsm8k-gen-stepwise` dataset.

## What This Does

Converts the GSM8K-Gen-Stepwise dataset (50K+ problems with multiple solutions each) into step-level training examples for your PRM.

## Three Simple Steps

### 1. Test It Works (30 seconds)

```bash
cd /home/datadi/Code/token-aware-hybrid-search/pgts_qwen
python test_stepwise_formatter.py
```

### 2. Format the Dataset (10-30 minutes)

```bash
# Full dataset
python data/gsm8k_stepwise_formatter.py --output_dir data/prm_formatted

# Or quick subset (5 minutes)
python data/gsm8k_stepwise_formatter.py --max_examples 5000 --output_dir data/prm_formatted
```

### 3. Train the PRM (2-8 hours)

```bash
# Quick test (30 min)
python train_prm_with_stepwise.py \
    --max_examples 1000 \
    --num_epochs 2 \
    --output_dir outputs/prm_stepwise_test

# Full training (4-8 hours)
python train_prm_with_stepwise.py \
    --output_dir outputs/prm_stepwise_full
```

## What You Get

```
outputs/prm_stepwise_full/
├── best_model/          # ← Use this in PGTS training
├── final_model/
└── data/
    ├── prm_train.jsonl
    └── prm_val.jsonl
```

## Use in PGTS Pipeline

```bash
# Train PGTS policy with your new PRM
python train_optimized.py \
    --skip_reward_training \
    --reward_model_path outputs/prm_stepwise_full/best_model \
    --output_dir outputs/training_with_stepwise_prm
```

## Key Advantages

- ✅ **Real LLM outputs** (not synthetic corruptions)
- ✅ **50K problems** (vs 7.5K in standard GSM8K)
- ✅ **Multiple solutions per problem** with correctness labels
- ✅ **Step-level training** for fine-grained reward modeling
- ✅ **Expected 75-85% accuracy** (Qwen3-4B)

## Files

- `data/gsm8k_stepwise_formatter.py` - Dataset formatter
- `test_stepwise_formatter.py` - Quick test
- `train_prm_with_stepwise.py` - Complete training pipeline
- `GSM8K_STEPWISE_TRAINING.md` - Full documentation

## Need Help?

See `GSM8K_STEPWISE_TRAINING.md` for:
- Detailed usage examples
- Integration options
- Troubleshooting
- Performance benchmarks
