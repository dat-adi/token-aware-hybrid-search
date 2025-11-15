# Training PRM with GSM8K-Gen-Stepwise Dataset

This guide explains how to use the `ebony59/gsm8k-gen-stepwise` dataset to train the Process Reward Model (PRM).

## Overview

The GSM8K-Gen-Stepwise dataset contains 50,916 math problems with multiple solution attempts for each problem, where each solution is labeled as correct or incorrect. This is ideal for training a Process Reward Model to evaluate reasoning steps.

### Dataset Structure

- **Source**: https://huggingface.co/datasets/ebony59/gsm8k-gen-stepwise
- **Format**: Parquet dataset with 50,916 rows
- **Fields**:
  - `prompt`: Math problem text with special tokens
  - `completions`: List of solution attempts (1-44 per problem)
  - `labels`: Boolean array indicating correctness of each completion

### Formatted Output

The formatter converts this to step-wise examples:

```python
{
    'problem': str,              # Clean problem text
    'step': str,                 # Individual reasoning step
    'previous_steps': List[str], # Steps before this one
    'reasoning_path': List[str], # Full path up to this step
    'label': int,                # 0 (incorrect) or 1 (correct)
    'step_index': int,           # Position of this step
    'total_steps': int           # Total steps in this path
}
```

## Quick Start

### 1. Test the Formatter (Quick Validation)

Test with a small sample to ensure everything works:

```bash
cd /home/datadi/Code/token-aware-hybrid-search/pgts_qwen
python test_stepwise_formatter.py
```

This will:
- Load 10 examples from the dataset
- Format them into step-wise training examples
- Show statistics and sample outputs
- Verify train/val split functionality

**Expected Output:**
```
✓ Generated ~200-500 step-wise examples from 10 problems
Dataset statistics:
  - Total examples: ~400
  - Correct: ~200 (50.0%)
  - Incorrect: ~200 (50.0%)
```

### 2. Format the Full Dataset

Format the complete dataset and save to JSONL files:

```bash
# Format entire dataset (~50K problems)
python data/gsm8k_stepwise_formatter.py \
    --output_dir data/prm_formatted

# Or limit to first 10,000 problems for faster processing
python data/gsm8k_stepwise_formatter.py \
    --max_examples 10000 \
    --output_dir data/prm_formatted
```

**Options:**
- `--dataset`: Dataset name (default: `ebony59/gsm8k-gen-stepwise`)
- `--max_examples`: Limit number of problems (default: all)
- `--no_balance`: Don't balance correct/incorrect examples
- `--val_ratio`: Validation split ratio (default: 0.1)
- `--output_dir`: Output directory for JSONL files
- `--seed`: Random seed for reproducibility

**Output Files:**
- `data/prm_formatted/prm_train.jsonl`: Training examples
- `data/prm_formatted/prm_val.jsonl`: Validation examples

### 3. Train the PRM Model

Train the Process Reward Model using the formatted data:

```bash
# Train with default settings (Qwen3-1.5B)
python train_prm_with_stepwise.py \
    --output_dir outputs/prm_stepwise

# Train with larger model (Qwen3-4B)
python train_prm_with_stepwise.py \
    --model_name Qwen/Qwen3-4B \
    --output_dir outputs/prm_stepwise_4b \
    --batch_size 8

# Quick training run (limited examples)
python train_prm_with_stepwise.py \
    --max_examples 5000 \
    --num_epochs 2 \
    --output_dir outputs/prm_stepwise_quick
```

**Training Options:**
- `--model_name`: Base model (default: `Qwen/Qwen3-1.5B`)
- `--dataset`: HuggingFace dataset name
- `--output_dir`: Directory for saved models
- `--max_examples`: Limit examples for faster training
- `--batch_size`: Training batch size (default: 16)
- `--num_epochs`: Number of epochs (default: 3)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--device`: Device (default: auto-detect)

**Output Structure:**
```
outputs/prm_stepwise/
├── data/
│   ├── prm_train.jsonl       # Formatted training data
│   └── prm_val.jsonl         # Formatted validation data
├── checkpoint-5000/          # Periodic checkpoints
├── checkpoint-10000/
├── best_model/               # Best model (highest val accuracy)
└── final_model/              # Final model after all epochs
```

## Usage Examples

### Example 1: Quick Test Run

Test the full pipeline with a small dataset:

```bash
cd pgts_qwen

# 1. Test formatter
python test_stepwise_formatter.py

# 2. Quick training run (100 problems, 2 epochs)
python train_prm_with_stepwise.py \
    --max_examples 100 \
    --num_epochs 2 \
    --batch_size 8 \
    --output_dir outputs/prm_test
```

### Example 2: Full Production Training

Train on the complete dataset with Qwen3-4B:

```bash
cd pgts_qwen

# Format full dataset
python data/gsm8k_stepwise_formatter.py \
    --output_dir data/prm_formatted_full

# Train PRM
python train_prm_with_stepwise.py \
    --model_name Qwen/Qwen3-4B \
    --batch_size 16 \
    --num_epochs 3 \
    --output_dir outputs/prm_stepwise_full
```

**Expected Time:**
- Dataset formatting: ~10-30 minutes (full dataset)
- Training: ~4-8 hours (depends on GPU, batch size, model size)

### Example 3: Use Formatted Data in Existing Pipeline

Load the formatted JSONL files in your own training script:

```python
import json
from typing import List, Dict

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file."""
    examples = []
    with open(filepath, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

# Load formatted data
train_examples = load_jsonl('data/prm_formatted/prm_train.jsonl')
val_examples = load_jsonl('data/prm_formatted/prm_val.jsonl')

# Use in your training loop
for example in train_examples:
    problem = example['problem']
    reasoning_path = example['reasoning_path']
    label = example['label']  # 0 or 1

    # Your training code here...
```

## Integration with Existing PGTS Pipeline

### Option 1: Replace Synthetic Data Generation

Use the stepwise dataset instead of synthetic corruption:

```python
# In train_optimized.py or main_train.py

# OLD (synthetic):
# from data.reward_annotator_fast import create_prm_dataset_fast
# prm_data = create_prm_dataset_fast(gsm8k_train, ...)

# NEW (stepwise dataset):
from data.gsm8k_stepwise_formatter import format_gsm8k_stepwise
train_data, val_data = format_gsm8k_stepwise(
    max_examples=50000,
    balance_dataset=True
)
```

### Option 2: Combine Both Datasets

Use both synthetic and stepwise data for more diversity:

```python
from data.reward_annotator_fast import create_prm_dataset_fast
from data.gsm8k_stepwise_formatter import format_gsm8k_stepwise

# Synthetic data
synthetic_data = create_prm_dataset_fast(gsm8k_train, max_examples=25000)

# Stepwise data
stepwise_train, stepwise_val = format_gsm8k_stepwise(max_examples=25000)

# Combine
combined_train = synthetic_data + stepwise_train
random.shuffle(combined_train)
```

### Option 3: Pre-train on Stepwise, Fine-tune on Synthetic

1. First, train on stepwise dataset:
```bash
python train_prm_with_stepwise.py \
    --output_dir outputs/prm_stepwise_pretrain
```

2. Then, use the trained model in PGTS training:
```bash
python train_optimized.py \
    --skip_reward_training \
    --reward_model_path outputs/prm_stepwise_pretrain/best_model
```

## Performance Expectations

### Dataset Statistics (Full Dataset)

- **Total problems**: ~50,916
- **Total completions**: ~200,000-500,000 (variable per problem)
- **After step-wise expansion**: ~2-5 million step-level examples
- **After balancing**: ~1-2 million examples (50/50 correct/incorrect)

### Training Metrics

Expected validation accuracy after training:

- **Qwen3-1.5B**: 70-80% accuracy
- **Qwen3-4B**: 75-85% accuracy
- **Qwen3-8B**: 80-90% accuracy

### Resource Requirements

| Model | VRAM | Training Time | Batch Size |
|-------|------|---------------|------------|
| Qwen3-1.5B | ~6-8 GB | 2-4 hours | 16-32 |
| Qwen3-4B | ~10-14 GB | 4-8 hours | 8-16 |
| Qwen3-8B | ~18-24 GB | 8-12 hours | 4-8 |

## Advantages Over Synthetic Data

1. **Real Model Outputs**: Uses actual LLM-generated reasoning paths, not synthetic corruptions
2. **Natural Error Patterns**: Learns from realistic mistakes models make
3. **Higher Quality**: Multiple solutions per problem with explicit correctness labels
4. **Larger Scale**: 50K problems vs typical 7.5K in GSM8K
5. **Better Generalization**: Diverse reasoning approaches for same problems

## Troubleshooting

### Issue: Dataset Download Slow

The dataset is ~300MB. If download is slow:

```python
from datasets import load_dataset

# Download with caching
dataset = load_dataset("ebony59/gsm8k-gen-stepwise", cache_dir="./cache")
```

### Issue: Out of Memory During Formatting

Reduce batch processing:

```bash
python data/gsm8k_stepwise_formatter.py \
    --max_examples 10000  # Process in chunks
```

### Issue: Out of Memory During Training

Reduce batch size and use gradient accumulation:

```bash
python train_prm_with_stepwise.py \
    --batch_size 4 \
    --model_name Qwen/Qwen3-1.5B  # Use smaller model
```

### Issue: Training Too Slow

Use mixed precision training (already enabled via `torch.bfloat16` in the code) or use fewer examples:

```bash
python train_prm_with_stepwise.py \
    --max_examples 20000 \
    --num_epochs 2
```

## Files Created

1. **`data/gsm8k_stepwise_formatter.py`**: Main formatter class
2. **`test_stepwise_formatter.py`**: Quick test script
3. **`train_prm_with_stepwise.py`**: Complete training pipeline
4. **`GSM8K_STEPWISE_TRAINING.md`**: This documentation

## Next Steps

After training the PRM on stepwise data:

1. **Evaluate**: Test the model on held-out validation set
2. **Integrate**: Use in PGTS policy training pipeline
3. **Compare**: Benchmark against synthetic-trained PRM
4. **Fine-tune**: Further train on domain-specific problems if needed

## References

- **Dataset**: https://huggingface.co/datasets/ebony59/gsm8k-gen-stepwise
- **GSM8K Original**: https://huggingface.co/datasets/gsm8k
- **Process Reward Models**: See PGTS paper (docs/pgts.pdf)
