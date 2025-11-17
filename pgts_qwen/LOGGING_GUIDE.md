# Reward Model Training Logging Guide

## Overview

The reward model training now includes comprehensive logging and evaluation metrics to help you monitor convergence and model performance.

## What's Been Added

### 1. **Custom Callbacks** (`training/reward_callbacks.py`)

Three new components for detailed logging:

#### `RewardMetricsCallback`
- Logs training loss at every logging step (default: every 50 steps)
- Logs evaluation metrics every evaluation step (default: every 250 steps)
- Saves metrics history to JSON file
- Writes formatted metrics to dedicated log file

#### `PredictionLoggingCallback`
- Logs sample predictions during evaluation
- Shows ground truth vs predicted labels
- Helps visualize model learning progress

#### `create_compute_metrics_fn`
- Computes comprehensive evaluation metrics:
  - **Accuracy**: Overall correctness
  - **Precision**: Of positive predictions, how many are correct
  - **Recall**: Of actual positives, how many we caught
  - **F1 Score**: Harmonic mean of precision and recall
  - **Specificity**: True negative rate
  - **Confusion Matrix**: TP, TN, FP, FN counts
  - **Mean Confidence**: Average prediction confidence
- Logs sample predictions with confidence scores

### 2. **Validation Dataset Split**

The training data is now split 90/10 into train/validation sets:
- Training set: Used for model updates
- Validation set: Used for unbiased evaluation during training

### 3. **Enhanced Training Arguments**

Added evaluation and logging configuration:
- `evaluation_strategy="steps"`: Evaluate periodically during training
- `eval_steps=250`: Run evaluation every 250 training steps
- `load_best_model_at_end=True`: Load the best checkpoint at the end
- `metric_for_best_model="eval_f1"`: Use F1 score to select best model
- `logging_first_step=True`: Log immediately to verify setup

## Output Files

When you run training, you'll get these log files in `outputs/reward_model/`:

### 1. `metrics.log`
Formatted log file with timestamped entries:
```
[Step   250 | Epoch 0.50] Loss: 0.4523 LR: 1.80e-05 | Eval Loss: 0.4201 Acc: 0.8234 F1: 0.8156
[Step   500 | Epoch 1.00] Loss: 0.3891 LR: 1.60e-05 | Eval Loss: 0.3756 Acc: 0.8512 F1: 0.8445
```

### 2. `metrics_history.json`
Complete metrics history in JSON format:
```json
[
  {
    "step": 250,
    "epoch": 0.5,
    "timestamp": "2025-11-15T10:30:45.123456",
    "train_loss": 0.4523,
    "learning_rate": 1.8e-05,
    "eval_loss": 0.4201,
    "eval_accuracy": 0.8234,
    "eval_f1": 0.8156,
    "eval_precision": 0.8189,
    "eval_recall": 0.8123
  },
  ...
]
```

### 3. `predictions.log`
Ground truth vs predicted labels with confidence scores:
```
================================================================================
Evaluation at Step 250 (Epoch 0.50)
Timestamp: 2025-11-15T10:30:45
================================================================================

Overall Metrics:
  Accuracy:   0.8234
  Precision:  0.8189
  Recall:     0.8123
  F1 Score:   0.8156
  Specificity: 0.8345

Confusion Matrix:
  TN:  4523  FP:   892
  FN:  1012  TP:  4373

Sample Predictions (20 random samples):
  Idx |   GT | Pred |   Conf | Correct
---------------------------------------------
   42 |    1 |    1 | 0.9234 |       ✓
  137 |    0 |    0 | 0.8756 |       ✓
  289 |    1 |    0 | 0.5432 |       ✗
  ...
```

### 4. `prediction_samples.json`
Stored prediction samples for analysis

## Monitoring Convergence

### What to Look For

1. **Training Loss** should steadily decrease
   - If it plateaus early, you may need more training
   - If it fluctuates wildly, reduce learning rate

2. **Validation Loss** should track training loss
   - If val_loss >> train_loss, you're overfitting
   - Consider early stopping or regularization

3. **Accuracy** should increase over time
   - Target: 80%+ for good step classification
   - If stuck below 70%, check data quality

4. **F1 Score** balances precision and recall
   - Better metric than accuracy for imbalanced data
   - The model saves checkpoints with best F1 score

5. **Confusion Matrix** shows error patterns
   - High FP (False Positives): Model too optimistic
   - High FN (False Negatives): Model too conservative

### Example of Good Convergence

```
Step 50:   Loss: 0.6234, Acc: 0.6523, F1: 0.6401
Step 100:  Loss: 0.5512, Acc: 0.7234, F1: 0.7156
Step 150:  Loss: 0.4891, Acc: 0.7656, F1: 0.7589
Step 200:  Loss: 0.4401, Acc: 0.7987, F1: 0.7912
Step 250:  Loss: 0.4123, Acc: 0.8234, F1: 0.8156  ← Converging nicely
```

### Signs of Problems

**Overfitting:**
```
Step 500: Train Loss: 0.2134, Val Loss: 0.4567  ← Big gap!
```

**Underfitting:**
```
Step 2000: Train Loss: 0.5234, Val Loss: 0.5189  ← Both high
```

**Class Imbalance Issues:**
```
Precision: 0.95, Recall: 0.45  ← Model biased to negative class
```

## Configuration

You can adjust logging frequency in `train_optimized.py`:

```python
# Evaluation frequency
eval_steps=250,  # Run evaluation every N steps

# Logging frequency
logging_steps=50,  # Log training loss every N steps

# Sample logging
num_samples_to_log=20,  # Number of predictions to log
```

## Usage

Just run your normal training command:

```bash
python train_optimized.py
```

Or with custom config:

```bash
python train_optimized.py \
  --model_config config/model_config_2gpu.yaml \
  --train_config config/training_config_2gpu.yaml \
  --output_dir outputs/training_run_1
```

## Viewing Logs During Training

### Tail the metrics log:
```bash
tail -f outputs/reward_model/metrics.log
```

### Monitor predictions:
```bash
tail -f outputs/reward_model/predictions.log
```

### View metrics history:
```bash
cat outputs/reward_model/metrics_history.json | jq '.[-10:]'  # Last 10 entries
```

## Advanced: TensorBoard Support

To enable TensorBoard visualization, change in `train_optimized.py`:

```python
report_to='tensorboard',  # Instead of 'none'
```

Then view with:
```bash
tensorboard --logdir outputs/reward_model/logs
```

## Troubleshooting

### Import Error
If you get `ModuleNotFoundError: No module named 'sklearn'`:
```bash
pip install scikit-learn
```

### Memory Issues
If evaluation causes OOM, reduce eval batch size in `train_optimized.py`:
```python
per_device_eval_batch_size=4,  # Smaller than training batch
```

### Too Much Logging
If logs are too verbose, increase logging intervals:
```python
logging_steps=100,  # Instead of 50
eval_steps=500,     # Instead of 250
```

## Summary

You now have:
- ✅ Training loss logged every 50 steps
- ✅ Validation metrics every 250 steps
- ✅ Ground truth vs predicted labels in logs
- ✅ Comprehensive metrics (accuracy, precision, recall, F1)
- ✅ Best model selection based on F1 score
- ✅ JSON files for programmatic analysis
- ✅ Human-readable log files for monitoring

Monitor these files during training to ensure your reward model is learning properly!

## Dataset Size Recommendations

### Current Configuration: 200k Examples

Training on 200k examples (180k train, 20k val) provides:
- **~33,750 gradient updates** over 3 epochs
- **Better domain transfer** from GPT-4 (PRM800K) to Qwen3B
- **More robust scoring** of reasoning steps
- **Training time**: ~8-12 hours on single 24GB GPU

### Performance Targets

For good downstream GSM8k performance, your PRM should achieve:
- **Validation Accuracy**: ≥85% (step-level correctness)
- **F1 Score**: ≥0.83 (balanced precision/recall)
- **Train-Val Gap**: <0.05 (minimal overfitting)

If your metrics fall short:
- Accuracy <80%: Increase to 400k examples
- Large train-val gap: Add more regularization (weight_decay=0.02)
- Underfit (both losses high): Increase model capacity or data
