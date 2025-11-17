"""
Custom callbacks for reward model training with detailed metrics logging.
"""
import logging
import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)


class RewardMetricsCallback(TrainerCallback):
    """
    Comprehensive callback for tracking reward model training metrics.

    Features:
    - Logs loss at every logging step
    - Logs ground truth vs predicted labels
    - Computes accuracy, precision, recall, F1
    - Saves metrics to JSON file
    - Logs to separate metrics log file
    """

    def __init__(
        self,
        output_dir: str,
        log_predictions_every: int = 500,
        num_samples_to_log: int = 10
    ):
        """
        Args:
            output_dir: Directory to save metrics and logs
            log_predictions_every: Log predictions every N steps
            num_samples_to_log: Number of sample predictions to log
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_predictions_every = log_predictions_every
        self.num_samples_to_log = num_samples_to_log

        # Metrics history
        self.metrics_history = []
        self.prediction_samples = []

        # Setup dedicated metrics log file
        self.metrics_log_path = self.output_dir / "metrics.log"
        self.metrics_json_path = self.output_dir / "metrics_history.json"
        self.predictions_json_path = self.output_dir / "prediction_samples.json"

        # Create file handler for metrics logging
        self.metrics_file_handler = logging.FileHandler(self.metrics_log_path)
        self.metrics_file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        self.metrics_file_handler.setFormatter(formatter)
        logger.addHandler(self.metrics_file_handler)

        # Write header
        with open(self.metrics_log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"REWARD MODEL TRAINING METRICS LOG\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

        logger.info(f"Initialized RewardMetricsCallback - logging to {self.metrics_log_path}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called after logging - track loss and basic metrics."""
        if logs is None:
            return

        step = state.global_step
        epoch = state.epoch

        # Extract metrics
        loss = logs.get('loss')
        learning_rate = logs.get('learning_rate')
        eval_loss = logs.get('eval_loss')
        eval_accuracy = logs.get('eval_accuracy')
        eval_f1 = logs.get('eval_f1')
        eval_precision = logs.get('eval_precision')
        eval_recall = logs.get('eval_recall')

        # Store in history
        metrics_entry = {
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        }

        if loss is not None:
            metrics_entry['train_loss'] = float(loss)
        if learning_rate is not None:
            metrics_entry['learning_rate'] = float(learning_rate)
        if eval_loss is not None:
            metrics_entry['eval_loss'] = float(eval_loss)
        if eval_accuracy is not None:
            metrics_entry['eval_accuracy'] = float(eval_accuracy)
        if eval_f1 is not None:
            metrics_entry['eval_f1'] = float(eval_f1)
        if eval_precision is not None:
            metrics_entry['eval_precision'] = float(eval_precision)
        if eval_recall is not None:
            metrics_entry['eval_recall'] = float(eval_recall)

        self.metrics_history.append(metrics_entry)

        # Log to file and console
        log_msg = f"[Step {step:5d} | Epoch {epoch:.2f}]"

        if loss is not None:
            log_msg += f" Loss: {loss:.4f}"
        if learning_rate is not None:
            log_msg += f" LR: {learning_rate:.2e}"
        if eval_loss is not None:
            log_msg += f" | Eval Loss: {eval_loss:.4f}"
        if eval_accuracy is not None:
            log_msg += f" Acc: {eval_accuracy:.4f}"
        if eval_f1 is not None:
            log_msg += f" F1: {eval_f1:.4f}"

        logger.info(log_msg)

        # Save metrics history periodically
        if step % 100 == 0:
            self._save_metrics_history()

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Called after evaluation - log detailed metrics."""
        if metrics is None:
            return

        step = state.global_step
        epoch = state.epoch

        # Log detailed evaluation metrics
        eval_msg = f"\n{'='*80}\n"
        eval_msg += f"EVALUATION at Step {step} (Epoch {epoch:.2f})\n"
        eval_msg += f"{'='*80}\n"

        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                eval_msg += f"  {key:20s}: {value:.4f}\n"
            else:
                eval_msg += f"  {key:20s}: {value}\n"

        eval_msg += f"{'='*80}\n"

        logger.info(eval_msg)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called during prediction - can be used to collect prediction samples."""
        # This gets called during evaluation
        # We'll collect samples in the compute_metrics function instead
        pass

    def _save_metrics_history(self):
        """Save metrics history to JSON file."""
        try:
            with open(self.metrics_json_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics history: {e}")

    def _save_prediction_samples(self):
        """Save prediction samples to JSON file."""
        try:
            with open(self.predictions_json_path, 'w') as f:
                json.dump(self.prediction_samples, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save prediction samples: {e}")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when saving checkpoint."""
        logger.info(f"Checkpoint saved at step {state.global_step}")
        self._save_metrics_history()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at end of training."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED")
        logger.info("="*80)

        # Save final metrics
        self._save_metrics_history()

        # Log summary statistics
        if self.metrics_history:
            train_losses = [m['train_loss'] for m in self.metrics_history if 'train_loss' in m]
            eval_accs = [m['eval_accuracy'] for m in self.metrics_history if 'eval_accuracy' in m]

            if train_losses:
                logger.info(f"Final train loss: {train_losses[-1]:.4f}")
                logger.info(f"Min train loss: {min(train_losses):.4f}")
                logger.info(f"Mean train loss: {np.mean(train_losses):.4f}")

            if eval_accs:
                logger.info(f"Final eval accuracy: {eval_accs[-1]:.4f}")
                logger.info(f"Max eval accuracy: {max(eval_accs):.4f}")
                logger.info(f"Mean eval accuracy: {np.mean(eval_accs):.4f}")

        logger.info(f"Metrics saved to: {self.metrics_json_path}")
        logger.info("="*80 + "\n")


class PredictionLoggingCallback(TrainerCallback):
    """
    Callback to log ground truth vs predicted labels during training.

    Logs sample predictions to show model learning progress.
    """

    def __init__(
        self,
        output_dir: str,
        log_every_n_steps: int = 500,
        num_samples: int = 20
    ):
        """
        Args:
            output_dir: Directory to save prediction logs
            log_every_n_steps: Log predictions every N steps
            num_samples: Number of samples to log
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_every_n_steps = log_every_n_steps
        self.num_samples = num_samples

        # Setup prediction log file
        self.pred_log_path = self.output_dir / "predictions.log"

        with open(self.pred_log_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GROUND TRUTH vs PREDICTED LABELS LOG\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

        logger.info(f"Initialized PredictionLoggingCallback - logging to {self.pred_log_path}")

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Log predictions during evaluation."""
        # Predictions are logged via compute_metrics function
        # This is just for timing
        if state.global_step % self.log_every_n_steps == 0:
            with open(self.pred_log_path, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Evaluation at Step {state.global_step} (Epoch {state.epoch:.2f})\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"{'='*80}\n\n")


def create_compute_metrics_fn(
    output_dir: str,
    log_predictions: bool = True,
    num_samples_to_log: int = 20
):
    """
    Create a compute_metrics function with prediction logging.

    Args:
        output_dir: Directory to save prediction logs
        log_predictions: Whether to log sample predictions
        num_samples_to_log: Number of sample predictions to log

    Returns:
        compute_metrics function for Trainer
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pred_log_path = output_path / "predictions.log"

    # Track call count for logging samples
    call_count = {'count': 0}

    def compute_metrics(eval_pred):
        """
        Compute metrics for evaluation.

        Args:
            eval_pred: EvalPrediction object with predictions and label_ids

        Returns:
            Dictionary of metrics
        """
        logits, labels = eval_pred

        # Get predicted classes
        predictions = np.argmax(logits, axis=1)

        # Compute probabilities for additional insights
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        confidence = probs[np.arange(len(predictions)), predictions]

        # Compute metrics
        accuracy = (predictions == labels).mean()

        # Handle binary classification metrics with zero_division parameter
        precision = precision_score(labels, predictions, average='binary', zero_division=0)
        recall = recall_score(labels, predictions, average='binary', zero_division=0)
        f1 = f1_score(labels, predictions, average='binary', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Log sample predictions
        if log_predictions and call_count['count'] % 5 == 0:  # Log every 5th evaluation
            with open(pred_log_path, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Evaluation Call #{call_count['count']}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"{'='*80}\n\n")

                f.write(f"Overall Metrics:\n")
                f.write(f"  Accuracy:   {accuracy:.4f}\n")
                f.write(f"  Precision:  {precision:.4f}\n")
                f.write(f"  Recall:     {recall:.4f}\n")
                f.write(f"  F1 Score:   {f1:.4f}\n")
                f.write(f"  Specificity: {specificity:.4f}\n\n")

                f.write(f"Confusion Matrix:\n")
                f.write(f"  TN: {tn:5d}  FP: {fp:5d}\n")
                f.write(f"  FN: {fn:5d}  TP: {tp:5d}\n\n")

                # Log sample predictions
                num_to_log = min(num_samples_to_log, len(predictions))
                indices = np.random.choice(len(predictions), num_to_log, replace=False)

                f.write(f"Sample Predictions ({num_to_log} random samples):\n")
                f.write(f"{'Idx':>5s} | {'GT':>4s} | {'Pred':>4s} | {'Conf':>6s} | {'Correct':>7s}\n")
                f.write(f"{'-'*45}\n")

                for idx in sorted(indices):
                    gt = int(labels[idx])
                    pred = int(predictions[idx])
                    conf = float(confidence[idx])
                    correct = "✓" if gt == pred else "✗"

                    f.write(f"{idx:5d} | {gt:4d} | {pred:4d} | {conf:6.4f} | {correct:>7s}\n")

                f.write("\n")

        call_count['count'] += 1

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'tp': float(tp),
            'tn': float(tn),
            'fp': float(fp),
            'fn': float(fn),
            'mean_confidence': float(confidence.mean()),
        }

    return compute_metrics
