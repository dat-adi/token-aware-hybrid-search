"""
Test script to verify the reward model callbacks work correctly.
"""
import numpy as np
import tempfile
import os
from pathlib import Path
from training.reward_callbacks import create_compute_metrics_fn, RewardMetricsCallback, PredictionLoggingCallback

def test_compute_metrics():
    """Test the compute_metrics function."""
    print("Testing compute_metrics function...")

    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        compute_metrics = create_compute_metrics_fn(
            output_dir=tmpdir,
            log_predictions=True,
            num_samples_to_log=10
        )

        # Create fake predictions
        # Logits shape: [batch_size, num_classes]
        logits = np.array([
            [0.2, 0.8],  # Predicts class 1 (correct)
            [0.7, 0.3],  # Predicts class 0 (correct)
            [0.9, 0.1],  # Predicts class 0 (incorrect - should be 1)
            [0.3, 0.7],  # Predicts class 1 (correct)
            [0.6, 0.4],  # Predicts class 0 (correct)
        ])

        labels = np.array([1, 0, 1, 1, 0])

        # Mock EvalPrediction
        class MockEvalPred:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids

        eval_pred = MockEvalPred(logits, labels)

        # Compute metrics
        metrics = compute_metrics(eval_pred)

        print("✓ Metrics computed successfully!")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1 Score:   {metrics['f1']:.4f}")
        print(f"  TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")

        # Check if log file was created
        pred_log = Path(tmpdir) / "predictions.log"
        if pred_log.exists():
            print(f"✓ Predictions log created: {pred_log}")
            with open(pred_log) as f:
                content = f.read()
                if "Sample Predictions" in content:
                    print("✓ Predictions logged correctly!")
        else:
            print("⚠ Predictions log not created (may require multiple calls)")

        # Verify expected accuracy
        # Predictions: [1, 0, 0, 1, 0]
        # Labels:      [1, 0, 1, 1, 0]
        # Correct:     [✓, ✓, ✗, ✓, ✓] = 4/5 = 0.8
        expected_acc = 0.8
        assert abs(metrics['accuracy'] - expected_acc) < 0.01, f"Expected accuracy {expected_acc}, got {metrics['accuracy']}"
        print(f"✓ Accuracy matches expected value: {expected_acc}")

    print("\n" + "="*60)
    print("compute_metrics test PASSED!")
    print("="*60 + "\n")


def test_callbacks():
    """Test the callback classes can be instantiated."""
    print("Testing callback instantiation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test RewardMetricsCallback
        metrics_cb = RewardMetricsCallback(
            output_dir=tmpdir,
            log_predictions_every=100,
            num_samples_to_log=5
        )
        print(f"✓ RewardMetricsCallback created")
        print(f"  Output dir: {tmpdir}")
        print(f"  Metrics log: {metrics_cb.metrics_log_path}")

        # Check that log file was created
        assert metrics_cb.metrics_log_path.exists(), "Metrics log file not created"
        print(f"✓ Metrics log file created")

        # Test PredictionLoggingCallback
        pred_cb = PredictionLoggingCallback(
            output_dir=tmpdir,
            log_every_n_steps=100,
            num_samples=10
        )
        print(f"✓ PredictionLoggingCallback created")
        print(f"  Prediction log: {pred_cb.pred_log_path}")

        # Check that log file was created
        assert pred_cb.pred_log_path.exists(), "Prediction log file not created"
        print(f"✓ Prediction log file created")

    print("\n" + "="*60)
    print("Callback instantiation test PASSED!")
    print("="*60 + "\n")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TESTING REWARD MODEL CALLBACKS")
    print("="*60 + "\n")

    try:
        test_compute_metrics()
        test_callbacks()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60 + "\n")
        print("The callbacks are ready to use in training.")
        print("Run: python train_optimized.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
