"""
Train Process Reward Model using GSM8K-Gen-Stepwise dataset.

This script:
1. Loads and formats the stepwise dataset
2. Trains the PRM model on step-level correctness
3. Saves the trained model
"""
import os
import torch
import logging
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from data.gsm8k_stepwise_formatter import format_gsm8k_stepwise
from models.reward_model import ProcessRewardModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class PRMDataset(Dataset):
    """
    PyTorch Dataset for Process Reward Model training.
    """

    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        """
        Initialize dataset.

        Args:
            examples: Formatted examples from GSM8K-Gen-Stepwise
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Get training example."""
        example = self.examples[idx]

        # Format input text
        problem = example['problem']
        reasoning_path = example['reasoning_path']

        # Create formatted text
        steps_text = "\n".join(
            f"Step {i+1}: {step}"
            for i, step in enumerate(reasoning_path)
        )

        text = f"""Problem: {problem}

Solution:
{steps_text}"""

        # Tokenize
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(example['label'], dtype=torch.long)
        }


def train_prm(
    model_name: str = "Qwen/Qwen3-1.5B",
    dataset_name: str = "ebony59/gsm8k-gen-stepwise",
    output_dir: str = "outputs/prm_stepwise",
    max_examples: int = None,
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_steps: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_every: int = 5000
):
    """
    Train Process Reward Model on GSM8K-Gen-Stepwise dataset.

    Args:
        model_name: Base model for PRM
        dataset_name: HuggingFace dataset name
        output_dir: Directory to save trained model
        max_examples: Maximum examples to use (None = all)
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Warmup steps for scheduler
        device: Device to train on
        save_every: Save checkpoint every N steps
    """
    logger.info("=" * 60)
    logger.info("Training Process Reward Model")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load and format dataset
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Loading and formatting dataset")
    logger.info("=" * 60)

    train_examples, val_examples = format_gsm8k_stepwise(
        dataset_name=dataset_name,
        max_examples=max_examples,
        balance_dataset=True,
        val_ratio=0.1,
        output_dir=os.path.join(output_dir, 'data')
    )

    logger.info(f"Train examples: {len(train_examples)}")
    logger.info(f"Val examples: {len(val_examples)}")

    # Step 2: Initialize model
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Initializing Process Reward Model")
    logger.info("=" * 60)

    prm = ProcessRewardModel(
        model_name=model_name,
        num_labels=2,
        device=device,
        torch_dtype=torch.bfloat16
    )

    # Step 3: Create datasets and dataloaders
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Creating dataloaders")
    logger.info("=" * 60)

    train_dataset = PRMDataset(train_examples, prm.tokenizer)
    val_dataset = PRMDataset(val_examples, prm.tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Step 4: Setup optimizer and scheduler
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Setting up optimizer")
    logger.info("=" * 60)

    optimizer = AdamW(prm.model.parameters(), lr=learning_rate)

    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")

    # Step 5: Training loop
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Training")
    logger.info("=" * 60)

    global_step = 0
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 60)

        # Training
        prm.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = prm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track metrics
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })

            # Save checkpoint
            if global_step % save_every == 0:
                checkpoint_dir = os.path.join(output_dir, f'checkpoint-{global_step}')
                prm.save_pretrained(checkpoint_dir)
                logger.info(f"\nSaved checkpoint to {checkpoint_dir}")

        # Epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Train Accuracy: {train_accuracy:.2f}%")

        # Validation
        logger.info("\nRunning validation...")
        prm.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = prm.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_dir = os.path.join(output_dir, 'best_model')
            prm.save_pretrained(best_model_dir)
            logger.info(f"\n✓ New best model! Accuracy: {val_accuracy:.2f}%")
            logger.info(f"  Saved to {best_model_dir}")

    # Step 6: Save final model
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Saving final model")
    logger.info("=" * 60)

    final_model_dir = os.path.join(output_dir, 'final_model')
    prm.save_pretrained(final_model_dir)
    logger.info(f"Final model saved to {final_model_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    logger.info(f"Best model: {os.path.join(output_dir, 'best_model')}")
    logger.info(f"Final model: {final_model_dir}")

    return prm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train PRM on GSM8K-Gen-Stepwise dataset"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen3-1.5B',
        help='Base model name'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='ebony59/gsm8k-gen-stepwise',
        help='Dataset name'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/prm_stepwise',
        help='Output directory'
    )
    parser.add_argument(
        '--max_examples',
        type=int,
        default=None,
        help='Maximum examples (default: all)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
        help='Number of epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device'
    )

    args = parser.parse_args()

    # Train model
    train_prm(
        model_name=args.model_name,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
