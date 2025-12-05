"""
Optimized training script for 2x24GB GPU setup.
Better GPU memory management and placement.
"""
import torch
import yaml
import logging
import argparse
import os
from pathlib import Path

from models.qwen3_wrapper import Qwen3ReasoningGenerator
from models.reward_model import ProcessRewardModel
from models.policy_network import GPSPolicyNetwork
from data.gsm8k_loader import load_gsm8k
from data.reward_annotator import create_prm_dataset
from training.ppo_trainer import PPOTrainer, PPOConfig
from data.reward_annotator_fast import create_prm_dataset_fast

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_distributed():
    """Check GPU availability and setup."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPUs")

    if num_gpus < 2:
        logger.warning(f"This script is optimized for 2 GPUs, but only {num_gpus} found")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")

    return num_gpus


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_reward_model_optimized(
    model_config: dict,
    train_config: dict,
    gsm8k_train_data: list,
    output_dir: str,
    device: str = "cuda:0"
):
    """
    Train the Process Reward Model on single GPU.

    Phase 1 uses only GPU 0 to conserve memory.
    """
    logger.info("=== Phase 1: Training Process Reward Model ===")
    logger.info(f"Using device: {device}")

    # Initialize reasoning generator for data generation (on GPU 0)
    logger.info("Loading Qwen3 for data generation...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    reasoning_generator = Qwen3ReasoningGenerator(
        model_name=model_config['qwen3']['model_name'],
        temperature=1.0,  # High temperature for diverse incorrect samples
        use_vllm=False,  # Need hidden states for reward model
        device=device
    )

    # Generate PRM training data
    logger.info("Generating PRM training data...")
    prm_dataset = create_prm_dataset_fast(  # Changed
        gsm8k_train_data,  # No reasoning_generator needed!
        num_incorrect_per_problem=train_config['reward_training']['num_incorrect_per_problem'],
        max_examples=train_config['reward_training']['max_examples']
    )

    # Free up memory
    del reasoning_generator
    torch.cuda.empty_cache()

    # Initialize reward model
    logger.info("Initializing reward model...")
    reward_model = ProcessRewardModel(
        model_name=model_config['reward_model']['model_name'],
        num_labels=model_config['reward_model']['num_labels'],
        device=device
    )

    # Fix padding token issue
    if reward_model.tokenizer.pad_token is None:
        reward_model.tokenizer.pad_token = reward_model.tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {reward_model.tokenizer.eos_token}")

    if reward_model.model.config.pad_token_id is None:
        reward_model.model.config.pad_token_id = reward_model.tokenizer.pad_token_id
        logger.info(f"Set model pad_token_id to: {reward_model.tokenizer.pad_token_id}")

    # Prepare training data
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset

    class PRMDataset(Dataset):
        def __init__(self, examples, tokenizer):
            self.examples = examples
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            example = self.examples[idx]
            text = reward_model.format_input(
                example['problem'],
                example['reasoning_path']
            )

            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(example['label'], dtype=torch.long)
            }

    # Create dataset
    train_dataset = PRMDataset(prm_dataset, reward_model.tokenizer)

    # Training arguments with memory optimization
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'reward_model'),
        num_train_epochs=train_config['reward_training']['num_epochs'],
        per_device_train_batch_size=train_config['reward_training']['batch_size'],
        gradient_accumulation_steps=train_config['reward_training'].get('gradient_accumulation_steps', 1),
        learning_rate=float(train_config['reward_training']['learning_rate']),
        warmup_ratio=train_config['reward_training']['warmup_ratio'],
        weight_decay=train_config['reward_training']['weight_decay'],
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,  # Only keep 2 checkpoints to save disk space
        fp16=False,  # Mixed precision
        gradient_checkpointing=True,  # Memory efficient
        dataloader_num_workers=4,
        report_to='none',
        remove_unused_columns=False
    )

    # Train
    trainer = Trainer(
        model=reward_model.model,
        args=training_args,
        train_dataset=train_dataset
    )

    logger.info("Training reward model...")
    trainer.train()

    # Save model
    reward_model_path = os.path.join(output_dir, 'reward_model_final')
    reward_model.save_pretrained(reward_model_path)
    logger.info(f"Reward model saved to {reward_model_path}")

    return reward_model_path


def train_policy_optimized(
    model_config: dict,
    train_config: dict,
    gsm8k_train_data: list,
    gsm8k_val_data: list,
    reward_model_path: str,
    output_dir: str,
    num_gpus: int = 2
):
    """
    Train the policy network with PPO using multiple GPUs.

    GPU placement:
    - GPU 0: Qwen3-8B reasoning generator
    - GPU 1: Qwen3-4B reward model + GPS policy network
    """
    logger.info("=== Phase 2: Training Policy Network with PPO ===")

    # Load models with explicit GPU placement
    logger.info("Loading models with optimized GPU placement...")

    # GPU 0: Reasoning generator
    logger.info("Loading reasoning generator on GPU 0...")
    with torch.cuda.device(0):
        reasoning_generator = Qwen3ReasoningGenerator(
            model_name=model_config['qwen3']['model_name'],
            temperature=model_config['qwen3']['temperature'],
            use_vllm=False,  # Need hidden states
            device="cuda:0"
        )

    # GPU 1 (or GPU 0 if only 1 GPU): Reward model + Policy
    reward_device = "cuda:1" if num_gpus >= 2 else "cuda:0"
    logger.info(f"Loading reward model on {reward_device}...")

    with torch.cuda.device(reward_device):
        reward_model = ProcessRewardModel.from_pretrained(
            reward_model_path,
            device=reward_device
        )

    # Initialize policy network (on same device as reward model)
    logger.info(f"Initializing policy network on {reward_device}...")
    policy_network = GPSPolicyNetwork(
        input_dim=reasoning_generator.get_hidden_dim(),
        hidden_dim=model_config['policy_network']['hidden_dim'],
        num_layers=model_config['policy_network']['num_layers'],
        num_heads=model_config['policy_network']['num_heads'],
        dropout=model_config['policy_network']['dropout'],
        activation=model_config['policy_network']['activation']
    ).to(reward_device)

    # PPO configuration
    ppo_config = PPOConfig(
        learning_rate=train_config['policy_training']['learning_rate'],
        batch_size=train_config['policy_training']['batch_size'],
        ppo_epochs=train_config['policy_training']['ppo_epochs'],
        clip_epsilon=train_config['policy_training']['clip_epsilon'],
        value_coeff=train_config['policy_training']['value_coeff'],
        entropy_coeff=train_config['policy_training']['entropy_coeff'],
        max_grad_norm=train_config['policy_training']['max_grad_norm'],
        final_reward_correct=train_config['policy_training']['final_reward_correct'],
        final_reward_incorrect=train_config['policy_training']['final_reward_incorrect'],
        step_penalty=train_config['policy_training']['step_penalty'],
        gae_lambda=train_config['policy_training']['gae_lambda'],
        discount_gamma=train_config['policy_training']['discount_gamma']
    )

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        policy_network=policy_network,
        reasoning_generator=reasoning_generator,
        reward_model=reward_model,
        config=ppo_config,
        device=reward_device
    )

    # Prepare training data
    train_problems = [ex['problem'] for ex in gsm8k_train_data]
    train_answers = [ex['answer'] for ex in gsm8k_train_data]

    # Logging callback
    def log_callback(iteration, metrics):
        logger.info(f"Iteration {iteration + 1}: {metrics}")

        # Save checkpoint
        if (iteration + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                output_dir,
                'policy_checkpoints',
                f'checkpoint_{iteration + 1}.pt'
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            ppo_trainer.save_checkpoint(checkpoint_path)

            # Log memory usage
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                logger.info(f"GPU {i} Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    # Train
    logger.info("Starting PPO training...")
    ppo_trainer.train(
        train_problems=train_problems,
        train_answers=train_answers,
        num_iterations=train_config['policy_training']['num_iterations'],
        problems_per_iteration=train_config['policy_training']['problems_per_iteration'],
        log_callback=log_callback
    )

    # Save final model
    final_path = os.path.join(output_dir, 'policy_final.pt')
    ppo_trainer.save_checkpoint(final_path)
    logger.info(f"Final policy saved to {final_path}")

    # Save action statistics
    action_stats_path = os.path.join(output_dir, 'action_statistics.json')
    ppo_trainer.save_action_statistics(action_stats_path)
    logger.info(f"Action statistics saved to {action_stats_path}")

    # Generate visualizations
    try:
        from utils.visualize_actions import generate_all_plots
        viz_dir = os.path.join(output_dir, 'action_visualizations')
        logger.info("Generating action statistics visualizations...")
        generate_all_plots(
            ppo_trainer.get_action_history(),
            viz_dir,
            prefix="action_stats"
        )
        logger.info(f"Action visualizations saved to {viz_dir}")
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PGTS for GSM8k (2x24GB optimized)')
    parser.add_argument('--model_config', type=str, default='config/model_config_2gpu.yaml',
                       help='Path to model config')
    parser.add_argument('--train_config', type=str, default='config/training_config_2gpu.yaml',
                       help='Path to training config')
    parser.add_argument('--output_dir', type=str, default='outputs/training_2gpu',
                       help='Output directory')
    parser.add_argument('--skip_reward_training', action='store_true',
                       help='Skip reward model training')
    parser.add_argument('--reward_model_path', type=str, default=None,
                       help='Path to pretrained reward model')

    args = parser.parse_args()

    # Setup GPUs
    num_gpus = setup_distributed()

    # Load configurations
    logger.info("Loading configurations...")
    model_config = load_config(args.model_config)
    train_config = load_config(args.train_config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load GSM8k dataset
    logger.info("Loading GSM8k dataset...")
    train_data, val_data, test_data = load_gsm8k()

    # Phase 1: Train reward model
    if args.skip_reward_training and args.reward_model_path:
        logger.info(f"Skipping reward model training, using {args.reward_model_path}")
        reward_model_path = args.reward_model_path
    else:
        reward_model_path = train_reward_model_optimized(
            model_config,
            train_config,
            train_data,
            args.output_dir,
            device="cuda:0"
        )

    # Clear GPU 0 before Phase 2
    torch.cuda.empty_cache()

    # Phase 2: Train policy
    train_policy_optimized(
        model_config,
        train_config,
        train_data,
        val_data,
        reward_model_path,
        args.output_dir,
        num_gpus=num_gpus
    )

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
