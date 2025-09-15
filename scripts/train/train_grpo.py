import os
import logging
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from reward import bge_reward_func


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training Script with Reward Function")

    # Model & Data
    parser.add_argument("--training_model_path", type=str, help="Training model directory")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training parquet file")

    # Training hyperparameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_generations", type=int, default=16, help="Number of generations per prompt")
    parser.add_argument("--max_seq_length", type=int, default=8000, help="Max prompt length")
    parser.add_argument("--max_completion_length", type=int, default=500, help="Max completion length")
    parser.add_argument("--max_steps", type=int, default=150, help="Total training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.008, help="Beta parameter for GRPO")

    # Misc
    parser.add_argument("--seed", type=str, default="random", help="Random seed")
    parser.add_argument("--output_dir", type=str, default="trained_models", help="Output directory")
    parser.add_argument("--log_file", type=str, default="training.log", help="Log file path")

    return parser.parse_args()


def setup_logging(log_file: str):
    """Setup logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )


def load_training_model_and_tokenizer(training_model_path: str):
    """Load causal LM model and tokenizer from a given path."""
    logging.info(f"Loading model and tokenizer from {training_model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(training_model_path)
    tokenizer = AutoTokenizer.from_pretrained(training_model_path)
    return model, tokenizer


def load_training_dataset(data_path: str, seed: int = 5202):
    """Load and shuffle dataset from parquet file."""
    logging.info(f"Loading dataset from {data_path} ...")
    dataset = load_dataset("parquet", data_files=data_path+'/train.parquet')
    dataset = dataset["train"].shuffle(seed=seed)
    logging.info(f"Dataset loaded with {len(dataset)} samples.")
    return dataset


def create_training_args(args) -> GRPOConfig:
    """Create GRPO training configuration from argparse args."""
    output_dir = os.path.join(
        args.output_dir,
        f"{args.train_data_path.split('/')[-1]}_seed{args.seed}_maxcomp{args.max_completion_length}_numgen{args.num_generations}"
    )
    logging.info(f"Training outputs will be saved to: {output_dir}")
    return GRPOConfig(
        use_vllm=True,
        learning_rate=args.learning_rate,
        beta=args.beta,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        logging_steps=1,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_seq_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        max_grad_norm=0.1,
        report_to=None,
        output_dir=output_dir,
    )


def main():
    args = parse_args()
    setup_logging(args.log_file)

    logging.info("======== Training Started ========")
    logging.info(f"Using seed: {args.seed}")
    logging.info(f"GPU count available: {torch.cuda.device_count()}")

    # Load model & tokenizer
    model, tokenizer = load_training_model_and_tokenizer(args.training_model_path)

    # Load dataset
    train_dataset = load_training_dataset(args.train_data_path)

    # Create training arguments
    training_args = create_training_args(args)

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[bge_reward_func],
        args=training_args,
        train_dataset=train_dataset,
    )

    logging.info("Starting training loop ...")
    trainer.train(resume_from_checkpoint=False)
    logging.info("======== Training Finished ========")


if __name__ == "__main__":
    main()
