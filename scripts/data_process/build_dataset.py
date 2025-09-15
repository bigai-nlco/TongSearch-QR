#!/usr/bin/env python3
"""
Convert sampled question-answer JSON into HuggingFace Dataset parquet format.
Supports optional <think>/<answer> tags and shuffling.
Usage example:
python ./scripts/data_process/build_dataset.py \
         --input data_train/raw_data/V1-LRM.json \
		 --output-dir data_train/built_dataset \
		 --output-name V1-LRM_no_think
"""

import argparse
import json
import logging
import random
from pathlib import Path

from datasets import Dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process sampled question-answer JSON and save as HuggingFace dataset."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output-dir", type=str, default="data/built_dataset",
                        help="Directory to save the output parquet")
    parser.add_argument("--output-name", type=str, default="stack_exchange_r1_rl",
                        help="Base name for output file (suffix will be added automatically)")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--use-think", action="store_true", help="Enable <think>/<answer> tags in system prompt")
    parser.add_argument("--max-length", type=int, default=16000, help="Maximum allowed question length")
    return parser.parse_args()


def build_system_prompt(use_think: bool) -> str:
    """Return system prompt depending on whether <think>/<answer> tags are enabled."""
    if not use_think:
        return ""
    return (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    )


def process_data(sampled_questions: dict, use_think: bool, max_length: int) -> tuple[list, int]:
    """
    Process sampled questions into dataset items.
    Returns:
        (list of items, max question length)
    """
    system_prompt = build_system_prompt(use_think)
    processed_data = []
    max_len = 0

    for question, answer in sampled_questions.items():
        # Skip image-based questions
        if ".png" in question:
            continue
        # Skip overly long questions
        if len(question) > max_length:
            continue

        # Ensure answer is list
        if not isinstance(answer, list):
            answer = [answer]

        # Messages for chat-style dataset
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        # Track max length
        max_len = max(max_len, len(question))

        # Store solution metadata as JSON string
        solution = {
            "query": question,
            "pos_docs_list": answer,
            "think_flag": use_think
        }

        item_dict = {
            "prompt": messages,
            "answer": json.dumps(solution, ensure_ascii=False)
        }
        processed_data.append(item_dict)

    return processed_data, max_len


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Load input JSON
    logging.info("Loading input JSON: %s", args.input)
    with open(args.input, "r", encoding="utf-8") as f:
        sampled_questions = json.load(f)

    # Process data
    logging.info("Processing data...")
    dataset_items, max_len = process_data(sampled_questions, args.use_think, args.max_length)
    random.seed(args.seed)
    random.shuffle(dataset_items)
    dataset = Dataset.from_list(dataset_items)

    # Save output
    suffix = "" if args.use_think else "_no"
    output_path = Path(args.output_dir) / f"{args.output_name}{suffix}_think/train.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))

    logging.info("Dataset saved: %s", output_path)
    logging.info("Total samples: %d | Max question length: %d", len(dataset_items), max_len)


if __name__ == "__main__":
    main()
