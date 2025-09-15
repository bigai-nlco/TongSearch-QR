#!/usr/bin/env python3
"""
Sample questions from the HuggingFaceH4/stack-exchange-preferences dataset
by specific domains and save them in JSON format.
Usage example:
python ./scripts/data_process/sample_stackexchange.py --max-per-domain 1200 --save-path my_sample.json
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from urllib.parse import urlparse

from datasets import load_dataset
from tqdm import tqdm


# Target domains
TARGETS = [
    "Stackoverflow", "biology", "chemistry", "codereview", 
    "cs", "earthscience", "economics", "math", "robotics"
]

TARGET_SET = set(TARGETS)


DEFAULT_INSTRUCTION = (
    f'Instructions:\n'
    f'1. Identify the essential problem.\n'
    f'2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail.\n'
    f'3. Draft an answer with as many thoughts as you have.\n'
    f'Question: '
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample StackExchange questions by domain and save as JSON."
    )
    parser.add_argument("--seed", type=int, default=2025, help="random seed for reproducibility")
    parser.add_argument("--max-per-domain", type=int, default=1200, help="maximum number of questions per domain")
    parser.add_argument("--save-path", type=str, default="sampled_questions.json", help="output JSON file path")
    parser.add_argument("--log-level", type=str, default="INFO", help="logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def collect_questions(dataset_split, targets: set) -> dict:
    """
    Collect questions grouped by domain.
    Returns:
        dict: {domain_name: [list of questions]}
    """
    result = defaultdict(list)
    for row in tqdm(dataset_split, desc="Collecting questions"):
        question = row['question']
        metadata = row['metadata'][0]
        if not question:
            continue
        domain = metadata.strip('https://').split('.')[0]
        if domain in targets:
            result[domain].append(question)
    return result


def build_samples(question_dict: dict, max_per_domain: int, seed: int) -> list:
    """
    Randomly sample questions for each domain and format them as output items.
    Each item contains an empty 'instruction' and an 'input' with the prompt + question.
    """
    random.seed(seed)
    samples = []
    for domain, questions in question_dict.items():
        random.shuffle(questions)
        for q in questions[:max_per_domain]:
            samples.append({
                "instruction": "",
                "input": DEFAULT_INSTRUCTION + q
            })
    return samples


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="[%(levelname)s] %(message)s")

    logging.info("Loading dataset HuggingFaceH4/stack-exchange-preferences")
    dataset = load_dataset("HuggingFaceH4/stack-exchange-preferences")
    split_data = dataset['train']

    # Collect questions by domain
    question_dict = collect_questions(split_data, TARGET_SET)
    for domain in TARGETS:
        logging.info("Domain %-15s: %d questions", domain, len(question_dict.get(domain, [])))

    # Build sampled data
    sampled = build_samples(question_dict, args.max_per_domain, args.seed)
    logging.info("Total sampled: %d", len(sampled))

    # Save as JSON
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=4)

    logging.info("Saved to %s", args.save_path)


if __name__ == "__main__":
    main()