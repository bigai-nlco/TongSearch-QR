import os
import json
import argparse
from datasets import load_dataset
from tqdm import tqdm
from vllm_model import VllmModel

# Default list of target datasets
TARGET_LIST = [
    'biology', 'earth_science', 'economics', 'pony', 'psychology',
    'sustainable_living', 'aops', 'theoremqa_theorems',
    'theoremqa_questions', 'robotics', 'stackoverflow', 'leetcode'
]

def load_existing_queries(filename):
    """Load previously reasoned queries if the file exists."""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_queries(filename, query_map):
    """Save the current query map to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(query_map, f, ensure_ascii=False, indent=4)

def reason_queries(model, examples, query_map, filename):
    """Iterate over examples and generate reasoned queries."""
    for item in tqdm(examples, desc="Reasoning queries"):
        question = item['query']
        if question in query_map:
            continue

        prompt = (
            f"Instructions:\n"
            f"1. Identify the essential problem.\n"
            f"2. Think step by step to reason and describe what information could be relevant and helpful to address the questions in detail.\n"
            f"3. Draft an answer with as many thoughts as you have.\n"
            f"Query: {question}\n\n"
        )

        query_map[question] = model.predict(prompt)
        save_queries(filename, query_map)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reason queries using VllmModel.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Vllm model checkpoint.")
    parser.add_argument("--use_think", action="store_true",
                        help="Enable think mode for reasoning.")
    parser.add_argument("--output_file", type=str, default="reasoned_query.json",
                        help="Filename to save reasoned queries.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize think_flag and system_prompt
    if args.use_think:
        think_flag = True
        system_prompt = (
            "A conversation between User and Assistant. "
            "The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind "
            "and then provides the user with the answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
            "i.e., <think> reasoning process here </think> <answer> answer here </answer>."
        )
    else:
        think_flag = False
        system_prompt = ""

    # Initialize the model
    model = VllmModel(args.model_path, system_prompt=system_prompt)
    model.think_flag = think_flag

    # Load already reasoned queries if any
    query_map = load_existing_queries(args.output_file)

    # Iterate over each target dataset
    for target in TARGET_LIST:
        examples = load_dataset('xlangai/BRIGHT', 'examples')[target]
        reason_queries(model, examples, query_map, args.output_file)

if __name__ == "__main__":
    main()