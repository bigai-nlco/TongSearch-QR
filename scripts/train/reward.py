import logging
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer


scoring_model_path = "BAAI/bge-base-en-v1.5"
scoring_model = SentenceTransformer(scoring_model_path)


def extract_query_text(solution_str: str, use_think_flag: bool) -> str:
    """
    Extract the actual query text from a solution string.
    
    - If `<|im_start|>assistant` exists, strip everything before it.
    - If `think_flag` is True, extract the text inside <answer>...</answer> 
      after optional <think>...</think>.
    """
    query = solution_str

    # Remove system prompt part if exists
    if "<|im_start|>assistant" in query:
        query = query.split("<|im_start|>assistant", 1)[1]

    # Extract <answer>...</answer> if think_flag is set
    if use_think_flag:
        pattern = re.compile(r"<think>.*?</think>\s*<answer>(.*?)</answer>", re.DOTALL)
        match = pattern.search(query)
        if not match:
            return ""  # Return empty string if no match found
        query = match.group(1)

    return query.strip()


def compute_score_bge(solution_str: str, ground_truth: str) -> float:
    """
    Compute the BGE-based reward score.

    Args:
        solution_str (str): The generated solution string.
        ground_truth (str): JSON string with fields:
            - "query": original query
            - "pos_docs_list": list of positive documents
            - "think_flag": whether <think>/<answer> formatting should be used

    Returns:
        float: Advantage score (scaled difference between query and original question similarity).
               Returns -1 if parsing fails.
    """
    try:
        # Parse the ground truth JSON
        label_item = json.loads(ground_truth)
        question = label_item["query"]
        docs = label_item["pos_docs_list"]
        think_flag = label_item.get("think_flag", False)

        # Extract the cleaned query text
        query = extract_query_text(solution_str, think_flag)
        if not query:
            return -1

        # Compute embeddings
        question_emb = scoring_model.encode([question], normalize_embeddings=True)
        query_emb = scoring_model.encode([query], normalize_embeddings=True)
        doc_emb = scoring_model.encode(docs, normalize_embeddings=True)

        # Cosine similarity (dot product since embeddings are normalized)
        original_sim = question_emb @ doc_emb.T
        query_sim = query_emb @ doc_emb.T

        # Compute advantage score
        advantage = float(np.sum(query_sim - original_sim) / len(docs) * 10)
        logging.info("Got Advantage:", advantage)
        return advantage

    except Exception as e:
        logging.ino(f"[Error in compute_score_bge] {e}")
        return -1


def bge_reward_func(completions, answer, **kwargs):
    """
    Reward function that computes BGE-based scores for multiple completions.

    Args:
        completions (list): List of completions, each a list of dicts with 'content'.
        answers (list): List of ground truth JSON strings.
        kwargs: Unused, for compatibility.

    Returns:
        list: List of reward scores for each (completion, answer) pair.
    """
    responses = [completion[0]["content"] for completion in completions]
    return [compute_score_bge(r, a) for r, a in zip(responses, answer)]