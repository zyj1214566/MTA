#!/usr/bin/env python3
"""
Create evaluation data for Movie Tagging task using BM25 retrieval.
"""
import json
import os
import argparse
from typing import List, Dict
from tqdm import tqdm
from rank_bm25 import BM25Okapi

ALL_TAGS = [
    "sci-fi", "based on a book", "comedy", "action", "twist ending", 
    "dystopia", "dark comedy", "classic", "psychology", "fantasy", 
    "romance", "thought-provoking", "social commentary", "violence", 
    "true story"
]

def load_user_data(source_path: str, user_id: str) -> Dict:
    """Load user data from filtered_users.json."""
    with open(source_path, "r", encoding="utf-8") as f:
        all_users = json.load(f)
    return next((u for u in all_users if str(u.get("user_id")) == str(user_id)), {})

def build_bm25_corpus(profile: List[Dict[str, str]]):
    """Build BM25 corpus for movie_tagging."""
    corpus = [f"{item.get('description', '')} {item.get('tag', '')}".strip() for item in profile]
    tokenized_corpus = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)

def create_eval_samples(user_profile: List[Dict], user_queries: List[Dict], k_history: int = 1) -> List[Dict]:
    """Generate evaluation samples list."""
    if not user_profile:
        return []

    bm25 = build_bm25_corpus(user_profile)

    instruction_tpl = (
        "### User History:\n"
        "{history_section}\n\n"
        "### User Instruction:\n"
        "{query_input} Tag:"
    )

    eval_data = []
    for q in tqdm(user_queries, desc="Generating evaluation samples"):
        query_input = q.get("input")
        correct_tag = q.get("gold")
        if not (query_input and correct_tag):
            continue

        query_tokens = query_input.split()
        top_indices = bm25.get_top_n(query_tokens, list(range(len(user_profile))), n=k_history)
        
        history_items = [user_profile[i] for i in top_indices]
        
        history_section = "\n".join(
            [f"Description: {item.get('description', 'N/A')} Tag: {item.get('tag', 'N/A')}" for item in history_items]
        )

        instruction = instruction_tpl.format(
            history_section=history_section,
            query_input=query_input
        )
        
        eval_data.append({"instruction": instruction, "input": "", "output": correct_tag})
        
    return eval_data

def main():
    parser = argparse.ArgumentParser(description="Create evaluation data for Movie Tagging task")
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--input_file", required=True, help="Source data file path (filtered_users.json)")
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    user_data = load_user_data(args.input_file, args.user_id)
    if not user_data:
        raise ValueError(f"User {args.user_id} not found in {args.input_file}")

    profile, queries = user_data.get("profile", []), user_data.get("query", [])
    if not queries:
        print("Warning: User has no query, generating empty file")
        json.dump([], open(args.output_file, "w", encoding="utf-8"), ensure_ascii=False)
        return

    eval_samples = create_eval_samples(profile, queries)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(eval_samples, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(eval_samples)} evaluation samples -> {args.output_file}")


if __name__ == "__main__":
    main() 