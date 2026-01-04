#!/usr/bin/env python3
"""
Create training data for Movie Tagging task using Leave-One-Out approach.
"""
import json
import os
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import random

def load_user_data(source_path: str, user_id: str) -> Dict:
    """Load user data from source file."""
    with open(source_path, "r", encoding="utf-8") as f:
        all_users = json.load(f)
    return next((u for u in all_users if str(u.get("user_id")) == str(user_id)), {})

def build_bm25_from_profile(profile_items: List[Dict]) -> BM25Okapi:
    """Build BM25 index from profile items."""
    corpus = [f"{item.get('description', '')} {item.get('tag', '')}".strip() for item in profile_items]
    tokenized_corpus = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)

def create_training_samples(user_profile: List[Dict]) -> List[Dict]:
    """Generate training samples list."""
    
    instruction_tpl = (
        "### User History:\n"
        "Description: {hist_description} Tag: {hist_tag}\n\n"
        "### User Instruction:\n"
        "Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story]\n"
        "Description: {query_description} Tag:"
    )

    training_data = []
    for i, current_item in enumerate(tqdm(user_profile, desc="Generating training samples")):
        query_description = current_item.get("description")
        correct_tag = current_item.get("tag")

        if not (query_description and correct_tag):
            continue

        history_pool = user_profile[:i] + user_profile[i+1:]
        if not history_pool:
            continue

        bm25 = build_bm25_from_profile(history_pool)
        query_tokens = query_description.split()
        top_indices = bm25.get_top_n(query_tokens, list(range(len(history_pool))), n=1)
        
        history_item = history_pool[top_indices[0]] if top_indices else random.choice(history_pool)
        hist_description = history_item.get('description', 'N/A')
        hist_tag = history_item.get('tag', 'N/A')

        instruction = instruction_tpl.format(
            hist_description=hist_description,
            hist_tag=hist_tag,
            query_description=query_description
        )
        
        training_data.append({"instruction": instruction, "input": "", "output": correct_tag})
        
    return training_data

def main():
    parser = argparse.ArgumentParser(description="Create training data for Movie Tagging task")
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--input_file", required=True, help="Source data file path (filtered_users.json)")
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    user_data = load_user_data(args.input_file, args.user_id)
    if not user_data:
        raise ValueError(f"User {args.user_id} not found in {args.input_file}")

    profile = user_data.get("profile", [])
    if len(profile) < 2:
        print("Warning: Profile has less than 2 items, cannot generate valid training samples. Creating empty file.")
        json.dump([], open(args.output_file, "w", encoding="utf-8"))
        return

    training_samples = create_training_samples(profile)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(training_samples, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(training_samples)} training samples -> {args.output_file}")

if __name__ == "__main__":
    main() 