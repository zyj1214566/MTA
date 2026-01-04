#!/usr/bin/env python3
"""
Create training data for Product Rating task fine-tuning.
"""

import json
import os
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import argparse
import random

def create_bitfit_training_data(user_id, source_data_path, output_file_path):
    """Create training data for Product Rating task fine-tuning."""
    print(f"Creating Product Rating training data for user {user_id}...")
    
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(source_data_path, 'r', encoding='utf-8') as f:
            all_users_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Source data file not found: {source_data_path}")
        return

    target_user_data = None
    for user_data in all_users_data:
        if str(user_data.get("user_id")) == str(user_id):
            target_user_data = user_data
            break
    
    if not target_user_data:
        print(f"ERROR: User '{user_id}' not found in {source_data_path}")
        return

    user_profile = target_user_data.get("profile", [])
    
    if len(user_profile) < 2:
        print(f"Warning: User {user_id} has insufficient profile data ({len(user_profile)} items), cannot create training data")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print(f"Created empty training file: {output_file_path}")
        return

    print(f"Processing user: {user_id}, profile data: {len(user_profile)} items")
    print(f"Using all {len(user_profile)} items for training")

    training_data = []
    instruction_template = "### User History:\n{history}\n\n### User Instruction:\n{query_input}"

    for i, target_item in enumerate(tqdm(user_profile, desc="Generating training samples")):
        history_candidates = user_profile[:i] + user_profile[i+1:]
        
        if not history_candidates:
            continue
            
        target_text = target_item.get('text', '')
        
        candidate_texts = [item['text'] for item in history_candidates]
        tokenized_corpus = [doc.split(" ") for doc in candidate_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = target_text.split(" ")
        top_indices = bm25.get_top_n(tokenized_query, list(range(len(history_candidates))), n=1)
        
        if not top_indices:
            continue
            
        most_relevant_item = history_candidates[top_indices[0]]
        
        history_str = f"Review: {most_relevant_item['text']} Score: {most_relevant_item['score']}"
        
        query_input = f"What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation.\nReview: {target_text} Score:"
        
        instruction = instruction_template.format(history=history_str, query_input=query_input)
        
        training_data.append({
            "instruction": instruction,
            "input": "",
            "output": str(target_item['score'])
        })

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully created {len(training_data)} Product Rating training samples")
    print(f"Data saved to: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training data for Product Rating task fine-tuning")
    parser.add_argument("--user_id", required=True, help="User ID")
    parser.add_argument("--input_file", required=True, help="Source data file path (e.g., filtered_users.json)")
    parser.add_argument("--output_file", required=True, help="Output training data file path")
    
    args = parser.parse_args()
    
    create_bitfit_training_data(args.user_id, args.input_file, args.output_file) 