#!/usr/bin/env python3
"""
Create training data for Citation task fine-tuning.
"""

import json
import os
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import argparse
import random

def create_citation_training_data(user_id, source_data_path, output_file_path):
    """Create training data for Citation task fine-tuning."""
    print(f"Creating Citation training data for user {user_id}...")
    
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
            
        target_title = target_item.get('title', '')
        target_citation = target_item.get('citation', '')
        
        if not target_title or not target_citation:
            continue
        
        distractor_candidates = [item for item in history_candidates if item.get('citation') != target_citation]
        if not distractor_candidates:
            continue
            
        distractor_titles = [item.get('title', '') for item in distractor_candidates]
        tokenized_corpus = [title.split() for title in distractor_titles]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = target_title.split()
        top_indices = bm25.get_top_n(tokenized_query, list(range(len(distractor_candidates))), n=1)
        
        if not top_indices:
            continue
            
        distractor_item = distractor_candidates[top_indices[0]]
        distractor_citation = distractor_item.get('citation', '')
        
        if random.random() < 0.5:
            option1, option2, correct_answer = target_citation, distractor_citation, "[1]"
        else:
            option1, option2, correct_answer = distractor_citation, target_citation, "[2]"
        
        history_texts = [f"Paper Title: {item['title']} Citation: {item['citation']}" for item in history_candidates[:1]]
        history_str = "\n".join(history_texts)
        
        query_input = f"For an author who has written the paper with the title \"{target_title}\", which reference is related? Just answer with [1] or [2] without explanation.\nReference: [1] - {option1} [2] - {option2}\nAnswer:"
        
        instruction = instruction_template.format(history=history_str, query_input=query_input)
        
        training_data.append({
            "instruction": instruction,
            "input": "",
            "output": correct_answer
        })

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully created {len(training_data)} Citation training samples")
    print(f"Data saved to: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training data for Citation task fine-tuning")
    parser.add_argument("--user_id", required=True, help="User ID")
    parser.add_argument("--input_file", required=True, help="Source data file path (e.g., filtered_users.json)")
    parser.add_argument("--output_file", required=True, help="Output training data file path")
    
    args = parser.parse_args()
    
    create_citation_training_data(args.user_id, args.input_file, args.output_file) 