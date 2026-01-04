#!/usr/bin/env python3
"""
Create training data for Scholarly Title task.
"""
import json
import os
import argparse
from tqdm import tqdm
from rank_bm25 import BM25Okapi

def create_scholarly_title_training_data(user_id, source_data_path, output_file_path):
    """Create training data for Scholarly Title task using BM25 for history selection."""
    print(f"Creating Scholarly Title training data for user {user_id}...")

    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(source_data_path, 'r', encoding='utf-8') as f:
            all_users_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Source data file not found: {source_data_path}")
        return

    target_user_data = next((user for user in all_users_data if str(user.get("user_id")) == str(user_id)), None)

    if not target_user_data:
        print(f"ERROR: User '{user_id}' not found in {source_data_path}")
        return

    user_profile = target_user_data.get("profile", [])

    if len(user_profile) < 2:
        print(f"Warning: User {user_id} has insufficient profile data ({len(user_profile)} items), cannot create valid training samples.")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print(f"Created empty training file: {output_file_path}")
        return

    print(f"Processing user: {user_id}, profile data: {len(user_profile)} items")

    training_data = []

    for i, target_item in enumerate(tqdm(user_profile, desc="Generating training samples")):
        history_candidates = user_profile[:i] + user_profile[i+1:]
        
        if not history_candidates:
            continue
            
        target_abstract = target_item.get('abstract', '')
        
        candidate_abstracts = [item.get('abstract', '') for item in history_candidates]
        tokenized_corpus = [doc.split(" ") for doc in candidate_abstracts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = target_abstract.split(" ")
        top_indices = bm25.get_top_n(tokenized_query, list(range(len(history_candidates))), n=1)
        
        if not top_indices:
            continue
            
        most_relevant_item = history_candidates[top_indices[0]]
        
        history_str = f"abstract: {most_relevant_item.get('abstract', '')} title: {most_relevant_item.get('title', '')}"
        
        instruction_text = (
            f"### User History:\n{history_str}\n\n\n"
            f"### User Instruction:\n"
            f"Generate a title for the following abstract of a paper.\n"
            f"Abstract: {target_abstract} Title:"
        )

        training_data.append({
            "instruction": instruction_text,
            "input": "",
            "output": target_item.get('title', '')
        })

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully created {len(training_data)} training samples for user {user_id}.")
    print(f"Data saved to: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training data for Scholarly Title task")
    parser.add_argument("--user_id", required=True, help="User ID")
    parser.add_argument("--input_file", required=True, help="Source data file path (e.g., filtered_users.json)")
    parser.add_argument("--output_file", required=True, help="Output training data file path")
    
    args = parser.parse_args()
    
    create_scholarly_title_training_data(args.user_id, args.input_file, args.output_file)