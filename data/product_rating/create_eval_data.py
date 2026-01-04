import json
import os
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import argparse

def create_product_rating_eval_data(user_id, source_data_path, output_file_path):
    """Creates evaluation data for Product Rating task using query and BM25."""
    print(f"Starting evaluation data creation for user_id: {user_id}...")
    
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(source_data_path, 'r', encoding='utf-8') as f:
            all_users_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Source data not found at {source_data_path}")
        return

    target_user_data = None
    for user_data in all_users_data:
        if str(user_data.get("user_id")) == str(user_id):
            target_user_data = user_data
            break
    
    if not target_user_data:
        print(f"ERROR: User with ID '{user_id}' not found in {source_data_path}")
        return

    user_profile = target_user_data.get("profile", [])
    user_queries = target_user_data.get("query", [])
    
    print(f"Processing user: {user_id} with {len(user_profile)} profile items and {len(user_queries)} queries.")

    alpaca_formatted_data = []
    instruction_template = "### User History:\n{history}\n\n\n### User Instruction:\n{query_input}"

    if len(user_profile) == 0:
        print("Warning: User has no profile items. Cannot generate data.")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print(f"Created an empty evaluation file at: {output_file_path}")
        return

    if len(user_queries) == 0:
        print("Warning: User has no query items. Cannot generate data.")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print(f"Created an empty evaluation file at: {output_file_path}")
        return

    # Prepare corpus for BM25 using all profile items
    profile_corpus = [item['text'] for item in user_profile]
    tokenized_corpus = [doc.split(" ") for doc in profile_corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    for query_item in tqdm(user_queries, desc=f"Generating samples for user {user_id}"):
        # Extract the review text from query input
        query_input = query_item.get('input', '')
        
        # Ensure the query_input ends with "Score:" if it doesn't already
        if not query_input.endswith('Score:'):
            if 'Score:' in query_input:
                # If Score: exists but doesn't end with it, make sure it ends properly
                query_input = query_input.rstrip() + ' Score:'
            else:
                # If no Score: exists, add it
                query_input = query_input.rstrip() + ' Score:'
        
        # Parse the review text from the query input for BM25 matching
        if 'Review:' in query_input:
            review_text = query_input.split('Review:')[-1].split('Score:')[0].strip()
        else:
            review_text = query_input.split('Score:')[0].strip()
        
        # Find the most relevant profile item using BM25
        tokenized_query = review_text.split(" ")
        top_n_indices = bm25.get_top_n(tokenized_query, list(range(len(user_profile))), n=1)
        
        if not top_n_indices:
            continue

        most_relevant_item = user_profile[top_n_indices[0]]
        
        # Format history string using only the most relevant item
        history_str = f"Review: {most_relevant_item['text']} Score: {most_relevant_item['score']}"

        instruction = instruction_template.format(history=history_str, query_input=query_input)
        
        alpaca_formatted_data.append({
            "instruction": instruction,
            "input": "",
            "output": query_item.get('gold', '')
        })

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_formatted_data, f, indent=2)

    print(f"Successfully created {len(alpaca_formatted_data)} evaluation samples for user {user_id}.")
    print(f"Data saved to: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create evaluation data for Product Rating task.")
    parser.add_argument("--user_id", required=True, help="User ID to process.")
    parser.add_argument("--input_file", required=True, help="Source data file path (e.g., filtered_users.json).")
    parser.add_argument("--output_file", required=True, help="Output evaluation data file path.")
    
    args = parser.parse_args()
    
    create_product_rating_eval_data(args.user_id, args.input_file, args.output_file) 