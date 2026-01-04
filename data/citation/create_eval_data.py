import json
import os
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import argparse
import random

def create_citation_eval_data(user_id, source_data_path, output_file_path):
    """Creates evaluation data for citation task using query as input and BM25."""
    print(f"Starting citation evaluation data creation for user_id: {user_id}...")
    
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
    instruction_template = "### User History:\n{history}\n\n### User Instruction:\n{query_input}"

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
    profile_corpus = [item.get('title', '') for item in user_profile]
    tokenized_corpus = [doc.split(" ") for doc in profile_corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    for query_item in tqdm(user_queries, desc=f"Generating citation samples for user {user_id}"):
        query_input = query_item.get('input', '')
        gold_answer = query_item.get('gold', '')
        
        if 'title "' in query_input:
            title_start = query_input.find('title "') + 7
            title_end = query_input.find('"', title_start)
            if title_end > title_start:
                paper_title = query_input[title_start:title_end]
            else:
                continue
        else:
            continue
        
        option1_match = query_input.find('[1]: "') + 6
        option1_end = query_input.find('"', option1_match)
        option2_match = query_input.find('[2]: "') + 6
        option2_end = query_input.find('"', option2_match)
        
        if option1_end <= option1_match or option2_end <= option2_match:
            continue
            
        option1 = query_input[option1_match:option1_end]
        option2 = query_input[option2_match:option2_end]
        
        tokenized_query = paper_title.split(" ")
        top_n_indices = bm25.get_top_n(tokenized_query, list(range(len(user_profile))), n=1)
        
        if not top_n_indices:
            continue

        relevant_item = user_profile[top_n_indices[0]]
        history_str = f"Paper Title: {relevant_item.get('title', '')} Citation: {relevant_item.get('citation', '')}"

        standardized_query_input = f"For an author who has written the paper with the title \"{paper_title}\", which reference is related? Just answer with [1] or [2] without explanation.\nReference: [1] - {option1} [2] - {option2}\nAnswer:"

        instruction = instruction_template.format(history=history_str, query_input=standardized_query_input)
        
        alpaca_formatted_data.append({
            "instruction": instruction,
            "input": "",
            "output": gold_answer
        })

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_formatted_data, f, indent=2)

    print(f"Successfully created {len(alpaca_formatted_data)} citation evaluation samples for user {user_id}.")
    print(f"Data saved to: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create citation evaluation data for a specific user.")
    parser.add_argument("--user_id", required=True, help="The ID of the user to process.")
    parser.add_argument("--input_file", required=True, help="Path to the source data file (e.g., filtered_users.json).")
    parser.add_argument("--output_file", required=True, help="Path to save the generated evaluation data.")
    
    args = parser.parse_args()
    
    create_citation_eval_data(args.user_id, args.input_file, args.output_file) 