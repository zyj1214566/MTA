#!/usr/bin/env python3
"""
Create evaluation data for News Headline task.
"""
import json
import os
import argparse
from tqdm import tqdm
from rank_bm25 import BM25Okapi

def clean_article_text(text):
    """Clean article text, remove duplicate instruction prefixes."""
    if not text:
        return text
    

    prefix_to_remove = "Generate a headline for the following article:"
    
    cleaned_text = text.strip()
    

    if cleaned_text.startswith(prefix_to_remove):
        cleaned_text = cleaned_text[len(prefix_to_remove):].strip()
    
    return cleaned_text

def create_news_headline_eval_data(user_id, source_data_path, output_file_path):
    """Create evaluation data for News Headline task using BM25 for history selection."""
    print(f"Creating News Headline evaluation data for user {user_id}...")

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
    user_queries = target_user_data.get("query", [])

    if not user_profile:
        print(f"Warning: User {user_id} has empty profile, cannot generate valid evaluation samples.")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print(f"Created empty evaluation file: {output_file_path}")
        return

    if not user_queries:
        print(f"Warning: User {user_id} has empty query, cannot generate evaluation samples.")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print(f"Created empty evaluation file: {output_file_path}")
        return

    print(f"Processing user: {user_id}, profile: {len(user_profile)} items, queries: {len(user_queries)} items")

    eval_data = []
    instruction_template = "### User History:\n{history}\n\n\n### User Instruction:\n{query_input}"


    if user_queries:

        profile_texts = [f"{item.get('title', '')} {item.get('text', '')}".strip() for item in user_profile]
        tokenized_corpus = [doc.split(" ") for doc in profile_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        for query_item in tqdm(user_queries, desc="Generating evaluation samples"):
            query_input_text = query_item.get('input', '')
            

            cleaned_article = clean_article_text(query_input_text)
            

            tokenized_query = cleaned_article.split(" ")
            top_indices = bm25.get_top_n(tokenized_query, list(range(len(user_profile))), n=1)
            
            if not top_indices:

                history_str = "Article: No relevant history found. Headline: N/A"
            else:
                most_relevant_item = user_profile[top_indices[0]]

                history_str = f"Article: {most_relevant_item.get('text', '')} Headline: {most_relevant_item.get('title', '')}"


            query_input = f"Generate a headline for the following article.\nArticle: {cleaned_article} Headline:"
            
            instruction = instruction_template.format(history=history_str, query_input=query_input)
            
            eval_data.append({
                "instruction": instruction,
                "input": "",
                "output": query_item.get('gold', '')
            })
    else:

        eval_count = min(3, len(user_profile))
        for i in range(eval_count):
            current_item = user_profile[i]
            current_text = current_item.get('text', '')
            current_title = current_item.get('title', '')
            
            if not current_text or not current_title:
                continue
            

            other_profile_items = [item for j, item in enumerate(user_profile) if j != i]
            
            if not other_profile_items:
                history_str = "Article: No relevant history found. Headline: N/A"
            else:

                other_texts = [f"{item.get('title', '')} {item.get('text', '')}".strip() for item in other_profile_items]
                tokenized_corpus = [doc.split(" ") for doc in other_texts]
                bm25 = BM25Okapi(tokenized_corpus)
                
    
                tokenized_query = current_text.split(" ")
                top_indices = bm25.get_top_n(tokenized_query, list(range(len(other_profile_items))), n=1)
                
                if not top_indices:
                    history_str = "Article: No relevant history found. Headline: N/A"
                else:
                    most_relevant_item = other_profile_items[top_indices[0]]
                    history_str = f"Article: {most_relevant_item.get('text', '')} Headline: {most_relevant_item.get('title', '')}"


            query_input = f"Generate a headline for the following article.\nArticle: {current_text} Headline:"
            
            instruction = instruction_template.format(history=history_str, query_input=query_input)
            
            eval_data.append({
                "instruction": instruction,
                "input": "",
                "output": current_title
            })

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully created {len(eval_data)} evaluation samples for user {user_id}.")
    print(f"Data saved to: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create evaluation data for News Headline task")
    parser.add_argument("--user_id", required=True, help="User ID to process")
    parser.add_argument("--input_file", required=True, help="Source data file path (e.g., filtered_users.json)")
    parser.add_argument("--output_file", required=True, help="Output evaluation data file path")
    
    args = parser.parse_args()
    
    create_news_headline_eval_data(args.user_id, args.input_file, args.output_file) 