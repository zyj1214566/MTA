import json
import os
import argparse
import re
import random
from pathlib import Path
from rank_bm25 import BM25Okapi
import yaml
from tqdm import tqdm

def parse_citation_input(input_str):
    """
    Helper function to parse the complex input string for the citation task.
    Returns the main title, option 1, and option 2.
    """
    try:
        # This regex is designed to be robust against different quote styles in JSON
        title_match = re.search(r'title "([^"]*)"', input_str)
        option1_match = re.search(r'\[1\]: "([^"]*)"', input_str)
        option2_match = re.search(r'\[2\]: "([^"]*)"', input_str)
        
        if not (title_match and option1_match and option2_match):
            return None, None, None
            
        return title_match.group(1), option1_match.group(1), option2_match.group(1)
    except Exception as e:
        print(f"Error parsing citation input: {input_str} | Error: {e}")
        return None, None, None

def prepare_finetuning_data(task_name: str, top_k: int):
    """
    Prepare alpaca-style dataset, LlamaFactory yaml configs, and a run-all shell script for a specific task.
    This final version correctly handles all specified tasks by iterating through the 'profile'.
    """
    
    # --- 1. Configuration and Template Loading ---
    project_root = '.'
    llamafactory_root = './LLaMA-Factory'

    try:
        prompt_file_path = './prompt/prompt_template.json'
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt_templates = json.load(f)
        task_prompts = prompt_templates[task_name]
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not load or find prompts for task '{task_name}'. Please check {prompt_file_path}. Details: {e}")
        return

    anchor_data_path = f'./data/{task_name}/anchor_user_data.jsonl'
    task_k_id = f"{task_name}_k{top_k}"

    # Define and create all necessary directories
    llama_factory_data_dir = os.path.join(llamafactory_root, 'data')
    dataset_info_path = os.path.join(llama_factory_data_dir, 'dataset_info.json')
    data_task_dir = os.path.join(llama_factory_data_dir, task_k_id)
    finetuning_scripts_task_dir = os.path.join(llamafactory_root, 'examples', 'train_lora', task_k_id)
    saves_dir = os.path.join(llamafactory_root, 'saves')
    task_shell_dir = project_root
    for directory in [data_task_dir, finetuning_scripts_task_dir, saves_dir, task_shell_dir]:
        os.makedirs(directory, exist_ok=True)
    
    dataset_info = json.load(open(dataset_info_path, 'r', encoding='utf-8')) if os.path.exists(dataset_info_path) else {}

    try:
        with open(anchor_data_path, 'r', encoding='utf-8') as f:
            user_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Anchor data file not found at {anchor_data_path}")
        return

    user_configs = []

    for line in tqdm(user_lines, desc=f"Processing Users for {task_name}"):
        user_data = json.loads(line)
        user_id = user_data['user_id']
        profile = user_data['profile']
        
        alpaca_formatted_data = []

        # Iterate through each profile item to generate training samples
        for i, current_item in enumerate(profile):
            history_pool = profile[:i] + profile[i+1:]
            if not history_pool: continue

            # --- Task-specific data processing ---
            if task_name == 'citation':
                query_title = current_item.get("title")
                correct_citation = current_item.get("citation")
                if not (query_title and correct_citation):
                    continue

                # 1. Use BM25 to select a hard negative example
                distract_pool = [p for p in history_pool if p.get("citation") != correct_citation]
                if not distract_pool:
                    continue

                distract_titles = [p.get("title", "") for p in distract_pool]
                distract_bm25 = BM25Okapi([t.split() for t in distract_titles])
                distract_indices = distract_bm25.get_top_n(query_title.split(), list(range(len(distract_pool))), n=1)
                
                hard_negative_item = distract_pool[distract_indices[0]] if distract_indices else random.choice(distract_pool)
                wrong_citation = hard_negative_item.get("citation")

                # 2. Create the final instruction part (without history) from the template
                if random.random() < 0.5:
                    option1, option2, gold_output = correct_citation, wrong_citation, "[1]"
                else:
                    option1, option2, gold_output = wrong_citation, correct_citation, "[2]"
                
                final_instruction = task_prompts["prompt"].format(
                    query_title=query_title, 
                    opt1=option1, 
                    opt2=option2
                )
                
                # 3. Prepare variables for the common history retrieval block
                history_corpus = [
                    task_prompts["retrieval_history"].format(title=item.get("title", "N/A"), citation=item.get("citation", "N/A")) 
                    for item in history_pool
                ]
                retrieval_query = task_prompts["retrieval_query_wokey"].format(query_title)

            else:
                # Generalized logic for other tasks
                key_map = {
                    "movie_tagging": ('description', 'tag'), 
                    "product_rating": ('text', 'score'),
                    "news_headline": ('text', 'title'),
                    "scholarly_title": ('abstract', 'title'),
                    "tweet_paraphrase": ('text', 'text') # Added tweet_paraphrase
                }
                if task_name not in key_map:
                    print(f"Unsupported task for this logic block: {task_name}")
                    continue
                input_key, output_key = key_map[task_name]
                
                # Filter history pool to ensure items have the necessary keys
                valid_history_pool = [item for item in history_pool if input_key in item]
                if not valid_history_pool: continue
                
                # For tweet_paraphrase, the history format is simpler
                if task_name == 'tweet_paraphrase':
                    history_corpus = [
                        task_prompts["retrieval_history"].format(**{input_key: item[input_key]}) 
                        for item in valid_history_pool
                    ]
                else:
                    valid_history_pool = [item for item in valid_history_pool if output_key in item]
                    if not valid_history_pool: continue
                    history_corpus = [
                        task_prompts["retrieval_history"].format(**{input_key: item[input_key], output_key: str(item[output_key])}) 
                        for item in valid_history_pool
                    ]


                # Ensure current item has the necessary keys
                if input_key not in current_item:
                    continue
                
                # For tweet_paraphrase, the input and output are the same text
                query_text = current_item[input_key]
                gold_output = current_item[output_key] if output_key in current_item else current_item[input_key]

                final_instruction = task_prompts["prompt"].format(query_text)
                retrieval_query = task_prompts["retrieval_query_wokey"].format(query_text)

            # --- Common BM25 and Instruction Building ---
            retrieved_history_str = ""
            if top_k > 0 and history_corpus:
                tokenized_corpus = [doc.split(" ") for doc in history_corpus]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = retrieval_query.split(" ")
                retrieved_history_items = bm25.get_top_n(tokenized_query, history_corpus, n=top_k)
                retrieved_history_str = "\n\n".join(retrieved_history_items)
            
            # Combine history with the final instruction
            if retrieved_history_str:
                instruction = f"### User History:\n{retrieved_history_str}\n\n{final_instruction}"
            else:
                instruction = final_instruction

            alpaca_formatted_data.append({"instruction": instruction, "input": "", "output": str(gold_output)})

        if not alpaca_formatted_data:
            print(f"Warning: No training data generated for user {user_id}.")
            continue
        
        # --- The rest of the script is common for all tasks ---
        dataset_filename = f"user_{user_id}_data.json"
        dataset_filepath = os.path.join(data_task_dir, dataset_filename)
        with open(dataset_filepath, 'w', encoding='utf-8') as f:
            json.dump(alpaca_formatted_data, f, indent=2, ensure_ascii=False)

        dataset_name = f"{task_k_id}_user_{user_id}"
        dataset_info[dataset_name] = {"file_name": os.path.join(task_k_id, dataset_filename)}

        yaml_config_str = f"""
# model
model_name_or_path: ../model/Meta-Llama-3-8B-Instruct
trust_remote_code: true
# method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
# dataset
dataset: {dataset_name}
template: llama3
cutoff_len: 8192
max_samples: 5000
overwrite_cache: true
preprocessing_num_workers: 16
# output
output_dir: {os.path.join('saves', 'llama3-8b', 'lora', task_k_id, f'sft_user_{user_id}')}
logging_steps: 5
save_steps: 2000
plot_loss: true
overwrite_output_dir: true
# train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
"""
        yaml_filename = f"user_{user_id}_finetune.yaml"
        yaml_filepath = os.path.join(finetuning_scripts_task_dir, yaml_filename)
        with open(yaml_filepath, 'w', encoding='utf-8') as f:
            f.write(yaml_config_str)
        
        user_configs.append(yaml_filepath)

    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    run_script_content = "#!/bin/bash\n"
    run_script_content += f"cd {llamafactory_root}\n"
    for config_path in user_configs:
        relative_config_path = os.path.relpath(config_path, llamafactory_root)
        run_script_content += f"llamafactory-cli train {relative_config_path}\n"
    
    run_script_path = os.path.join(task_shell_dir, f'run_{task_k_id}_training.sh')
    with open(run_script_path, 'w', encoding='utf-8') as f:
        f.write(run_script_content)
    os.chmod(run_script_path, 0o755)

    print(f"\n--- Preparation Complete for Task: {task_k_id} ---")
    print(f"Generated {len(user_configs)} user-specific configurations.")
    print(f"To start all trainings for this task, run: {run_script_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare finetuning data for a specific task.")
    parser.add_argument("--task_name", type=str, required=True, help="Task name (e.g., movie_tagging, product_rating, citation, news_headline, scholarly_title, tweet_paraphrase)")
    parser.add_argument("--top_k", type=int, default=1, help="k value used in BM25 retrieval")
    args = parser.parse_args()
    prepare_finetuning_data(task_name=args.task_name, top_k=args.top_k)