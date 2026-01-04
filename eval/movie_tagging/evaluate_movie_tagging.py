#!/usr/bin/env python3
"""
Stacked LoRA model evaluation script for Movie Tagging task
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft.peft_model import PeftModel
from sklearn.metrics import accuracy_score, f1_score

ALL_TAGS = [
    "sci-fi", "based on a book", "comedy", "action", "twist ending", 
    "dystopia", "dark comedy", "classic", "psychology", "fantasy", 
    "romance", "thought-provoking", "social commentary", "violence", 
    "true story"
]

def load_stacked_lora_model_and_tokenizer(merged_base_model_path, stacked_lora_path):
    """Load merged base model and stacked LoRA adapter"""
    print(f"Loading merged base model from {merged_base_model_path}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_base_model_path,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    print(f"Loading tokenizer from {merged_base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(merged_base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading stacked LoRA from: {stacked_lora_path}")
    final_model = PeftModel.from_pretrained(merged_model, stacked_lora_path)
    final_model.eval()

    return final_model, tokenizer



def run_evaluation(model, tokenizer, eval_data):
    """Run evaluation on data and collect predictions"""
    results = []
    for sample in tqdm(eval_data, desc="Evaluating samples"):
        instruction = sample.get("instruction", "")
        ground_truth = sample.get("output", "")

        if not instruction or not ground_truth:
            continue

        messages = [{"role": "user", "content": instruction}]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        prompt_length = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        raw_prediction = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
      
        cleaned_prediction = raw_prediction
        
        results.append({
            "instruction": instruction,
            "raw_prediction": raw_prediction,
            "prediction": cleaned_prediction,
            "ground_truth": ground_truth
        })
    return results

def calculate_metrics(results, all_possible_labels):
    """Calculate Accuracy and Macro-F1 scores"""
    if not results:
        return {}
    
    predictions = [res["prediction"] for res in results]
    labels = [res["ground_truth"] for res in results]
    
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, labels=ALL_TAGS, average="macro", zero_division="warn")
    
    final_scores = {"accuracy": accuracy, "macro_f1": macro_f1}
    print("\n--- Metrics Summary ---")
    print(f"  Accuracy: {final_scores['accuracy']:.4f}")
    print(f"  Macro F1: {final_scores['macro_f1']:.4f}")
    print("----------------------\n")
    return final_scores

def print_result_samples(results, num_samples=5):
    """Print sample predictions and ground truth comparison"""
    print("\n--- Sample Predictions vs. Ground Truth ---\n")
    if not results:
        print("No results to display.")
        return
    for i, item in enumerate(results[:num_samples]):
        print(f"--- Sample {i+1} ---")
        print(f"  [Raw Prediction]: {item['raw_prediction']}")
        print(f"  [Cleaned Prediction]: {item['prediction']}")
        print(f"  [Ground Truth]: {item['ground_truth']}")
        print("-" * (len(f"--- Sample {i+1} ---")))
    print("\n-----------------------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a stacked LoRA model for the movie_tagging task.")
    parser.add_argument("--merged_base_model_path", required=True, help="Path to the merged base model (already contains fused LoRA).")
    parser.add_argument("--stacked_lora_path", required=True, help="Path to the stacked LoRA adapter (second stage).")
    parser.add_argument("--eval_data_file", required=True, help="Path to the single evaluation data JSON file for one user.")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value used for fusion.")
    parser.add_argument("--results_dir", required=True, help="Directory to save the evaluation results.")
    
    args = parser.parse_args()

    model, tokenizer = load_stacked_lora_model_and_tokenizer(args.merged_base_model_path, args.stacked_lora_path)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\nProcessing: {os.path.basename(args.eval_data_file)}")
    try:
        with open(args.eval_data_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Error reading or decoding JSON from file: {args.eval_data_file}")
        return

    if not eval_data:
        print("Skipping empty evaluation file.")
        return

    evaluation_results = run_evaluation(model, tokenizer, eval_data)
    metrics = calculate_metrics(evaluation_results, ALL_TAGS)
    print_result_samples(evaluation_results)

    user_id_match = re.search(r'_(\d+)\.json', os.path.basename(args.eval_data_file))
    user_id = user_id_match.group(1) if user_id_match else "unknown_user"
    
    result_filename = f"results_{user_id}_alpha_{args.alpha}.json"
    output_path = os.path.join(args.results_dir, result_filename)

    # Compatible format for aggregate_results.py
    final_output = {
        "user_id": user_id,
        "alpha_value": args.alpha,
        "metrics": metrics,
        "total_samples": len(eval_data),
        "predictions": [
            {
                "prediction": res["prediction"],
                "ground_truth": res["ground_truth"]
            } for res in evaluation_results
        ],
        "detailed_predictions_original": evaluation_results,
        "merged_base_model_path": args.merged_base_model_path,
        "stacked_lora_path": args.stacked_lora_path
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    main() 