#!/usr/bin/env python3
"""
Stacked LoRA model evaluation script for Citation task
"""

import os
import json
import torch
import logging
import argparse
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALL_CITATION_LABELS = ["[1]", "[2]"]

def load_stacked_lora_model_and_tokenizer(merged_base_model_path, stacked_lora_path):
    """Load merged base model and stacked LoRA adapter."""
    logger.info(f"Loading merged model from {merged_base_model_path}...")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(merged_base_model_path)
        merged_model = AutoModelForCausalLM.from_pretrained(
            merged_base_model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Merged model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load merged model: {e}")
        return None, None

    logger.info(f"Loading stacked LoRA from {stacked_lora_path}...")
    try:
        final_model = PeftModel.from_pretrained(merged_model, stacked_lora_path)
        final_model.eval()
        logger.info("Stacked LoRA loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load stacked LoRA: {e}")
        return None, None

    logger.info("Stacked LoRA model and tokenizer loaded successfully.")
    return final_model, tokenizer



def run_evaluation(model, tokenizer, eval_data):
    """Run evaluation on data and collect predictions"""
    results = []
    for sample in tqdm(eval_data, desc="Evaluating samples"):
        instruction = sample.get("instruction", "")
        ground_truth = sample.get("output", "")

        if not instruction or not ground_truth:
            continue

        try:
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
                    max_new_tokens=5,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            raw_prediction = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
   
            cleaned_prediction = raw_prediction
            
        except Exception as e:
            logger.warning(f"Error processing sample: {e}")
            raw_prediction = "Error during generation"
            cleaned_prediction = "ERROR"
        
        results.append({
            "instruction": instruction,
            "raw_prediction": raw_prediction,
            "prediction": cleaned_prediction,
            "ground_truth": ground_truth
        })
    return results

def calculate_metrics(results):
    """Calculate Accuracy and F1 scores."""
    if not results:
        return {}
    
    predictions = [res["prediction"] for res in results]
    labels = [res["ground_truth"] for res in results]
    
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, labels=ALL_CITATION_LABELS, average="macro", zero_division=0)

    final_scores = {
        "accuracy": accuracy, 
        "f1": macro_f1
    }
    
    print("\n--- Metrics Summary ---")
    print(f"  Accuracy: {final_scores['accuracy']:.4f}")
    print(f"  F1 Score: {final_scores['f1']:.4f}")
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
    parser = argparse.ArgumentParser(description="Evaluate stacked LoRA model for Citation task")
    parser.add_argument("--merged_base_model_path", type=str, required=True, help="Path to merged base model")
    parser.add_argument("--stacked_lora_path", type=str, required=True, help="Path to stacked LoRA model")
    parser.add_argument("--eval_data_file", type=str, required=True, help="Path to evaluation data JSON file")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value for fusion")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save results")
    
    args = parser.parse_args()

    model, tokenizer = load_stacked_lora_model_and_tokenizer(
        args.merged_base_model_path, args.stacked_lora_path
    )
    if model is None or tokenizer is None:
        logger.error("Model loading failed")
        return

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
    metrics = calculate_metrics(evaluation_results)
    print_result_samples(evaluation_results)

    user_id_match = re.search(r'eval_data_(\d+)\.json', os.path.basename(args.eval_data_file))
    user_id = user_id_match.group(1) if user_id_match else "unknown_user"
    
    output_filename = f"results_{user_id}_alpha_{args.alpha}.json"
    output_path = os.path.join(args.results_dir, output_filename)

    # Compatible format for aggregate_results.py
    final_output = {
        "user_id": user_id,
        "metrics": metrics,
        "total_samples": len(eval_data),
        "predictions": [
            {
                "prediction": res["prediction"],
                "ground_truth": res["ground_truth"]
            } for res in evaluation_results
        ],
        "detailed_predictions_original": evaluation_results,
        "alpha_value": args.alpha,
        "stacked_lora_id": os.path.basename(os.path.dirname(args.stacked_lora_path)),
        "eval_data_file": args.eval_data_file,
        "stacked_lora_path": args.stacked_lora_path,
        "merged_base_model_path": args.merged_base_model_path,
        "evaluation_timestamp": datetime.now().isoformat()
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"Citation evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()