#!/usr/bin/env python3
"""
Stacked LoRA model evaluation script for Scholarly Title task using ROUGE scores.
"""
import os
import json
import torch
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from rouge_score import rouge_scorer, scoring
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_evaluation(model, tokenizer, eval_data):
    """Run evaluation on data and collect predictions"""
    results = []
    for sample in tqdm(eval_data, desc="Evaluating samples", leave=False):
        instruction = sample.get('instruction', '')
        ground_truth = sample.get('output', '')

        if not instruction or not ground_truth:
            continue

        try:
            inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode output skipping prompt part
            raw_prediction = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
            
            # Simple processing: truncate at newline and period, take first part
            truncated_prediction = raw_prediction.split('\n')[0].split('.')[0].strip()

            results.append({
                "instruction": instruction,
                "raw_prediction": raw_prediction,
                "prediction": truncated_prediction,
                "ground_truth": ground_truth
            })
        except Exception as e:
            logger.warning(f"Error processing sample: {e}")
            results.append({
                "instruction": instruction,
                "raw_prediction": "",
                "prediction": "",
                "ground_truth": ground_truth
            })
    
    return results

def calculate_rouge_scores(results):
    """Calculate ROUGE scores"""
    if not results:
        return {}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for item in results:
        prediction = item['prediction']
        target = item['ground_truth']
        if not prediction:
            prediction = ""
        scores = scorer.score(target, prediction)
        aggregator.add_scores(scores)

    aggregated_scores = aggregator.aggregate()
    
    final_scores = {
        key: {
            'precision': score.mid.precision,
            'recall': score.mid.recall,
            'fmeasure': score.mid.fmeasure
        }
        for key, score in aggregated_scores.items()
    }
    
    return final_scores

def save_json(data: dict, filepath: str):
    """Save dictionary to JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Results saved to: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save file to {filepath}: {e}")

def print_result_samples(results, num_samples=5):
    """Print sample predictions and ground truth comparison"""
    logger.info("\n--- Sample Results ---\n")
    if not results:
        logger.info("No results to display.")
        return
        
    for i, item in enumerate(results[:num_samples]):
        logger.info(f"--- Sample {i+1} ---")
        logger.info(f"  [Instruction]: {item['instruction']}")
        logger.info(f"  [Raw Prediction]: {item['raw_prediction']}")
        logger.info(f"  [Processed Prediction]: {item['prediction']}")
        logger.info(f"  [Ground Truth]: {item['ground_truth']}")
        logger.info("-" * 50)
    logger.info("\n-----------------------------------------\n")

def main(args):
    """Main function to evaluate stacked LoRA model."""
    logger.info("Starting stacked LoRA model evaluation for Scholarly Title task...")
    
    if args.merged_base_model_path:
        logger.info(f"Using merged base model: {args.merged_base_model_path}")
        
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(args.merged_base_model_path)
            merged_model = AutoModelForCausalLM.from_pretrained(
                args.merged_base_model_path,
                quantization_config=quantization_config,
                device_map="auto"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            logger.info("Merged model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load merged model: {e}")
            return
            
    else:
        if not args.fused_lora_path:
            logger.error("Traditional mode requires --fused_lora_path parameter")
            return
            
        logger.info(f"Using traditional mode: {args.base_model_path} + {args.fused_lora_path}")
        
        logger.info(f"Loading base model from {args.base_model_path}...")
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model_path,
                quantization_config=quantization_config,
                device_map="auto"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            logger.info("Base model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            return

        logger.info(f"Loading and merging first stage LoRA from {args.fused_lora_path}...")
        try:
            fused_model = PeftModel.from_pretrained(base_model, args.fused_lora_path)
            merged_model = fused_model.merge_and_unload()
            logger.info("First stage LoRA merged successfully")
        except Exception as e:
            logger.error(f"Failed to load or merge first stage LoRA: {e}")
            return

    logger.info(f"Loading second stage LoRA from {args.stacked_lora_path}...")
    try:
        final_model = PeftModel.from_pretrained(merged_model, args.stacked_lora_path)
        final_model.eval()
        logger.info("Second stage LoRA loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load second stage LoRA: {e}")
        return

    logger.info(f"Loading evaluation data: {args.eval_data_file}")
    try:
        with open(args.eval_data_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        logger.info(f"Successfully loaded {len(eval_data)} samples")
    except Exception as e:
        logger.error(f"Failed to load evaluation data: {e}")
        return

    logger.info("Starting evaluation...")
    try:
        evaluation_results = run_evaluation(final_model, tokenizer, eval_data)
        rouge_metrics = calculate_rouge_scores(evaluation_results)
        
        user_id = os.path.basename(args.eval_data_file).replace('eval_data_', '').replace('.json', '')
        stacked_lora_id = os.path.basename(os.path.dirname(args.stacked_lora_path))
        
        description = f"Stacked LoRA model {stacked_lora_id} evaluation on user {user_id} data (Scholarly Title)"
        
        final_metrics = {
            'description': description,
            'user_id': user_id,
            'alpha_value': args.alpha,
            'stacked_lora_id': stacked_lora_id,
            'eval_data_file': args.eval_data_file,
            'fused_lora_path': args.fused_lora_path if hasattr(args, 'fused_lora_path') else None,
            'stacked_lora_path': args.stacked_lora_path,
            'base_model_path': args.base_model_path if hasattr(args, 'base_model_path') else args.merged_base_model_path,
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': rouge_metrics,
            'rouge_metrics': rouge_metrics,
            'total_samples': len(evaluation_results),
            'detailed_results': evaluation_results
        }
        
        logger.info("Evaluation completed")
        print_result_samples(evaluation_results)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return

    os.makedirs(args.results_dir, exist_ok=True)
    result_filename = f"results_{user_id}_alpha_{args.alpha}.json"
    result_filepath = os.path.join(args.results_dir, result_filename)
    save_json(final_metrics, result_filepath)
    
    logger.info("Stacked LoRA model evaluation finished for Scholarly Title task")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate stacked LoRA model for Scholarly Title generation task")
    parser.add_argument("--base_model_path", type=str, default="./model/Meta-Llama-3-8B-Instruct", help="Path to base model (for traditional mode)")
    parser.add_argument("--fused_lora_path", type=str, help="Path to first stage fused LoRA model (for traditional mode)")
    parser.add_argument("--merged_base_model_path", type=str, help="Path to merged base model with fused LoRA (new mode)")
    parser.add_argument("--stacked_lora_path", type=str, required=True, help="Path to second stage stacked LoRA model")
    parser.add_argument("--eval_data_file", type=str, required=True, help="Path to evaluation data JSON file")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value for fusion")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save evaluation results")
    
    args = parser.parse_args()
    main(args) 