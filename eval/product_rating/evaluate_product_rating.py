#!/usr/bin/env python3
"""
Stacked LoRA model evaluation script for Product Rating task
"""

import os
import json
import torch
import logging
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_on_samples(model, tokenizer, samples: list) -> list:
    """Evaluate samples with penalty mechanism for invalid predictions."""
    results = []
    if not samples:
        return results
        
    for sample in tqdm(samples, desc="Evaluating samples", leave=False):
        prompt = sample.get('instruction', '')
        true_rating_str = sample.get('output')
        
        predicted_rating = -1
        is_correct = False
        prediction_text = ""
        
        try:
            if not prompt or not true_rating_str:
                raise ValueError("Sample missing instruction or output")

            true_rating = int(true_rating_str)

            messages = [{"role": "user", "content": prompt}]
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            prompt_length = input_ids.shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=3,
                    num_return_sequences=1,
                    do_sample=False, 
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.1,
                    repetition_penalty=1.1
                )
            prediction_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
            
            try:
                predicted_rating_float = float(prediction_text)
                predicted_rating = max(1, min(5, round(predicted_rating_float)))
                is_correct = (predicted_rating == true_rating)
            except ValueError:
                pass

        except Exception as e:
            logger.warning(f"Error processing sample: {e}")

        is_penalized = False
        if predicted_rating == -1:
            is_penalized = True
            true_rating_float = float(true_rating_str)
            if abs(1 - true_rating_float) > abs(5 - true_rating_float):
                predicted_rating = 1.0
            else:
                predicted_rating = 5.0
            is_correct = False

        results.append({
            'true_rating': true_rating_str,
            'predicted_rating': predicted_rating,
            'raw_prediction': prediction_text,
            'is_correct': is_correct,
            'is_penalized': is_penalized
        })
    
    return results

def calculate_metrics(results: list, description: str = "LoRA Model") -> dict:
    """Calculate metrics from evaluation results."""
    if not results:
        return {}

    df = pd.DataFrame(results)
    accuracy = df['is_correct'].mean() * 100
    
    df['true_rating'] = pd.to_numeric(df['true_rating'])
    mae = mean_absolute_error(df['true_rating'], df['predicted_rating'])
    rmse = np.sqrt(mean_squared_error(df['true_rating'], df['predicted_rating']))
    
    penalized_count = int(df['is_penalized'].sum())
    valid_predictions_count = len(df) - penalized_count

    metrics = {
        'description': description,
        'accuracy_percent': float(accuracy),
        'mae_on_all_samples': float(mae),
        'rmse_on_all_samples': float(rmse),
        'total_samples': int(len(df)),
        'valid_prediction_samples': int(valid_predictions_count),
        'penalized_samples': int(penalized_count)
    }
    return metrics

def save_json(data: dict, filepath: str):
    """Save dictionary to JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"Results saved to: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save file to {filepath}: {e}")

def main(args):
    """Main function to evaluate stacked LoRA model."""
    logger.info("Starting stacked LoRA model evaluation...")
    
    logger.info(f"Loading merged model from {args.merged_base_model_path}...")
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

    logger.info(f"Loading stacked LoRA from {args.stacked_lora_path}...")
    try:
        final_model = PeftModel.from_pretrained(merged_model, args.stacked_lora_path)
        final_model.eval()
        logger.info("Stacked LoRA loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load stacked LoRA: {e}")
        return

    logger.info(f"Loading evaluation data: {args.eval_data_file}")
    try:
        with open(args.eval_data_file, 'r', encoding='utf-8') as f:
            validation_data = json.load(f)
        logger.info(f"Successfully loaded {len(validation_data)} samples")
    except Exception as e:
        logger.error(f"Failed to load evaluation data: {e}")
        return

    logger.info("Starting evaluation...")
    try:
        results = evaluate_on_samples(final_model, tokenizer, validation_data)
        
        user_id = os.path.basename(args.eval_data_file).replace('eval_data_', '').replace('.json', '')
        stacked_lora_id = os.path.basename(os.path.dirname(args.stacked_lora_path))
        
        description = f"Stacked LoRA model {stacked_lora_id} evaluation on user {user_id} data"
        metrics = calculate_metrics(results, description=description)
        
        metrics.update({
            'user_id': user_id,
            'alpha_value': args.alpha,
            'stacked_lora_id': stacked_lora_id,
            'eval_data_file': args.eval_data_file,
            'stacked_lora_path': args.stacked_lora_path,
            'merged_base_model_path': args.merged_base_model_path,
            'evaluation_timestamp': datetime.now().isoformat()
        })
        
        final_result = metrics.copy()
        final_result['metrics'] = {
            'accuracy_percent': metrics.get('accuracy_percent', 0),
            'mae_on_all_samples': metrics.get('mae_on_all_samples', 0),
            'rmse_on_all_samples': metrics.get('rmse_on_all_samples', 0)
        }
        
        metrics = final_result
        
        logger.info("Evaluation completed")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return

    os.makedirs(args.results_dir, exist_ok=True)
    result_filename = f"results_{user_id}_alpha_{args.alpha}.json"
    result_filepath = os.path.join(args.results_dir, result_filename)
    save_json(metrics, result_filepath)
    
    logger.info("Stacked LoRA model evaluation finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate stacked LoRA model performance")
    parser.add_argument("--merged_base_model_path", type=str, required=True, help="Path to merged base model")
    parser.add_argument("--stacked_lora_path", type=str, required=True, help="Path to stacked LoRA model")
    parser.add_argument("--eval_data_file", type=str, required=True, help="Path to evaluation data JSON file")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value for fusion")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save results")
    
    args = parser.parse_args()
    main(args) 