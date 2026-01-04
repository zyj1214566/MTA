#!/usr/bin/env python3
"""
Evaluation script for News Headline task using ROUGE scores.
"""
import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import signal
import time
from rouge_score import rouge_scorer

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def load_stacked_lora_model_and_tokenizer(merged_base_model_path, stacked_lora_path):
    """Load merged base model and stacked LoRA adapter"""
    print(f"Loading merged base model from {merged_base_model_path}...")
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading tokenizer from {merged_base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(merged_base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading stacked LoRA from: {stacked_lora_path}")
    final_model = PeftModel.from_pretrained(merged_model, stacked_lora_path)
    final_model.eval()

    print("Model and tokenizer loaded successfully.")
    return final_model, tokenizer

def run_evaluation(model, tokenizer, eval_data):
    """Run evaluation on data and collect predictions"""
    results = []
    for sample in tqdm(eval_data, desc="Evaluating samples"):
        instruction = sample.get('instruction', '')
        ground_truth = sample.get('output', '')

        if not instruction or not ground_truth:
            continue

        inputs = tokenizer(instruction, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        
        # Set generation timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode output skipping prompt part
            raw_prediction = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
            
        except TimeoutError:
            print("Generation timeout, using empty string as prediction")
            raw_prediction = ""
        except Exception as e:
            print(f"Generation error: {e}, using empty string as prediction")
            raw_prediction = ""
        finally:
            signal.alarm(0)  # Cancel timeout
        
        # Post-processing for news_headline task
        prediction = raw_prediction

        # Truncate at newline, take first line
        lines = prediction.split('\n')
        if lines:
            prediction = lines[0]
        
        prediction = prediction.strip()

        results.append({
            "instruction": instruction,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "raw_output": raw_prediction
        })
    return results

def calculate_rouge_scores(results):
    """Calculate ROUGE scores"""
    if not results:
        return {}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rougeL_scores = []
    
    for item in results:
        prediction = item['prediction']
        ground_truth = item['ground_truth']
        
        # Set ROUGE calculation timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60 second timeout
        
        try:
            scores = scorer.score(ground_truth, prediction)
            rouge1_scores.append(scores['rouge1'])
            rougeL_scores.append(scores['rougeL'])
        except TimeoutError:
            print("ROUGE calculation timeout, using default scores")
            from rouge_score.scoring import Score
            rouge1_scores.append(Score(precision=0.0, recall=0.0, fmeasure=0.0))
            rougeL_scores.append(Score(precision=0.0, recall=0.0, fmeasure=0.0))
        except Exception as e:
            print(f"ROUGE calculation error: {e}, using default scores")
            from rouge_score.scoring import Score
            rouge1_scores.append(Score(precision=0.0, recall=0.0, fmeasure=0.0))
            rougeL_scores.append(Score(precision=0.0, recall=0.0, fmeasure=0.0))
        finally:
            signal.alarm(0)  # Cancel timeout
    
    # Calculate average scores
    def average_scores(scores_list):
        if not scores_list:
            return {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
        
        avg_precision = sum(score.precision for score in scores_list) / len(scores_list)
        avg_recall = sum(score.recall for score in scores_list) / len(scores_list)
        avg_fmeasure = sum(score.fmeasure for score in scores_list) / len(scores_list)
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'fmeasure': avg_fmeasure
        }
    
    final_scores = {
        'rouge1': average_scores(rouge1_scores),
        'rougeL': average_scores(rougeL_scores)
    }
    
    print("\n--- ROUGE Metrics Summary ---")
    for metric, values in final_scores.items():
        print(f"  {metric.upper()}:")
        print(f"    F1-Score: {values['fmeasure']:.4f}")
        print(f"    Precision: {values['precision']:.4f}")
        print(f"    Recall: {values['recall']:.4f}")
    print("-----------------------------\n")

    return final_scores

def print_result_samples(results, num_samples=5):
    """Print sample predictions and ground truth comparison"""
    print("\n--- Sample Predictions vs. Ground Truth ---\n")
    if not results:
        print("No results to display.")
        return
        
    for i, item in enumerate(results[:num_samples]):
        print(f"--- Sample {i+1} ---")
        print(f"  [Raw Output]: {item.get('raw_output', 'N/A')}")
        print(f"  [Processed Prediction]: {item['prediction']}")
        print(f"  [Ground Truth]: {item['ground_truth']}")
        print("-" * (len(f"--- Sample {i+1} ---")))
    print("\n-----------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model for news_headline task using ROUGE scores.")
    parser.add_argument("--merged_base_model_path", required=True, help="Path to merged base model")
    parser.add_argument("--stacked_lora_path", required=True, help="Path to stacked LoRA model")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value for fusion")
    parser.add_argument("--eval_data_file", required=True, help="Path to the evaluation data JSON file.")
    parser.add_argument("--results_dir", required=True, help="Directory to save the evaluation results.")
    
    args = parser.parse_args()

    model, tokenizer = load_stacked_lora_model_and_tokenizer(args.merged_base_model_path, args.stacked_lora_path)
    with open(args.eval_data_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    evaluation_results = run_evaluation(model, tokenizer, eval_data)
    rouge_metrics = calculate_rouge_scores(evaluation_results)
    print_result_samples(evaluation_results)

    os.makedirs(args.results_dir, exist_ok=True)
    eval_filename = os.path.basename(args.eval_data_file)
    user_id = ''.join(filter(str.isdigit, eval_filename))
    if not user_id:
        user_id = "unknown_user"
        
    output_filename = f"result_user_{user_id}.json"
    output_path = os.path.join(args.results_dir, output_filename)

    final_output = {
        "user_id": user_id,
        "merged_base_model_path": args.merged_base_model_path,
        "stacked_lora_path": args.stacked_lora_path,
        "alpha": args.alpha,
        "metrics": rouge_metrics,
        "rouge_metrics": rouge_metrics,
        "predictions": evaluation_results,
        "total_samples": len(evaluation_results)
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to {output_path}")

if __name__ == "__main__":
    main() 