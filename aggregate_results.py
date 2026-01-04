#!/usr/bin/env python3
"""
Universal experiment results aggregation script.
"""
import os
import json
import sys
import pandas as pd
import argparse
import glob
from datetime import datetime
from collections import defaultdict
from typing import Optional
import numpy as np

# Try to import sklearn for classification metrics
try:
    from sklearn.metrics import accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def aggregate_results(results_dir: str, output_file: Optional[str] = None, task: Optional[str] = None):
    """Aggregate all experiment results in the directory."""
    print(f"Starting results aggregation...")
    print(f"Results directory: {results_dir}")

    if not os.path.isdir(results_dir):
        print(f"ERROR: Directory not found {results_dir}")
        sys.exit(1)


    if results_dir is None:
        print(f"ERROR: Results directory path cannot be None.")
        sys.exit(1)
    result_files = glob.glob(os.path.join(results_dir, "*.json"))

    result_files = [f for f in result_files if "aggregated_results" not in os.path.basename(f)]

    if not result_files:
        print("No result files found for aggregation. Expected file format: *.json")
        sys.exit(0)

    print(f"Found {len(result_files)} result files for aggregation.")

    all_results = []
    for filepath in result_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                

                if 'metrics' in data:

                    all_results.append(data)
                elif any(key in data for key in ['accuracy_percent', 'mae_on_all_samples', 'rouge1', 'rouge-1']):

                    wrapped_data = {
                        'metrics': {},
                        'total_samples': data.get('total_samples', 1),
                        'user_id': data.get('user_id', 'unknown'),
                        'model_id': data.get('model_id', 'unknown'),
                        'evaluation_timestamp': data.get('evaluation_timestamp', ''),
                        'description': data.get('description', '')
                    }
                    

                    for key in ['accuracy_percent', 'mae_on_all_samples', 'rmse_on_all_samples', 
                               'rouge1', 'rouge-1', 'rougeL', 'rouge-L', 'macro_f1', 'accuracy']:
                        if key in data:
                            wrapped_data['metrics'][key] = data[key]
                    

                    if 'predictions' in data:
                        wrapped_data['predictions'] = data['predictions']
                    
                    all_results.append(wrapped_data)
                else:
                    print(f"WARNING: File {os.path.basename(filepath)} missing valid metric fields, skipped.")
        except Exception as e:
            print(f"WARNING: Error processing file {os.path.basename(filepath)}: {e}. Skipped.")

    if not all_results:
        print("Could not load any valid metrics. Exiting.")
        sys.exit(1)


    sum_metrics = defaultdict(float)

    total_samples = 0

    total_correct_predictions = 0
    total_predictions = 0

    total_correct_samples = 0
    
    # For classification tasks: collect all predictions and ground truths
    all_predictions = []
    all_ground_truths = []
    

    first_metrics = all_results[0]['metrics']
    is_rouge_experiment = 'rouge1' in first_metrics or 'rouge-1' in first_metrics
    is_mae_experiment = 'mae_on_all_samples' in first_metrics or 'mae' in first_metrics
    is_cls_experiment = 'accuracy' in first_metrics and ('macro_f1' in first_metrics or 'f1' in first_metrics)

    for result in all_results:
        metrics = result['metrics']
        
        if is_rouge_experiment:
            # Use sample-weighted averaging for ROUGE scores
            samples = result.get('total_samples', 1)
            total_samples += samples
            
            for rouge_key in ['rouge1', 'rouge-1', 'rougeL', 'rouge-L']:
                if rouge_key in metrics:
                    for sub_key in ['fmeasure', 'precision', 'recall']:
                        # Weight by sample count (sample-weighted averaging)
                        score = metrics[rouge_key].get(sub_key, 0)
                        sum_metrics[f"weighted_{rouge_key}_{sub_key}"] += score * samples
        
        elif is_mae_experiment:

            samples = result.get('total_samples', 1)
            total_samples += samples
            

            mae = metrics.get('mae_on_all_samples', metrics.get('mae', 0))
            accuracy = metrics.get('accuracy_percent', 0)
            rmse = metrics.get('rmse_on_all_samples', metrics.get('rmse', 0))
            
            sum_metrics['weighted_mae'] += mae * samples

            correct_samples = (accuracy / 100.0) * samples
            total_correct_samples += correct_samples

            sum_metrics['squared_error_sum'] += (rmse ** 2) * samples
        elif is_cls_experiment:
            # Collect all predictions and ground truths for proper F1 calculation
            if 'predictions' in result and isinstance(result['predictions'], list):
                for pred_item in result['predictions']:
                    prediction = str(pred_item.get('prediction', '')).strip()
                    ground_truth = str(pred_item.get('ground_truth', '')).strip()
                    
                    if prediction and ground_truth:  # Ensure labels are not empty
                        all_predictions.append(prediction)
                        all_ground_truths.append(ground_truth)
                        total_predictions += 1
                        if prediction == ground_truth:
                            total_correct_predictions += 1


    avg_metrics = {}
    if is_rouge_experiment and total_samples > 0:
        # Calculate sample-weighted averages for ROUGE
        for rouge_key in ['rouge1', 'rouge-1', 'rougeL', 'rouge-L']:
            for sub_key in ['fmeasure', 'precision', 'recall']:
                weighted_key = f"weighted_{rouge_key}_{sub_key}"
                final_key = f"{rouge_key}_{sub_key}"
                if weighted_key in sum_metrics:
                    avg_metrics[final_key] = round(sum_metrics[weighted_key] / total_samples, 4)
    elif is_mae_experiment and total_samples > 0:
        avg_metrics['mean_mae'] = round(sum_metrics['weighted_mae'] / total_samples, 4)

        avg_metrics['mean_accuracy_percent'] = round((total_correct_samples / total_samples) * 100, 2)

        if sum_metrics['squared_error_sum'] > 0:
            avg_metrics['mean_rmse'] = round((sum_metrics['squared_error_sum'] / total_samples) ** 0.5, 4)
        else:
            avg_metrics['mean_rmse'] = 0.0
    elif is_cls_experiment and all_predictions:
        # Use sklearn for proper F1 calculation
        if SKLEARN_AVAILABLE:
            # Calculate micro-averaged accuracy (same as overall accuracy)
            avg_metrics['accuracy'] = round(accuracy_score(all_ground_truths, all_predictions), 4)
            
            # Calculate macro-averaged F1 score
            avg_metrics['macro_f1'] = round(f1_score(all_ground_truths, all_predictions, average='macro', zero_division=0), 4)
        else:
            # Fallback to simple accuracy calculation
            if total_predictions > 0:
                avg_metrics['accuracy'] = round(total_correct_predictions / total_predictions, 4)
            else:
                avg_metrics['accuracy'] = 0.0
            print("WARNING: sklearn not available, macro_f1 calculation skipped")


    final_report = {
        "metadata": {
            "aggregation_timestamp": datetime.now().isoformat(),
            "source_directory": results_dir,
            "total_users_processed": len(all_results)
        },
        "aggregated_metrics": avg_metrics,
        "individual_results": all_results
    }
    

    print("\n" + "="*80)
    print("Final Experiment Summary Statistics")
    print("="*80)
    print(f"Number of users: {len(all_results)}")
    if is_mae_experiment:
        print(f"Total samples: {int(total_samples)}")

    print("\n--- Average Metrics ---")
    if not avg_metrics:
        print("  Could not calculate any average metrics.")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value}")
    print("="*80)


    if output_file is None:
        output_file = os.path.join(results_dir, "aggregated_results.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    
    print(f"\nAggregation completed! Final report saved to: {output_file}")
    print("\nScript execution finished!")

def main():
    parser = argparse.ArgumentParser(description="Universal experiment results aggregation script")
    parser.add_argument("--results_dir", required=True, help="Experiment results directory path")
    parser.add_argument("--output_file", default=None, help="Output file path (optional, default: results_dir/aggregated_results.json)")
    parser.add_argument("--task", type=str, help="Optional task name for specifying metric aggregation logic.")
    parser.add_argument("--task_name", type=str, help="Task name (equivalent to --task, for compatibility)")
    
    args = parser.parse_args()
    

    task = args.task or args.task_name
    
    aggregate_results(args.results_dir, args.output_file, task)

if __name__ == "__main__":
    main()