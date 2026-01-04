#!/bin/bash

# ==============================================================================
#                    MTA Hybrid Experiment Workflow
# ==============================================================================
#
# This script executes a complete hybrid strategy experiment for MTA framework:
# 1. Parse matches_results.json and calculate fusion weights (alpha)
# 2. For each user:
#    - LoRA Fusion: Weighted merge of anchor LoRAs using simple_lora_merger.py
#    - Data Preparation: Create training and evaluation datasets
#    - Fine-tuning: Use LLaMA-Factory for additional training
#    - Evaluation: Task-specific evaluation
# 3. Aggregate all results
#
# Usage: bash run_experiment.sh <task_name> <num_users>
# Example: bash run_experiment.sh movie_tagging 72
#
# ==============================================================================

set -e # Exit immediately if any command fails

# ==============================================================================
#                            Parameter Setup
# ==============================================================================

if [ $# -ne 2 ]; then
    echo "Usage: $0 <task_name> <num_users>"
    echo "Example: $0 movie_tagging 72"
    echo "Supported tasks: movie_tagging, news_headline, product_rating, citation, scholarly_title"
    exit 1
fi

TASK_NAME=$1
NUM_USERS=$2

echo "Starting MTA Hybrid Experiment: $TASK_NAME ($NUM_USERS users)"

# ==============================================================================
#                            Path Configuration
# ==============================================================================

# Base paths
BASE_MODEL_PATH="../model/Meta-Llama-3-8B-Instruct"
BASE_DIR=$(pwd)
LLAMA_FACTORY_DIR="./LLaMA-Factory"

# Task-specific paths
FEW_SHOT_DATA_FILE="./data/${TASK_NAME}/test_100/test_100.json"
LORA_BASE_PATH="${LLAMA_FACTORY_DIR}/saves/llama3-8b/lora/${TASK_NAME}_k1"
MATCH_FILE="./dense_retrieval_results/${TASK_NAME}/matches_results.json"

# LLaMA-Factory related paths
LLAMA_FACTORY_DATA_DIR="${LLAMA_FACTORY_DIR}/data"
DATASET_INFO_JSON="${LLAMA_FACTORY_DATA_DIR}/dataset_info.json"

# Experiment parameters
STRATEGY="hybrid_${TASK_NAME}"

# Script paths - use task-specific names
CREATE_TRAINING_SCRIPT="./data/${TASK_NAME}/create_training_data.py"
CREATE_EVAL_SCRIPT="./data/${TASK_NAME}/create_eval_data.py"
MERGE_SCRIPT="./simple_lora_merger.py"
EVALUATE_SCRIPT="./eval/${TASK_NAME}/evaluate_${TASK_NAME}.py"
AGGREGATE_SCRIPT="./aggregate_results.py"

# Output directories
MERGE_LORA_DIR="./merge_lora/${STRATEGY}"
TRAINING_DATA_CACHE_DIR="./data/${TASK_NAME}/training_data_cache_for_factory"
EVAL_DATA_CACHE_DIR="./data/${TASK_NAME}/eval_data_cache"
ADAPT_LORA_BASE_DIR="./adapt_lora/${STRATEGY}"
FINAL_EVAL_RESULTS_DIR="./final_eval_results/${STRATEGY}"

# Create necessary directories
mkdir -p "$MERGE_LORA_DIR"
mkdir -p "$TRAINING_DATA_CACHE_DIR"
mkdir -p "$EVAL_DATA_CACHE_DIR"
mkdir -p "$ADAPT_LORA_BASE_DIR"
mkdir -p "$FINAL_EVAL_RESULTS_DIR"

# ==============================================================================
#                            Data Preprocessing
# ==============================================================================

echo "Step 1: Data Preprocessing"
TEMP_CSV_INFO_FILE="${BASE_DIR}/user_fusion_info_${TASK_NAME}.csv"

# Parse matches_results.json and generate CSV with fusion weights
python3 << EOF
import json
import sys

try:
    with open('$MATCH_FILE', 'r', encoding='utf-8') as f:
        matches_data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(matches_data, dict):
        matches = list(matches_data.values())
    elif isinstance(matches_data, list):
        matches = matches_data
    else:
        raise TypeError('Unsupported matches_results.json format.')

    with open('$TEMP_CSV_INFO_FILE', 'w') as out_f:
        for i, user_data in enumerate(matches):
            if i >= $NUM_USERS:
                break
            
            few_shot_user_id = user_data['few_shot_user_id']

            # Parse matches format
            if 'matches' in user_data and len(user_data['matches']) >= 2:
                match1, match2 = user_data['matches'][0], user_data['matches'][1]
                anchor1_id = match1['anchor_user_id']
                anchor2_id = match2['anchor_user_id']
                score1 = float(match1['similarity_score'])
                score2 = float(match2['similarity_score'])
            else:
                print(f'Warning: Insufficient match info for user {few_shot_user_id}, skipping.', file=sys.stderr)
                continue

            # Calculate fusion weight alpha
            if (score1 + score2) == 0:
                alpha = 0.5
            else:
                alpha = score1 / (score1 + score2)
            
            out_f.write(f'{few_shot_user_id},{anchor1_id},{anchor2_id},{alpha:.6f}\\n')

    print(f'Generated fusion info for {min(len(matches), $NUM_USERS)} users')

except FileNotFoundError:
    print(f'Error: Match file not found: $MATCH_FILE', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'Error processing match file: {e}', file=sys.stderr)
    sys.exit(1)
EOF



# ==============================================================================
#                            Main Processing Loop
# ==============================================================================

echo "Step 2: Processing Users"

# Counters
success_count=0
fail_count=0
eval_success_count=0
eval_fail_count=0

while IFS=',' read -r few_shot_user_id anchor1_id anchor2_id alpha; do
    echo "Processing User: $few_shot_user_id"

    # --- Stage 1: LoRA Fusion ---
    lora_path1="${LORA_BASE_PATH}/sft_user_${anchor1_id}"
    lora_path2="${LORA_BASE_PATH}/sft_user_${anchor2_id}"
    fused_lora_path="${MERGE_LORA_DIR}/merge_lora_${few_shot_user_id}"

    if [ ! -d "$lora_path1" ] || [ ! -d "$lora_path2" ]; then
        echo "LoRA paths not found for user $few_shot_user_id"
        fail_count=$((fail_count + 1))
        continue
    fi

    if python3 "$MERGE_SCRIPT" \
        --lora_path1 "$lora_path1" \
        --lora_path2 "$lora_path2" \
        --output_path "$fused_lora_path" \
        --alpha "$alpha"; then
        echo "LoRA fusion successful for user $few_shot_user_id"
    else
        echo "LoRA fusion failed for user $few_shot_user_id"
        fail_count=$((fail_count + 1))
        continue
    fi

    # --- Stage 2: Merge Fused LoRA with Base Model ---
    merge_model_path="${ADAPT_LORA_BASE_DIR}/merge_model_${few_shot_user_id}"
    
    # Convert to absolute path before changing directory
    fused_lora_absolute_path=$(realpath "$fused_lora_path")
    merge_model_absolute_path=$(realpath "$merge_model_path")
    llama_factory_data_absolute_path=$(realpath "$LLAMA_FACTORY_DATA_DIR")
    dataset_info_absolute_path=$(realpath "$DATASET_INFO_JSON")
    
    cd "$LLAMA_FACTORY_DIR" || exit
    if llamafactory-cli export \
        --model_name_or_path "$BASE_MODEL_PATH" \
        --adapter_name_or_path "$fused_lora_absolute_path" \
        --export_dir "$merge_model_absolute_path" \
        --export_size 2 \
        --export_device cpu; then

        cd "$BASE_DIR" || exit
    else
        echo "Model merge failed for user $few_shot_user_id"
        cd "$BASE_DIR" || exit
        fail_count=$((fail_count + 1))
        continue
    fi

    # --- Stage 3: Prepare LLaMA-Factory Dataset ---
    user_training_data_cache_file="${TRAINING_DATA_CACHE_DIR}/training_data_${few_shot_user_id}.json"
    
    # Create training data
    python3 "$CREATE_TRAINING_SCRIPT" \
        --user_id "$few_shot_user_id" \
        --input_file "$FEW_SHOT_DATA_FILE" \
        --output_file "$user_training_data_cache_file"
    
    # Copy training data to Factory data directory
    factory_data_filename="stacked_lora_train_${few_shot_user_id}.json"
    cp "$user_training_data_cache_file" "${llama_factory_data_absolute_path}/${factory_data_filename}"
    
    # Backup original dataset_info.json
    cp "$dataset_info_absolute_path" "${dataset_info_absolute_path}.bak"
    
    # Register dataset dynamically
    dataset_name="stacked_lora_train_${few_shot_user_id}"
    python3 -c "
import json
dataset_info_path = '$dataset_info_absolute_path'
dataset_name = '$dataset_name'
file_name = '$factory_data_filename'
with open(dataset_info_path, 'r') as f:
    info = json.load(f)
info[dataset_name] = {'file_name': file_name, 'formatting': 'alpaca'}
with open(dataset_info_path, 'w') as f:
    json.dump(info, f, indent=2)
"


    # --- Stage 4: Stacked LoRA Fine-tuning with LLaMA-Factory ---
    adapt_lora_dir="${ADAPT_LORA_BASE_DIR}/adapt_lora_${few_shot_user_id}"
    adapt_lora_absolute_path=$(realpath "$adapt_lora_dir")
    
    cd "$LLAMA_FACTORY_DIR" || exit
    if llamafactory-cli train \
        --stage sft \
        --do_train \
        --model_name_or_path "$merge_model_absolute_path" \
        --dataset "$dataset_name" \
        --dataset_dir "$llama_factory_data_absolute_path" \
        --finetuning_type lora \
        --lora_target all \
        --lora_rank 4 \
        --output_dir "$adapt_lora_absolute_path" \
        --overwrite_output_dir \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate 5e-5 \
        --num_train_epochs 1 \
        --plot_loss \
        --bf16; then

        cd "$BASE_DIR" || exit
        success_count=$((success_count + 1))
    else
        echo "Stacked LoRA fine-tuning failed for user $few_shot_user_id"
        cd "$BASE_DIR" || exit
        fail_count=$((fail_count + 1))
        
        # Cleanup and continue
        rm -f "${llama_factory_data_absolute_path}/${factory_data_filename}"
        cp "${dataset_info_absolute_path}.bak" "$dataset_info_absolute_path"
        continue
    fi
    
    # Cleanup temporary dataset
    rm -f "${llama_factory_data_absolute_path}/${factory_data_filename}"
    cp "${dataset_info_absolute_path}.bak" "$dataset_info_absolute_path"

    # --- Stage 5: Evaluation ---
    eval_data_file="${EVAL_DATA_CACHE_DIR}/eval_data_${few_shot_user_id}.json"
    
    # Create evaluation data
    python3 "$CREATE_EVAL_SCRIPT" \
        --user_id "$few_shot_user_id" \
        --input_file "$FEW_SHOT_DATA_FILE" \
        --output_file "$eval_data_file"
    
    # Stacked LoRA evaluation
    stacked_lora_model_path="$adapt_lora_absolute_path"
    
    if python3 "$EVALUATE_SCRIPT" \
        --merged_base_model_path "$merge_model_absolute_path" \
        --stacked_lora_path "$stacked_lora_model_path" \
        --eval_data_file "$eval_data_file" \
        --alpha "$alpha" \
        --results_dir "$FINAL_EVAL_RESULTS_DIR"; then

        eval_success_count=$((eval_success_count + 1))
    else
        echo "Evaluation failed for user $few_shot_user_id"
        eval_fail_count=$((eval_fail_count + 1))
    fi
    
    # Cleanup merged base model to save space
    if [ -d "$merge_model_absolute_path" ]; then
        rm -rf "$merge_model_absolute_path"
    fi

    success_count=$((success_count + 1))

done < "$TEMP_CSV_INFO_FILE"

# ==============================================================================
#                            Results Aggregation
# ==============================================================================

echo "Step 3: Results Aggregation"

if python3 "$AGGREGATE_SCRIPT" \
    --results_dir "$FINAL_EVAL_RESULTS_DIR" \
    --output_file "${FINAL_EVAL_RESULTS_DIR}/final_aggregated_results_${TASK_NAME}.json" \
    --task "$TASK_NAME"; then
    echo "Results saved to: ${FINAL_EVAL_RESULTS_DIR}/final_aggregated_results_${TASK_NAME}.json"
else
    echo "Results aggregation failed"
fi

# ==============================================================================
#                            Final Summary
# ==============================================================================

echo "Experiment Summary - Task: $TASK_NAME"
echo "Users processed: $success_count, Failed: $fail_count"
echo "Evaluations successful: $eval_success_count, Failed: $eval_fail_count"

# ==============================================================================
#                            Cleanup and Space Management
# ==============================================================================

# Cleanup temporary files
rm -f "$TEMP_CSV_INFO_FILE"

# Cleanup any remaining merged base models
merge_models_pattern="${ADAPT_LORA_BASE_DIR}/merge_model_*"
for merge_model in $merge_models_pattern; do
    if [ -d "$merge_model" ]; then
        rm -rf "$merge_model"
    fi
done

# Check other locations
for potential_location in "./merged_models" "./final_models/merged_base_*" "./*/merged_base_*"; do
    if [ -d "$potential_location" ]; then
        rm -rf "$potential_location"
    fi
done

echo "Experiment completed!"