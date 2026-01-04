
import os
import shutil
import torch
import logging
import argparse
from typing import Dict
from safetensors.torch import load_file, save_file
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_adapter_path(model_path: str) -> str:
    """Finds the path to the adapter weights file (.safetensors or .bin)."""
    safetensors_path = os.path.join(model_path, "adapter_model.safetensors")
    bin_path = os.path.join(model_path, "adapter_model.bin")
    if os.path.exists(safetensors_path):
        return safetensors_path
    elif os.path.exists(bin_path):
        return bin_path
    else:
        raise FileNotFoundError(f"Could not find adapter_model.safetensors or adapter_model.bin in {model_path}")

def load_lora_weights(file_path: str):
    """Loads LoRA weights from either .safetensors or .bin file."""
    if file_path.endswith(".safetensors"):
        return load_file(file_path, device="cpu")
    else:
        return torch.load(file_path, map_location="cpu")

def save_lora_weights(weights, file_path: str):
    """Saves LoRA weights to either .safetensors or .bin file."""
    if file_path.endswith(".safetensors"):
        save_file(weights, file_path)
    else:
        torch.save(weights, file_path)



def main(args):
    """
    Main function to merge LoRA models using weighted interpolation.
    """
    lora_path1 = args.lora_path1
    lora_path2 = args.lora_path2
    output_path = args.output_path
    alpha = args.alpha

    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")

    logger.info("Starting LoRA merge process...")
    logger.info(f"LoRA 1: {lora_path1} (weight: {1-alpha:.2f})")
    logger.info(f"LoRA 2: {lora_path2} (weight: {alpha:.2f})")
    
    # --- 1. Load weights for both models ---
    try:
        adapter_file1 = find_adapter_path(lora_path1)
        adapter_file2 = find_adapter_path(lora_path2)
        
        weights1 = load_lora_weights(adapter_file1)
        weights2 = load_lora_weights(adapter_file2)
        
        # Determine the file extension for saving
        output_ext = ".safetensors" if adapter_file1.endswith(".safetensors") else ".bin"

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return

    # --- 2. Merge the weights ---
    merged_weights = {}
    all_keys = set(weights1.keys()) | set(weights2.keys())

    logger.info("Merging weights...")
    for key in tqdm(all_keys, desc="Merging layers"):
        w1 = weights1.get(key)
        w2 = weights2.get(key)

        if w1 is not None and w2 is not None:
            merged_weights[key] = (1 - alpha) * w1 + alpha * w2
        elif w1 is not None:
            merged_weights[key] = w1
        elif w2 is not None:
            merged_weights[key] = w2
            
    logger.info("Weight merge complete.")

    # --- 3. Save the merged model ---
    if not output_path:
        logger.error("Error: --output_path is required.")
        return

    logger.info(f"Saving merged LoRA to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    # Copy configuration files from the first LoRA path
    for filename in ["adapter_config.json", "README.md", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"]:
        src_file = os.path.join(lora_path1, filename)
        if os.path.exists(src_file):
            shutil.copy2(src_file, output_path)
    
    output_weights_file = os.path.join(output_path, f"adapter_model{output_ext}")
    save_lora_weights(merged_weights, output_weights_file)
    logger.info("Merged LoRA saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two LoRA models using weighted interpolation.")
    parser.add_argument("--lora_path1", type=str, required=True, help="Path to the first LoRA model directory.")
    parser.add_argument("--lora_path2", type=str, required=True, help="Path to the second LoRA model directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the merged LoRA model.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for the second LoRA model (0.0 to 1.0). First model gets weight (1-alpha).")
    
    args = parser.parse_args()
    main(args) 