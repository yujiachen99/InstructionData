#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQ1: Knowledge Complexity
This script computes complexity scores for seed knowledge based on:
1. Semantic Difficulty: LLM-based complexity scoring (0-5 scale) using XCoder-Complexity-Scorer
2. Linguistic Difficulty: Perplexity of task descriptions

The final complexity score is the average of normalized semantic and linguistic difficulty.
Tasks are then partitioned into three tertiles (T1, T2, T3) based on complexity scores.
"""

import os
import math
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from complexity import Scorer


def predict_semantic_difficulty(scorer, instructions):
    """
    Predict semantic difficulty scores (0-5 scale) using XCoder-Complexity-Scorer.
    Reference: https://huggingface.co/banksy235/XCoder-Complexity-Scorer
    """
    scores = []
    for instruction in tqdm(instructions, desc="Computing semantic difficulty"):
        try:
            complexity_score = scorer.infer_complexity(instruction)
            scores.append(float(complexity_score))
        except Exception as e:
            print(f"Warning: Failed to compute complexity for instruction, using default score 2.0. Error: {e}")
            scores.append(2.0)
    
    return scores


@torch.no_grad()
def compute_linguistic_difficulty(instructions, model_name="gpt2", device="cuda", max_length=512):
    """
    Compute linguistic difficulty as perplexity of task descriptions.
    Higher perplexity indicates greater ambiguity and interpretive difficulty.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    perplexities = []
    for instr in tqdm(instructions, desc="Computing linguistic difficulty (perplexity)"):
        enc = tokenizer(
            instr, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length
        ).to(device)
        
        outputs = model(**enc, labels=enc.input_ids)
        loss = outputs.loss.item()
        perplexity = math.exp(loss)
        perplexities.append(perplexity)
    
    return perplexities


def normalize(arr):
    """Min-max normalization."""
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def partition_by_complexity(data, scores, output_dir):
    """
    Partition data into three tertiles based on complexity scores.
    T1: Lower complexity (easiest 1/3)
    T2: Middle complexity (middle 1/3)
    T3: Higher complexity (hardest 1/3)
    """
    # Sort data by complexity scores
    sorted_indices = np.argsort(scores)
    sorted_data = [data[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    
    total = len(sorted_data)
    tertile_size = total // 3
    
    # Partition into three tertiles
    tertiles = {
        "T1_lower": sorted_data[:tertile_size],
        "T2_middle": sorted_data[tertile_size:2*tertile_size],
        "T3_higher": sorted_data[2*tertile_size:]
    }
    
    tertile_scores = {
        "T1_lower": sorted_scores[:tertile_size],
        "T2_middle": sorted_scores[tertile_size:2*tertile_size],
        "T3_higher": sorted_scores[2*tertile_size:]
    }
    
    # Save partitioned datasets
    os.makedirs(output_dir, exist_ok=True)
    
    for name, subset in tertiles.items():
        output_path = os.path.join(output_dir, f"RQ1_COMPLEXITY_{name.upper()}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, indent=4, ensure_ascii=False)
        
        avg_score = np.mean(tertile_scores[name])
        print(f"Saved {len(subset)} samples to {output_path} (avg complexity: {avg_score:.3f})")
    
    # Save all data with scores for analysis
    all_data_with_scores = []
    for i, (item, score) in enumerate(zip(sorted_data, sorted_scores)):
        item_copy = item.copy()
        item_copy["complexity_score"] = float(score)
        all_data_with_scores.append(item_copy)
    
    scores_path = os.path.join(output_dir, "all_data_with_complexity_scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(all_data_with_scores, f, indent=4, ensure_ascii=False)
    print(f"Saved all data with complexity scores to {scores_path}")


def main():
    parser = argparse.ArgumentParser(description="RQ1: Compute complexity scores and partition seed knowledge")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to input JSONL file (e.g., evol_instruct.jsonl)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save partitioned datasets")
    parser.add_argument("--scoring_model", type=str, default="banksy235/XCoder-Complexity-Scorer",
                        help="Model for semantic difficulty scoring (XCoder-Complexity-Scorer)")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for faster inference")
    parser.add_argument("--perplexity_model", type=str, default="gpt2",
                        help="Model for linguistic difficulty (perplexity)")
    parser.add_argument("--instruction_key", type=str, default="instruction",
                        help="Key for instruction field in input data")
    parser.add_argument("--output_key", type=str, default="output",
                        help="Key for output field in input data")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = []
    instructions = []
    
    with open(args.input_file, "r", encoding="utf-8") as f:
        if args.input_file.endswith(".jsonl"):
            for line in f:
                item = json.loads(line)
                data.append(item)
                instructions.append(item.get(args.instruction_key, ""))
        else:  # JSON
            data = json.load(f)
            instructions = [item.get(args.instruction_key, "") for item in data]
    
    print(f"Loaded {len(data)} samples")
    
    # Initialize complexity scorer
    print("\n=== Initializing Complexity Scorer ===")
    print(f"Model: {args.scoring_model}")
    print(f"Using vLLM: {args.use_vllm}")
    scorer = Scorer(args.scoring_model, is_vllm=args.use_vllm)
    
    # Compute semantic difficulty
    print("\n=== Computing Semantic Difficulty ===")
    semantic_scores = predict_semantic_difficulty(scorer, instructions)
    
    # Compute linguistic difficulty
    print("\n=== Computing Linguistic Difficulty (Perplexity) ===")
    linguistic_scores = compute_linguistic_difficulty(
        instructions,
        model_name=args.perplexity_model,
        device=args.device
    )
    
    # Normalize and combine scores
    print("\n=== Computing Final Complexity Scores ===")
    norm_semantic = normalize(semantic_scores)
    norm_linguistic = normalize(linguistic_scores)
    final_scores = (norm_semantic + norm_linguistic) / 2
    
    print(f"Semantic difficulty range: [{min(semantic_scores):.3f}, {max(semantic_scores):.3f}]")
    print(f"Linguistic difficulty range: [{min(linguistic_scores):.3f}, {max(linguistic_scores):.3f}]")
    print(f"Final complexity score range: [{min(final_scores):.3f}, {max(final_scores):.3f}]")
    
    # Partition data by complexity
    print("\n=== Partitioning Data by Complexity ===")
    
    # Convert data to standard format
    formatted_data = []
    for item in data:
        formatted_data.append({
            "instruction": item.get(args.instruction_key, ""),
            "input": item.get("input", ""),
            "output": item.get(args.output_key, "")
        })
    
    partition_by_complexity(formatted_data, final_scores, args.output_dir)
    
    print("\n=== RQ1 Complexity Analysis Complete ===")


if __name__ == "__main__":
    main()

