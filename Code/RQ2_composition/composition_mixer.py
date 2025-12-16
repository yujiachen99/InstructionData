#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQ2: Knowledge Composition - Mix different task types
This script creates seed knowledge variants with different mixing ratios of
Algorithmic and Practical tasks.

Mixing ratios: 0:10, 2:8, 5:5, 8:2, 10:0 (Algorithmic:Practical)
"""

import os
import json
import random
import argparse
from collections import defaultdict


ALGORITHMIC_CATEGORIES = {
    "Algorithmic & Data Structure",
    "Mathematical & Computational"
}


def load_classified_data(input_file):
    """Load data with task type classifications."""
    algorithmic_tasks = []
    practical_tasks = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            category = item.get("assigned_category", "")
            
            # Convert to standard format
            formatted_item = {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", "")
            }
            
            if category in ALGORITHMIC_CATEGORIES:
                algorithmic_tasks.append(formatted_item)
            else:
                practical_tasks.append(formatted_item)
    
    return algorithmic_tasks, practical_tasks


def create_mixed_datasets(algorithmic_tasks, practical_tasks, ratios, subset_size, num_repeats, output_dir):
    """
    Create mixed datasets with different ratios of algorithmic and practical tasks.
    
    Args:
        algorithmic_tasks: List of algorithmic tasks
        practical_tasks: List of practical tasks
        ratios: List of (algorithmic_ratio, practical_ratio) tuples
        subset_size: Total size of each subset
        num_repeats: Number of random samples per ratio
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for alg_ratio, prac_ratio in ratios:
        num_alg = int(subset_size * alg_ratio)
        num_prac = subset_size - num_alg
        
        # Check if we have enough samples
        if num_alg > len(algorithmic_tasks):
            print(f"Warning: Requested {num_alg} algorithmic tasks but only {len(algorithmic_tasks)} available")
            num_alg = len(algorithmic_tasks)
            num_prac = subset_size - num_alg
        
        if num_prac > len(practical_tasks):
            print(f"Warning: Requested {num_prac} practical tasks but only {len(practical_tasks)} available")
            num_prac = len(practical_tasks)
            num_alg = subset_size - num_prac
        
        # Create multiple random samples for robustness
        for repeat_idx in range(num_repeats):
            selected_alg = random.sample(algorithmic_tasks, num_alg) if num_alg > 0 else []
            selected_prac = random.sample(practical_tasks, num_prac) if num_prac > 0 else []
            
            mixed_dataset = selected_alg + selected_prac
            random.shuffle(mixed_dataset)
            
            # Save dataset
            ratio_str = f"{int(alg_ratio*10)}_{int(prac_ratio*10)}"
            output_path = os.path.join(
                output_dir, 
                f"RQ2_COMPOSITION_ALG_{int(alg_ratio*10)}_PRAC_{int(prac_ratio*10)}_repeat{repeat_idx+1}.json"
            )
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(mixed_dataset, f, indent=4, ensure_ascii=False)
            
            print(f"Saved {len(mixed_dataset)} samples (Alg:{num_alg}, Prac:{num_prac}) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RQ2: Create mixed datasets with different task type ratios")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to classified JSONL file (output from task_classifier.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save mixed datasets")
    parser.add_argument("--subset_size", type=int, default=20000,
                        help="Size of each mixed dataset")
    parser.add_argument("--num_repeats", type=int, default=3,
                        help="Number of random samples per ratio for robustness")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load classified data
    print(f"Loading classified data from {args.input_file}...")
    algorithmic_tasks, practical_tasks = load_classified_data(args.input_file)
    
    print(f"Loaded {len(algorithmic_tasks)} algorithmic tasks")
    print(f"Loaded {len(practical_tasks)} practical tasks")
    
    # Define mixing ratios (Algorithmic:Practical)
    ratios = [
        (0.0, 1.0),   # 0:10 - Pure practical
        (0.2, 0.8),   # 2:8
        (0.5, 0.5),   # 5:5 - Balanced
        (0.8, 0.2),   # 8:2
        (1.0, 0.0),   # 10:0 - Pure algorithmic
    ]
    
    print(f"\n=== Creating Mixed Datasets ===")
    print(f"Subset size: {args.subset_size}")
    print(f"Number of repeats per ratio: {args.num_repeats}")
    print(f"Mixing ratios: {ratios}")
    
    create_mixed_datasets(
        algorithmic_tasks,
        practical_tasks,
        ratios,
        args.subset_size,
        args.num_repeats,
        args.output_dir
    )
    
    print("\n=== RQ2 Composition Mixing Complete ===")


if __name__ == "__main__":
    main()

