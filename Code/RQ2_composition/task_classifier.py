#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQ2: Knowledge Composition
This script classifies seed knowledge into task categories and analyzes composition.
It uses INSTRUCTOR embeddings to assign tasks to predefined categories.

Categories are grouped into:
- Algorithmic Tasks: Algorithmic & Data Structure, Mathematical & Computational
- Practical Tasks: All other categories (Domain-Specific, Web, Database, etc.)
"""

import os
import json
import argparse
import numpy as np
from collections import Counter

import torch
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity


# Task categories from the paper
CATEGORY_LIST = [
    "Algorithmic & Data Structure",
    "Mathematical & Computational",
    "Database & SQL",
    "System Design & Architecture",
    "Web Development",
    "Data Science & Machine Learning",
    "Performance Optimization",
    "User Interface & Application Design",
    "Domain-Specific",
    "Security & Cryptography"
]

# Group categories
ALGORITHMIC_CATEGORIES = {
    "Algorithmic & Data Structure",
    "Mathematical & Computational"
}


def classify_tasks(instructions, model_name="hkunlp/instructor-base", batch_size=32):
    """
    Classify tasks into predefined categories using INSTRUCTOR embeddings.
    
    Args:
        instructions: List of task descriptions
        model_name: INSTRUCTOR model name
        batch_size: Batch size for encoding
    
    Returns:
        assigned_categories: List of assigned category names
        similarities: Similarity scores for each assignment
    """
    print(f"Loading INSTRUCTOR model: {model_name}")
    model = INSTRUCTOR(model_name)
    
    # Encode category names
    print("Encoding categories...")
    category_prompts = [
        ["Represent the topic for coding category:", cat] 
        for cat in CATEGORY_LIST
    ]
    category_embeddings = model.encode(
        category_prompts,
        batch_size=16,
        show_progress_bar=True
    )
    
    # Encode instructions
    print(f"Encoding {len(instructions)} instructions...")
    sample_prompts = [
        ["Represent coding category of the programming task:", instr] 
        for instr in instructions
    ]
    
    all_sample_embs = []
    for i in range(0, len(instructions), batch_size):
        batch_prompts = sample_prompts[i:i + batch_size]
        batch_embs = model.encode(
            batch_prompts,
            batch_size=len(batch_prompts),
            show_progress_bar=True
        )
        all_sample_embs.append(batch_embs)
    
    sample_embeddings = np.vstack(all_sample_embs)
    
    # Compute similarities and assign categories
    print("Computing similarities and assigning categories...")
    sims = cosine_similarity(sample_embeddings, category_embeddings)
    best_category_idxs = np.argmax(sims, axis=1)
    
    assigned_categories = [CATEGORY_LIST[idx] for idx in best_category_idxs]
    max_similarities = [sims[i, idx] for i, idx in enumerate(best_category_idxs)]
    
    return assigned_categories, max_similarities


def analyze_composition(assigned_categories):
    """Analyze the composition of task types."""
    counter = Counter(assigned_categories)
    total = len(assigned_categories)
    
    print("\n=== Task Category Distribution ===")
    for cat, count in counter.most_common():
        percentage = (count / total) * 100
        print(f"{cat}: {count} ({percentage:.1f}%)")
    
    # Group into Algorithmic vs Practical
    algorithmic_count = sum(counter[cat] for cat in ALGORITHMIC_CATEGORIES if cat in counter)
    practical_count = total - algorithmic_count
    
    print(f"\n=== High-Level Grouping ===")
    print(f"Algorithmic Tasks: {algorithmic_count} ({algorithmic_count/total*100:.1f}%)")
    print(f"Practical Tasks: {practical_count} ({practical_count/total*100:.1f}%)")
    
    return counter


def save_classified_data(data, assigned_categories, similarities, output_path):
    """Save data with assigned categories."""
    output_data = []
    for item, category, sim in zip(data, assigned_categories, similarities):
        item_copy = item.copy()
        item_copy["assigned_category"] = category
        item_copy["category_similarity"] = float(sim)
        item_copy["task_type"] = "Algorithmic" if category in ALGORITHMIC_CATEGORIES else "Practical"
        output_data.append(item_copy)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\nSaved classified data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RQ2: Classify tasks into categories")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output JSONL file with categories")
    parser.add_argument("--instructor_model", type=str, default="hkunlp/instructor-base",
                        help="INSTRUCTOR model name")
    parser.add_argument("--instruction_key", type=str, default="instruction",
                        help="Key for instruction field in input data")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding")
    
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
    
    # Classify tasks
    assigned_categories, similarities = classify_tasks(
        instructions,
        model_name=args.instructor_model,
        batch_size=args.batch_size
    )
    
    # Analyze composition
    analyze_composition(assigned_categories)
    
    # Save results
    save_classified_data(data, assigned_categories, similarities, args.output_file)
    
    print("\n=== RQ2 Task Classification Complete ===")


if __name__ == "__main__":
    main()

