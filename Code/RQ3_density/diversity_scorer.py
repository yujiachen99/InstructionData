#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQ3: Knowledge Density (Semantic Diversity)
This script computes semantic diversity scores for each task in the seed knowledge.

Diversity score for task i = 1 - average_similarity(i, all_other_tasks)
Higher diversity score = more semantically distinct from other tasks
Lower diversity score = more semantically similar to other tasks

Tasks are partitioned into three tertiles based on diversity scores.
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from InstructorEmbedding import INSTRUCTOR
import faiss


def compute_embeddings_instructor(instructions, model_name="hkunlp/instructor-base", batch_size=32):
    """Compute embeddings using INSTRUCTOR model."""
    print(f"Loading INSTRUCTOR model: {model_name}")
    model = INSTRUCTOR(model_name)
    
    # Prepare prompts
    sample_prompts = [
        ["Represent the programming task:", instr] 
        for instr in instructions
    ]
    
    print(f"Encoding {len(instructions)} instructions...")
    all_embeddings = []
    for i in tqdm(range(0, len(instructions), batch_size), desc="Encoding batches"):
        batch_prompts = sample_prompts[i:i + batch_size]
        batch_embs = model.encode(
            batch_prompts,
            batch_size=len(batch_prompts),
            show_progress_bar=False
        )
        all_embeddings.append(batch_embs)
    
    embeddings = np.vstack(all_embeddings)
    return embeddings


def compute_embeddings_transformer(instructions, model_name="microsoft/codebert-base", 
                                   device="cuda", batch_size=32, max_length=512):
    """Compute embeddings using a transformer model (e.g., CodeBERT)."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(instructions), batch_size), desc="Encoding batches"):
            batch_texts = instructions[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(device)
            
            outputs = model(**inputs)
            # Mean pooling
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs.attention_mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
            all_embeddings.append(pooled.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    return embeddings


def compute_diversity_scores(embeddings):
    """
    Compute diversity scores using FAISS for efficient similarity search.
    
    For each task i:
        similarity(i) = average cosine similarity to all other tasks
        diversity(i) = 1 - similarity(i)
    """
    print("Computing diversity scores...")
    
    # Normalize embeddings for cosine similarity
    norm_X = np.sqrt((embeddings ** 2).sum(1))
    embeddings_normalized = embeddings / norm_X[:, np.newaxis]
    
    # Build FAISS index for efficient similarity search
    d = embeddings_normalized.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner product = cosine similarity for normalized vectors
    index.add(embeddings_normalized)
    
    # Search for all neighbors
    k = embeddings_normalized.shape[0]
    similarities, indices = index.search(embeddings_normalized, k)
    
    # Compute average similarity for each task (excluding self-similarity)
    # similarities[:, 0] is self-similarity (always 1.0)
    average_similarities = np.mean(similarities[:, 1:], axis=1)  # Exclude first column (self)
    
    # Diversity score = 1 - average_similarity
    diversity_scores = 1 - average_similarities
    
    return diversity_scores, average_similarities


def partition_by_diversity(data, diversity_scores, output_dir):
    """
    Partition data into three tertiles based on diversity scores.
    T1: Lower diversity (high redundancy, similar to others)
    T2: Middle diversity
    T3: Higher diversity (high distinctiveness, unique tasks)
    """
    # Sort data by diversity scores
    sorted_indices = np.argsort(diversity_scores)
    sorted_data = [data[i] for i in sorted_indices]
    sorted_scores = [diversity_scores[i] for i in sorted_indices]
    
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
        output_path = os.path.join(output_dir, f"RQ3_DIVERSITY_{name.upper()}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, indent=4, ensure_ascii=False)
        
        avg_score = np.mean(tertile_scores[name])
        min_score = np.min(tertile_scores[name])
        max_score = np.max(tertile_scores[name])
        print(f"Saved {len(subset)} samples to {output_path}")
        print(f"  Diversity range: [{min_score:.3f}, {max_score:.3f}], avg: {avg_score:.3f}")
    
    # Save all data with scores
    all_data_with_scores = []
    for i, (item, score) in enumerate(zip(sorted_data, sorted_scores)):
        item_copy = item.copy()
        item_copy["diversity_score"] = float(score)
        all_data_with_scores.append(item_copy)
    
    scores_path = os.path.join(output_dir, "all_data_with_diversity_scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(all_data_with_scores, f, indent=4, ensure_ascii=False)
    print(f"\nSaved all data with diversity scores to {scores_path}")


def main():
    parser = argparse.ArgumentParser(description="RQ3: Compute semantic diversity scores and partition data")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save partitioned datasets")
    parser.add_argument("--embedding_model", type=str, default="instructor",
                        choices=["instructor", "codebert", "transformer"],
                        help="Type of embedding model to use")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Specific model name (default: hkunlp/instructor-base for instructor, microsoft/codebert-base for codebert)")
    parser.add_argument("--instruction_key", type=str, default="instruction",
                        help="Key for instruction field in input data")
    parser.add_argument("--output_key", type=str, default="output",
                        help="Key for output field in input data")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Set default model names
    if args.model_name is None:
        if args.embedding_model == "instructor":
            args.model_name = "hkunlp/instructor-base"
        elif args.embedding_model == "codebert":
            args.model_name = "microsoft/codebert-base"
        else:
            args.model_name = "microsoft/codebert-base"
    
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
    
    # Compute embeddings
    print(f"\n=== Computing Embeddings using {args.embedding_model} ===")
    if args.embedding_model == "instructor":
        embeddings = compute_embeddings_instructor(
            instructions,
            model_name=args.model_name,
            batch_size=args.batch_size
        )
    else:
        embeddings = compute_embeddings_transformer(
            instructions,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size
        )
    
    print(f"Embedding shape: {embeddings.shape}")
    
    # Compute diversity scores
    print("\n=== Computing Diversity Scores ===")
    diversity_scores, avg_similarities = compute_diversity_scores(embeddings)
    
    print(f"Diversity score range: [{diversity_scores.min():.3f}, {diversity_scores.max():.3f}]")
    print(f"Average similarity range: [{avg_similarities.min():.3f}, {avg_similarities.max():.3f}]")
    print(f"Mean diversity: {diversity_scores.mean():.3f}")
    print(f"Std diversity: {diversity_scores.std():.3f}")
    
    # Convert data to standard format
    formatted_data = []
    for item in data:
        formatted_data.append({
            "instruction": item.get(args.instruction_key, ""),
            "input": item.get("input", ""),
            "output": item.get(args.output_key, "")
        })
    
    # Partition data by diversity
    print("\n=== Partitioning Data by Diversity ===")
    partition_by_diversity(formatted_data, diversity_scores, args.output_dir)
    
    print("\n=== RQ3 Diversity Analysis Complete ===")


if __name__ == "__main__":
    main()

