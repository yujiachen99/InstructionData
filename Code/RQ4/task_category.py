#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
import json

def main():

    INSTRUCTOR_MODEL_NAME = "/data/cyjia/instructor-base"
    CATEGORY_LIST = [
        "Database & SOL", "System Design & Architecture", "Web", "Algorithmic & Data Structure",
        "Mathematical & Computational", "Data Science & Machine Learning", "Performance Optimization", "User Interface & Application Design",
        "Domain Specific", "Security & Cryptography"
    ]
    
    OSS_INSTRUCT_PATH = "../../Datas/data-oss_instruct-decontaminated.jsonl"
    
    model = INSTRUCTOR(INSTRUCTOR_MODEL_NAME)
    
    category_prompts = [
        ["Represent the topic for coding category:", cat] 
        for cat in CATEGORY_LIST
    ]
    
    category_embeddings = model.encode(
        category_prompts,
        batch_size=16,
        show_progress_bar=True
    )  
    
    instructions = []
    with open(OSS_INSTRUCT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            instructions.append(data["problem"])

    num_samples = len(instructions)
    
    sample_prompts = [
        ["Represent coding category of the programming task:", instr_text] 
        for instr_text in instructions
    ]
    
    BATCH_SIZE = 32
    all_sample_embs = [] 
    for i in range(0, num_samples, BATCH_SIZE):
        batch_prompts = sample_prompts[i : i + BATCH_SIZE]
        batch_embs = model.encode(
            batch_prompts,
            batch_size=len(batch_prompts),
            show_progress_bar=True
        )
        all_sample_embs.append(batch_embs)

    sample_embeddings = np.vstack(all_sample_embs)
    
    sims = cosine_similarity(sample_embeddings, category_embeddings)
    best_category_idxs = np.argmax(sims, axis=1) 
    
    
    OUTPUT_PATH = "oss_instruct_with_category.jsonl"
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for i, example in enumerate(instructions):
            assigned_idx = int(best_category_idxs[i])
            assigned_cat = CATEGORY_LIST[assigned_idx]
            out_record = {
                "instruction": example,
                "assigned_category": assigned_cat,
                "category_similarity": float(sims[i, assigned_idx])
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()


