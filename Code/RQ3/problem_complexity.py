import os
import math
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.special import softmax


def load_scoring_model(model_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=False, device_map={"": device})
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_score_for_instructions(model, tokenizer, instructions, device: str = "cuda"):
    scores = []
    for instruction in tqdm(instructions, desc="Predicting score"):
        prompt = (
            "You are a helpful assistant. Please identify the complexity score of the following programming problem.\n"
            f"##Problem: {instruction}\n##Complexity: "
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs, max_length=512, num_return_sequences=1, return_dict_in_generate=True, output_scores=True
        )
        logits = outputs.scores[0][0]  # shape: [vocab_size]

        id2score = {
            29896: 1,
            29906: 2,
            29941: 3,
            29946: 4,
            29945: 5,
            29953: 6
        }

        logit_scores = np.array([logits[k].item() for k in id2score])
        prob_scores = softmax(logit_scores)
        expected_score = np.dot(prob_scores, list(id2score.values()))
        scores.append(expected_score)
    return scores


@torch.no_grad()
def compute_perplexity_for_instructions(instructions, perp_model_name="llama2-7b", device="cuda", max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(perp_model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(perp_model_name).to(device)
    model.eval()

    perps = []
    for instr in tqdm(instructions, desc="Computing perplexity"):
        enc = tokenizer(instr, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        outputs = model(**enc, labels=enc.input_ids)
        loss = outputs.loss.item()
        perps.append(math.exp(loss))
    return perps


def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def main():
    INSTRUCT_PATH = "../../Datas/data-evol_instruct-decontaminated.jsonl"
    SCORING_MODEL_PATH = "scoring_model"
    PERP_MODEL_NAME = "llama2-7B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    instructions = []
    with open(INSTRUCT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            instructions.append(data["instruction"])

    scoring_model, scoring_tokenizer = load_scoring_model(SCORING_MODEL_PATH, device=DEVICE)
    
    model_scores = predict_score_for_instructions(scoring_model, scoring_tokenizer, instructions, device=DEVICE)
    perp_scores = compute_perplexity_for_instructions(instructions, perp_model_name=PERP_MODEL_NAME, device=DEVICE, max_length=256)

    norm_model_scores = normalize(model_scores)
    norm_perp_scores = normalize(perp_scores)
    final_scores = norm_model_scores + norm_perp_scores
    
    data = []
    with open(INSTRUCT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line, score in zip(lines, final_scores):
        obj = json.loads(line)
        obj["score"] = float(score)
        data.append(obj)

    data.sort(key=lambda x: x["score"])

    total = len(data)
    parts = [data[:total // 3], data[total // 3:2 * total // 3], data[2 * total // 3:]]

    for idx, part in enumerate(parts):
        with open(f"EVOL/RQ3_EVOL_{['SIMPLE', 'MODERATE', 'COMPLEX'][idx]}.json", "w", encoding="utf-8") as f:
            for item in part:
                json.dump({
                    "instruction": item["instruction"],
                    "input": "",
                    "output": item["response"]
                }, f, ensure_ascii=False,indent=4)
                f.write("\n")


if __name__ == "__main__":
    main()
