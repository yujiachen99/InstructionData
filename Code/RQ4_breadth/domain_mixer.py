#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQ4: Knowledge Breadth (Cross-Domain Knowledge)
This script creates seed knowledge variants by mixing code-domain tasks with
tasks from other domains (General and Math).

Mixing ratios: 10:0 (pure code), 9.7:0.3, 9:1, 7:3 (code:non-code)
"""

import os
import json
import random
import argparse


def load_code_data(code_file, instruction_key="instruction", output_key="output"):
    """Load code-domain data."""
    code_data = []
    
    with open(code_file, "r", encoding="utf-8") as f:
        if code_file.endswith(".jsonl"):
            for line in f:
                item = json.loads(line)
                code_data.append({
                    "instruction": item.get(instruction_key, ""),
                    "input": item.get("input", ""),
                    "output": item.get(output_key, "")
                })
        else:  # JSON
            data = json.load(f)
            for item in data:
                code_data.append({
                    "instruction": item.get(instruction_key, ""),
                    "input": item.get("input", ""),
                    "output": item.get(output_key, "")
                })
    
    return code_data


def load_non_code_data(non_code_file, instruction_key="instruction", output_key="output"):
    """Load non-code domain data (General or Math)."""
    non_code_data = []
    
    with open(non_code_file, "r", encoding="utf-8") as f:
        if non_code_file.endswith(".jsonl"):
            for line in f:
                item = json.loads(line)
                # Filter out any code-related content
                instruction = item.get(instruction_key, "")
                output = item.get(output_key, "")
                
                # Simple heuristic to exclude code snippets
                if not contains_code(instruction) and not contains_code(output):
                    non_code_data.append({
                        "instruction": instruction,
                        "input": item.get("input", ""),
                        "output": output
                    })
        else:  # JSON
            data = json.load(f)
            for item in data:
                instruction = item.get(instruction_key, "")
                output = item.get(output_key, "")
                
                if not contains_code(instruction) and not contains_code(output):
                    non_code_data.append({
                        "instruction": instruction,
                        "input": item.get("input", ""),
                        "output": output
                    })
    
    return non_code_data


def contains_code(text):
    """Simple heuristic to detect code snippets."""
    code_indicators = [
        "def ", "class ", "import ", "from ", "function ", 
        "const ", "let ", "var ", "public ", "private ",
        "```", "```python", "```java", "```javascript",
        "{", "}", "[", "]", "();", "=>", "->",
    ]
    
    text_lower = text.lower()
    for indicator in code_indicators:
        if indicator in text_lower:
            return True
    return False


def create_mixed_datasets(code_data, non_code_data, domain_name, ratios, output_dir):
    """
    Create mixed datasets with different code:non-code ratios.
    
    Args:
        code_data: List of code-domain tasks
        non_code_data: List of non-code domain tasks
        domain_name: Name of non-code domain (e.g., "GENERAL", "MATH")
        ratios: List of (code_ratio, non_code_ratio) tuples
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine base size from code data
    base_size = len(code_data)
    
    for code_ratio, non_code_ratio in ratios:
        num_code = int(base_size * code_ratio)
        num_non_code = int(base_size * non_code_ratio)
        
        # Sample data
        if num_code > len(code_data):
            print(f"Warning: Requested {num_code} code tasks but only {len(code_data)} available")
            num_code = len(code_data)
        
        if num_non_code > len(non_code_data):
            print(f"Warning: Requested {num_non_code} {domain_name} tasks but only {len(non_code_data)} available")
            num_non_code = len(non_code_data)
        
        selected_code = random.sample(code_data, num_code) if num_code > 0 else []
        selected_non_code = random.sample(non_code_data, num_non_code) if num_non_code > 0 else []
        
        mixed_dataset = selected_code + selected_non_code
        random.shuffle(mixed_dataset)
        
        # Save dataset
        ratio_str = f"{int(code_ratio*10)}_{int(non_code_ratio*10)}"
        output_path = os.path.join(
            output_dir,
            f"RQ4_BREADTH_CODE_{int(code_ratio*100)}_{domain_name}_{int(non_code_ratio*100)}.json"
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mixed_dataset, f, indent=4, ensure_ascii=False)
        
        print(f"Saved {len(mixed_dataset)} samples (Code:{num_code}, {domain_name}:{num_non_code}) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RQ4: Mix code and non-code domain knowledge")
    parser.add_argument("--code_file", type=str, required=True,
                        help="Path to code-domain JSONL file")
    parser.add_argument("--general_file", type=str, default=None,
                        help="Path to general-domain JSONL file (e.g., Alpaca)")
    parser.add_argument("--math_file", type=str, default=None,
                        help="Path to math-domain JSONL file (e.g., MathInstruct)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save mixed datasets")
    parser.add_argument("--code_instruction_key", type=str, default="instruction",
                        help="Instruction key for code data")
    parser.add_argument("--code_output_key", type=str, default="output",
                        help="Output key for code data")
    parser.add_argument("--non_code_instruction_key", type=str, default="instruction",
                        help="Instruction key for non-code data")
    parser.add_argument("--non_code_output_key", type=str, default="output",
                        help="Output key for non-code data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load code data
    print(f"Loading code data from {args.code_file}...")
    code_data = load_code_data(args.code_file, args.code_instruction_key, args.code_output_key)
    print(f"Loaded {len(code_data)} code samples")
    
    # Define mixing ratios (code:non-code)
    ratios = [
        (1.0, 0.0),     # 10:0 - Pure code (baseline)
        (0.97, 0.03),   # 9.7:0.3
        (0.9, 0.1),     # 9:1
        (0.7, 0.3),     # 7:3
    ]
    
    print(f"\nMixing ratios: {ratios}")
    
    # Mix with General domain
    if args.general_file:
        print(f"\n=== Mixing with General Domain ===")
        print(f"Loading general data from {args.general_file}...")
        general_data = load_non_code_data(
            args.general_file,
            args.non_code_instruction_key,
            args.non_code_output_key
        )
        print(f"Loaded {len(general_data)} general samples (after filtering)")
        
        create_mixed_datasets(
            code_data,
            general_data,
            "GENERAL",
            ratios,
            args.output_dir
        )
    
    # Mix with Math domain
    if args.math_file:
        print(f"\n=== Mixing with Math Domain ===")
        print(f"Loading math data from {args.math_file}...")
        math_data = load_non_code_data(
            args.math_file,
            args.non_code_instruction_key,
            args.non_code_output_key
        )
        print(f"Loaded {len(math_data)} math samples (after filtering)")
        
        create_mixed_datasets(
            code_data,
            math_data,
            "MATH",
            ratios,
            args.output_dir
        )
    
    print("\n=== RQ4 Domain Mixing Complete ===")


if __name__ == "__main__":
    main()

