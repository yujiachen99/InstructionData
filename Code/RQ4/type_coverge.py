import json
import os
import random
from tqdm import tqdm

def main():
    CATEGORY_FILE = "oss_instruct_with_category.jsonl"
    ORIGINAL_FILE = "../../Datas/data-oss_instruct-decontaminated.jsonl"
    OUTPUT_DIR = "OSS"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    GROUP_A = {"Algorithmic & Data Structure", "Mathematical & Computational"}
    SUBSET_SIZE = 20000
    RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]

    instr2output = {}
    with open(ORIGINAL_FILE, "r", encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line)
            instr2output[data["problem"]] = data["solution"]
            
    group_a, group_b = [], []
    with open(CATEGORY_FILE, "r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            instr = item["instruction"]
            cat = item["assigned_category"]

            record = {
                "instruction": instr,
                "input": "",
                "output": instr2output.get(instr, "")  
            }

            if cat in GROUP_A:
                group_a.append(record)
            else:
                group_b.append(record)

    print(f"Total Group A: {len(group_a)}, Group B: {len(group_b)}")

    # 3. 构造五个子集
    for ratio in RATIOS:
        num_a = int(SUBSET_SIZE * ratio)
        num_b = SUBSET_SIZE - num_a

        selected_a = random.sample(group_a, num_a)
        selected_b = random.sample(group_b, num_b)

        final_subset = selected_a + selected_b
        random.shuffle(final_subset)

        out_path = os.path.join(OUTPUT_DIR, f"RQ4_OSS_ALG_{int(ratio * 100)}.json")
        with open(out_path, "w", encoding="utf-8") as fout:
            for sample in final_subset:
                json.dump(sample, fout, ensure_ascii=False,indent=4)
                fout.write("\n")

        print(f"Saved {len(final_subset)} samples to {out_path}")

if __name__ == "__main__":
    main()
