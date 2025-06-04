## What makes good instruction data for  LLM Fine-tuning on code intelligence tasks？

This repository contains code and resources for the empirical study on how various instruction data features affect the performance of instruction fine-tuning (IFT) for code large language models (LLMs). The study explores four key research questions (RQs) regarding domain composition, semantic diversity, problem complexity, and task-type coverage.

### 📁 Project Structure

```bash

INSTRUCTIONDATA/
├── Code/                         # Scripts for analyzing the impact of different instruction properties
│   ├── RQ1/
│   │   ├── domain_composition.py
│   ├── RQ2/
│   │   ├── semantic_diversity.py
│   ├── RQ3/
│   │   └── problem_complexity.py
│   ├── RQ4/
│   │   ├── evol_instruct_with_category.jsonl
│   │   ├── oss_instruct_with_category.jsonl
│   │   ├── task_category.py
│   │   └── type_coverage.py
├── Datas/                        # Raw and split instruction datasets for each RQ
│   ├── RQ1/
│   ├── RQ2/
│   ├── RQ3/
│   └── RQ4/
├── .gitattributes                # Git LFS tracking configuration
├── README.md
└── requirements.txt              # Python dependencies
```

## 📦 Large File Handling via Git LFS

Some files in this repository (e.g., JSONL datasets and embedding files) are large and thus managed with Git LFS (Large File Storage). This ensures efficient storage and versioning without bloating the Git history.

> 💡 When you clone this repository, Git LFS will automatically download the necessary large files if you have Git LFS installed and initialized:

```bash
git lfs install          # Run once per machine
git clone <this-repo>    # LFS files will be pulled automatically
```

If you already cloned the repo but didn't get the LFS files:

```bash
git lfs pull
```

### Model Training and Inference

We use the [LLaMA-Factory library](https://github.com/hiyouga/LLaMA-Factory) for the training and inference process. 


### Evaluation

We evaluate the fine-tuned models using the [BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness), a standard toolkit for benchmarking code LLMs across multiple programming languages and tasks.

