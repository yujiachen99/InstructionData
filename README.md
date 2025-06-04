## What makes good instruction data for  LLM Fine-tuning on code intelligence tasks？

This repository contains code and resources for the empirical study on how various instruction data features affect the performance of instruction fine-tuning (IFT) for code large language models (LLMs). The study explores four key research questions (RQs) regarding domain composition, semantic diversity, problem complexity, and task-type coverage.

### 📁 Project Structure

```bash

├── Code/
│   ├── RQ1/                        # Domain Composition
│   │   ├── EVOL/
│   │   ├── OSS/
│   │   └── domain_composition.py
│   ├── RQ2/                        # Semantic Diversity
│   │   ├── EVOL/
│   │   ├── OSS/
│   │   ├── evol_embeddings.pickle
│   │   ├── oss_embeddings.pickle
│   │   └── semantic_diversity.py
│   ├── RQ3/                        # Problem Complexity
│   │   ├── EVOL/
│   │   ├── OSS/
│   │   └── problem_complexity.py
│   ├── RQ4/                        # Task-type Coverage
│   │   ├── EVOL/
│   │   ├── OSS/
│   │   ├── evol_instruct_with_category.jsonl
│   │   ├── oss_instruct_with_category.jsonl
│   │   ├── task_category.py
│   │   └── type_coverage.py
├── Datas/                          # Raw and processed instruction datasets


### Model Training and Inference

We use the [LLaMA-Factory library](https://github.com/hiyouga/LLaMA-Factory) for the training and inference process. 


### Evaluation

We evaluate the fine-tuned models using the [BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness), a standard toolkit for benchmarking code LLMs across multiple programming languages and tasks.

