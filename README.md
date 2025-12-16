### What Makes Effective Knowledge Distillation for Large Language Models in Code Generation?

This repository contains code and resources for the empirical study on how various characteristics of seed knowledge affect the effectiveness of knowledge distillation (KD) for code generation. The study systematically investigates four key research questions (RQs) regarding knowledge complexity, task composition, semantic diversity, and cross-domain knowledge.


### 🔍 Research Questions

#### RQ1: Knowledge Complexity
- How does the difficulty distribution of seed knowledge affect distillation effectiveness? 

#### RQ2: Knowledge Composition
-  How do different proportions of task types in seed knowledge influence distillation outcomes?

#### RQ3: Knowledge Density
-  How does the semantic diversity of seed knowledge impact the student model's code generation ability?

#### RQ4: Knowledge Breadth
-  How does incorporating cross-domain knowledge enhance code generation capabilities during knowledge distillation?

### 📁 Project Structure

```
InstructionData/
├── Code/                                    # Scripts for each RQ
│   ├── RQ1_complexity/                      # Knowledge Complexity
│   │   ├── complexity.py                    # Scorer class for XCoder-Complexity-Scorer
│   │   └── complexity_scorer.py             # Compute complexity scores and partition data
│   ├── RQ2_composition/                     # Knowledge Composition
│   │   ├── task_classifier.py               # Classify tasks into categories
│   │   └── composition_mixer.py             # Mix algorithmic and practical tasks
│   ├── RQ3_density/                         # Knowledge Density
│   │   └── diversity_scorer.py              # Compute semantic diversity and partition
│   └── RQ4_breadth/                         # Knowledge Breadth
│       └── domain_mixer.py                  # Mix code and non-code domains
├── Datas/                                   # Raw and processed datasets
│   ├── RQ1/                                 # Complexity-partitioned datasets
│   │   ├── EVOL/
│   │   └── OSS/
│   ├── RQ2/                                 # Composition-mixed datasets
│   │   ├── EVOL/
│   │   └── OSS/
│   ├── RQ3/                                 # Diversity-partitioned datasets
│   │   ├── EVOL/
│   │   └── OSS/
│   └── RQ4/                                 # Cross-domain mixed datasets
│       ├── EVOL/
│       └── OSS/
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

### 🚀 Quick Start

1. Clone the repository:
```bash
git clone <this-repo>
cd InstructionData
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install vLLM for faster inference:
```bash
pip install vllm>=0.2.0
```

4. (Optional) Install Git LFS for large files:
```bash
git lfs install
git lfs pull
```

### 📊 Datasets

#### Seed Knowledge Sources

We use two widely-adopted seed knowledge sources:

1. **Evol-Instruct**: Evolution-based seed knowledge with progressive task enhancement
2. **OSS-Instruct**: Code-based seed knowledge derived from real-world open-source snippets

#### Cross-Domain Sources

For RQ4, we incorporate:

1. **Alpaca**: General-domain natural language tasks
2. **MathInstruct**: Math-domain reasoning tasks


### Model Training

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for fine-tuning student models


### Benchmarks

We evaluate on multiple benchmarks:

- **HumanEval**: 164 hand-written programming problems
- **MBPP**:  427 crowd-sourced Python programming problems
- **LiveCodeBench**: 880 Competitive programming tasks
- **MultiPL-E**: Multi-language evaluation (Python, Java, JavaScript, C++, etc.)


### 📦 Large File Handling via Git LFS

Some files in this repository (e.g., JSONL datasets and embedding files) are large and managed with Git LFS (Large File Storage).

When you clone this repository, Git LFS will automatically download the necessary large files if you have Git LFS installed:

```bash
git lfs install          # Run once per machine
git clone <this-repo>    # LFS files will be pulled automatically
```

If you already cloned the repo but didn't get the LFS files:

```bash
git lfs pull
```

### 🙏 Acknowledgments

We thank the developers of:
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for the training framework
- [Complexity-Scorer](https://huggingface.co/banksy235/XCoder-Complexity-Scorer) for scoring complexity 
- [INSTRUCTOR](https://github.com/xlang-ai/instructor-embedding) for semantic embeddings
