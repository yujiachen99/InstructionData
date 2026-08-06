# Supplementary Learning-Curve Analysis for RQ5

## Overview

This directory contains a supplementary analysis for **RQ5 (Practical Competitiveness)** in *What Makes Effective Knowledge Distillation for Large Language Models in Code Generation?* The analysis examines whether the advantage of the curated seed-knowledge subset over the full dataset can be explained by differences in the training budget or learning-rate schedule.

The main RQ5 experiment trains each configuration for four epochs. Because the full Evol-Instruct dataset contains approximately 75k samples while the curated subset contains approximately one-third as many samples (about 25k), four epochs correspond to different numbers of optimizer updates. This supplementary experiment therefore compares the two datasets under the same optimizer-update budget and the same learning-rate schedule.

## Context: Main RQ5 Results

Under the four-epoch setting used in the main study, the curated data outperforms all data on every reported Python benchmark:

| Training data | HumanEval | HumanEval+ | MBPP | MBPP+ | LiveCodeBench Full | LiveCodeBench Easy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All data | 70.7 | 64.0 | 75.7 | 64.4 | 13.2 | 34.5 |
| Curated data | **75.0** | **68.9** | **79.2** | **67.7** | **14.5** | **40.5** |
| Difference | +4.3 | +4.9 | +3.5 | +3.3 | +1.3 | +6.0 |

The controlled learning-curve experiment below tests whether this pattern is merely an artifact of the unequal numbers of optimizer updates produced by fixed-epoch training.

## Experimental Setup

- **Seed-knowledge source:** Evol-Instruct
- **Student model:** DeepSeekCoder-6.7B
- **Training-data variants:** the full dataset and the subset produced by the final RQ5 curation pipeline
- **Training budget:** 600 optimizer update steps for each variant
- **Learning-rate schedule:** the same 600-step cosine schedule for both variants
- **Other hyperparameters:** identical across the two runs
- **Checkpoint interval:** every 50 optimizer update steps
- **Recorded quantities:** training loss and Pass@1 on HumanEval, HumanEval+, MBPP, MBPP+, LiveCodeBench Full, and LiveCodeBench Easy

All checkpoints were evaluated only after training had finished. The benchmark results are therefore a **post hoc analysis** and were not used for early stopping or checkpoint selection.

For reference, four epochs correspond to approximately step 196 for the curated subset and step 588 for the full dataset. The vertical lines in the figure mark these two points.

## Learning Curves

The complete figure is available in [`loss.pdf`](./loss.pdf). Panel (a) reports training loss, and panels (b)-(g) report checkpoint-level Pass@1 on the six benchmarks.

## Results

1. **The curated data achieves a higher peak Pass@1 on all six benchmarks under the controlled 600-step budget.** Both variants use the same number of optimizer updates and the same learning-rate schedule, so the curated subset's advantage cannot be attributed solely to the unequal update counts or schedule lengths in the main four-epoch experiment.

2. **The full-data model stabilizes before its four-epoch point at step 588.** Its lower Pass@1 in the main RQ5 comparison is therefore unlikely to be caused by stopping training before the model has reached a stable performance region.

3. **The curated-data model peaks earlier and then declines with continued training.** After the benchmark scores reach their maxima, the curated model's training loss continues to decrease while its Pass@1 decreases. At step 600, which is approximately 12 epochs over the curated subset, it performs worse than the full-data model on all six benchmarks. This pattern indicates that additional optimization on the smaller dataset improves training fit but does not produce sustained benchmark gains.

## Interpretation

Fixed-epoch and fixed-step comparisons favor different aspects of the training process: fixed epochs give the larger dataset more optimizer updates, whereas fixed steps expose the smaller dataset to more repeated passes. The controlled experiment shows that the curated subset can still reach better benchmark performance when optimizer updates and the learning-rate schedule are held constant. Its advantage is therefore consistent with the value of the RQ5 curation pipeline rather than being solely a consequence of the original training-budget difference.

At the same time, the late-stage decline of the curated model shows that training the smaller subset for many additional epochs is not beneficial. Because the checkpoint evaluations are post hoc and cannot be used to select a stopping point, the main study retains the uniform four-epoch protocol rather than choosing the best checkpoint separately for each benchmark.
