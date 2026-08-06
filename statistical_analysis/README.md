# Problem-Level Statistical Analysis

This directory contains the supplementary statistical data. The analysis replaces comparisons based on aggregate benchmark scores with paired, problem-level analyses conducted separately within each benchmark.

## Statistical method

- Each `evaluation_results.csv` records whether a model solved each problem (`1` = pass, `0` = fail) under the compared training settings.
- McNemar tests are applied to paired per-problem correctness outcomes within each benchmark. A result is counted as statistically significant when `p < 0.05`.
- Pass@1 differences are reported in percentage points. Their 95% confidence intervals are calculated from 500 paired bootstrap samples within each benchmark.
- The tables below report the average Pass@1 difference and its 95% confidence interval, together with the number of significant McNemar tests among all evaluated configurations.

The evaluated benchmarks are HumanEval, HumanEval+, MBPP, MBPP+, LiveCodeBench Full, LiveCodeBench Easy, and MultiPL-E. MultiPL-E covers C++, Java, JavaScript, PHP, Rust, and Swift.

## Data organization

```text
RQ1/  Knowledge complexity
RQ2/  Knowledge composition
RQ3/  Knowledge density
RQ4/  Knowledge breadth
```

Within each RQ, results are organized by benchmark, seed knowledge source (`Evol-Instruct` or `OSS-Instruct`), and student model (`CodeQwen-7B` or `DeepSeekCoder-6.7B`). The `bootstrap_problem_id_samples/` directory contains the 500 paired bootstrap samples used for each benchmark.

## RQ1: Knowledge Complexity

`T_medium` is compared with `T_low` and `T_high` using four combinations of seed knowledge source and student model.

| Benchmark | `T_medium` vs. `T_low`: Pass@1 difference [95% CI] | Tests with `p < 0.05` | `T_medium` vs. `T_high`: Pass@1 difference [95% CI] | Tests with `p < 0.05` |
|---|---:|---:|---:|---:|
| HumanEval | +5.18 [2.74, 7.62] | 3/4 | +3.51 [0.76, 6.40] | 1/4 |
| HumanEval+ | +5.03 [2.74, 7.16] | 4/4 | +3.66 [2.13, 5.34] | 4/4 |
| MBPP | +3.57 [1.73, 5.50] | 4/4 | +3.45 [1.87, 5.18] | 4/4 |
| MBPP+ | +3.75 [1.84, 5.74] | 4/4 | +3.45 [2.05, 5.04] | 4/4 |
| LCB Full | +1.85 [0.97, 2.60] | 4/4 | +1.25 [0.67, 1.96] | 4/4 |
| LCB Easy | +4.93 [2.96, 6.90] | 4/4 | +3.32 [1.79, 4.93] | 4/4 |
| MultiPL-E | +2.34 [1.80, 2.90] | 4/4 | +1.50 [1.10, 1.88] | 4/4 |
| **Total** |  | **27/28** |  | **25/28** |

The results consistently support the advantage of medium-complexity seed knowledge: all benchmark-level confidence intervals exclude zero in both comparisons.

## RQ2: Knowledge Composition

The mixed task compositions (`5:5` and `8:2`) are compared with the two pure compositions (`0:10` and `10:0`). Each comparison covers 12 combinations of two seed knowledge sources, two student models, and three independently sampled task sets.

| Comparison | Range of average Pass@1 differences across benchmarks | Significant McNemar tests | Bootstrap result |
|---|---:|---:|---|
| `5:5` vs. `0:10` | +2.63 to +7.93 | 84/84 | All seven 95% CIs exclude zero |
| `5:5` vs. `10:0` | +1.34 to +5.78 | 77/84 | All seven 95% CIs exclude zero |
| `8:2` vs. `0:10` | +2.70 to +7.62 | 84/84 | All seven 95% CIs exclude zero |
| `8:2` vs. `10:0` | +1.40 to +3.57 | 77/84 | All seven 95% CIs exclude zero |

The paired analyses support higher Pass@1 for the mixed compositions than for either pure composition.

## RQ3: Knowledge Density

For Evol-Instruct, `T_high` is compared with `T_low` using CodeQwen-7B and DeepSeekCoder-6.7B.

| Benchmark | `T_high` vs. `T_low`: Pass@1 difference [95% CI] | Tests with `p < 0.05` |
|---|---:|---:|
| HumanEval | +6.40 [2.44, 10.53] | 2/2 |
| HumanEval+ | +5.79 [2.44, 9.45] | 2/2 |
| MBPP | +2.58 [0.94, 4.22] | 2/2 |
| MBPP+ | +2.93 [1.05, 4.92] | 2/2 |
| LCB Full | +2.33 [0.82, 3.78] | 2/2 |
| LCB Easy | +5.73 [1.79, 9.32] | 2/2 |
| MultiPL-E | +3.96 [3.12, 4.85] | 2/2 |
| **Total** |  | **14/14** |

All 14 tests give `p < 0.05`, and every benchmark-level confidence interval excludes zero, supporting the higher Pass@1 of the high-density setting for both student models.

## RQ4: Knowledge Breadth

The focal comparison is Code+General at `9.7:0.3` versus pure Code at `10:0`, using four combinations of seed knowledge source and student model.

| Benchmark | Pass@1 difference [95% CI] | Tests with `p < 0.05` |
|---|---:|---:|
| HumanEval | +2.44 [1.14, 3.66] | 1/4 |
| HumanEval+ | +2.59 [1.22, 3.81] | 2/4 |
| MBPP | +0.82 [0.18, 1.52] | 0/4 |
| MBPP+ | +1.11 [0.20, 2.05] | 2/4 |
| LCB Full | +0.68 [0.37, 1.05] | 2/4 |
| LCB Easy | +1.79 [0.81, 2.69] | 1/4 |
| MultiPL-E | +3.07 [2.55, 3.58] | 4/4 |
| **Total** |  | **12/28** |

The mean Pass@1 difference is positive on all seven benchmark metrics, and all benchmark-level confidence intervals exclude zero. However, only 12 of the 28 individual tests are significant, so this result should be interpreted as an overall positive pattern rather than a statistically significant improvement in every configuration.
