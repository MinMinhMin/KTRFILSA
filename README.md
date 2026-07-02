# Knowledge Tracing Representations for Interpretable Learning-State Trajectory Analysis

This repository contains the code for **Knowledge Tracing Representations for Interpretable Learning-State Trajectory Analysis**.

The project uses timestep-level hidden representations from CL4KT to discover latent learning states. A clustering-oriented objective shapes the representation space during training, while an independent post-hoc k-means model produces the final state assignments. The resulting state sequences are used to analyze student trajectories and transition patterns.

## Experiments

The code runs four groups of experiments:

1. representation quality and ablation;
2. cluster stability across model seeds, k-means seeds, and student bootstrap samples;
3. pedagogical interpretation and order-shuffling control;
4. learning-state trajectory and transition analysis.

The supported datasets are:

- XES3G5M;
- ASSISTments 2009;
- ASSISTments 2017.

## Raw Data Sources

Raw datasets are not redistributed in this repository. Download them from the original sources and place them under `dataset/` before running the preprocessing scripts.

| Dataset | Raw data source |
| --- | --- |
| XES3G5M | https://github.com/ai4ed/XES3G5M |
| ASSISTments 2009 | http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip |
| ASSISTments 2017 | http://base.ustc.edu.cn/data/ASSISTment/anonymized_full_release_competition_dataset.zip |

## Setup

```bash
pip install -r requirements.txt
```

Place each dataset under:

```text
dataset/
├── XES3G5M/
├── ASSISTMENT2009/
└── ASSISTMENT2017/
```

Each processed dataset directory must contain `preprocessed_df.csv`. Dataset preparation scripts are provided in `preprocessing/`.

## Running the experiments

Run all three datasets:

```bash
python run_all_datasets.py --output-root paper_results
```

Run one dataset:

```bash
python run_experiments.py \
  --dataset XES3G5M \
  --output-dir paper_results/XES3G5M
```

The default model seeds are `12405`, `12406`, and `12407`. At most three model seeds are used.

Use `--dry-run` to inspect the execution plan without training:

```bash
python run_all_datasets.py --dry-run
```

The experiment runner is resumable. Existing completed stages are reused unless `--force` is supplied.

## Outputs

Each dataset produces separate directories for representation quality, stability, pedagogical analysis, and trajectory analysis. Outputs include CSV tables, trained checkpoints, cluster models, logs, and figures.

Third-party code and attributions are listed in `THIRD_PARTY.txt`.
