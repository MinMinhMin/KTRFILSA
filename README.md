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
python run_all_datasets.py \
  --output-root paper_results \
  --k-values 3,4,5,6 \
  --selected-k 3
```

Run one dataset:

```bash
python run_experiments.py \
  --dataset XES3G5M \
  --output-dir paper_results/XES3G5M \
  --k-values 3,4,5,6 \
  --selected-k 3
```

`--k-values` controls the post-hoc KMeans sensitivity experiment. The default
candidate values are `3,4,5,6`. `--selected-k 3` keeps K=3 as the canonical
state granularity consumed by the paper's trajectory and order-shuffle
experiments. The pipeline also reports `metric_best_k`, the K ranked by
internal geometry, but that value never silently replaces the explicit
selected K.

The default model seeds are `12405`, `12406`, and `12407`. At most three model seeds are used.

Use `--dry-run` to inspect the execution plan without training:

```bash
python run_all_datasets.py --dry-run
```

The experiment runner is resumable. Existing completed stages are reused unless `--force` is supplied.

For a Kaggle T4×2 runtime, the camera-ready notebook enables distributed
training with one process per GPU. The training batch size is per GPU, so a
safe starting point is 128:

```bash
python run_all_datasets.py \
  --gpu 0,1 --device cuda:0 --ddp --ddp-gpus 2 \
  --batch-size 128
```

DDP is used only for model training; extraction and the post-hoc analysis run
once on `cuda:0`. Checkpoints and logs are written by rank 0, while validation
and test predictions are gathered across workers.

## Outputs

Each dataset produces separate directories for representation quality, stability, pedagogical analysis, and trajectory analysis. Outputs include CSV tables, trained checkpoints, cluster models, logs, and figures.

The K-sensitivity stage is stored under:

```text
experiment_1/
├── k_sensitivity.csv                         # legacy metric-table alias
├── k_sensitivity/
│   ├── cluster_metrics.csv                   # Silhouette, Davies–Bouldin, Calinski–Harabasz
│   ├── interpretability_summary.csv          # per-K/per-cluster profiles
│   ├── interpretability_feature_effects.csv  # internal feature separation and p-values
│   ├── interpretability_sensitivity.md       # camera-ready interpretation
│   └── k_{3,4,5,6}/                          # detailed per-K profiles/reports
└── runs/.../latent_clustering/.../
    ├── kmeans_model_k{K}.pkl
    ├── cluster_assignments_k{K}.csv
    └── representatives_k{K}.csv
```

The Markdown report treats states as descriptive and correctness-correlated,
not diagnostic cognitive labels. It also records that this validation is
internal to the representation/interaction data and does not establish a link
to later course performance, dropout, or independent instructor judgment.

## Kaggle camera-ready workflow

The notebook `notebooks/camera_ready_kaggle_hf.ipynb` is ready for Kaggle.
Set `REPO_OWNER`, `REPO_NAME`, and `REPO_BRANCH` in its first code cell. It
reads the GitHub token from the Kaggle secret `GITHUB_TOKEN`, maps:

```text
/kaggle/input/datasets/minhvu111/dataset-kl/ASSISTMENT2009 -> dataset/ASSISTMENT2009
/kaggle/input/datasets/minhvu111/dataset-kl/ASSISTMENT2017 -> dataset/ASSISTMENT2017
/kaggle/input/datasets/minhvu111/dataset-kl/XES3G5M0       -> dataset/XES3G5M
```

It runs the full K=3,4,5,6 pipeline, packages the complete output and report
files, then reads `HF_TOKEN` from Kaggle Secrets and uploads the zip to
`MinMinMinMin/KL` with `HF_REPO_TYPE = "model"`. No token is stored in the
repository or the uploaded archive.

Third-party code and attributions are listed in `THIRD_PARTY.txt`.
