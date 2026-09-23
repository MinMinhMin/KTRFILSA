"""Generate the seven resumable Kaggle phase notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


COMMON = r'''
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import zipfile

from kaggle_secrets import UserSecretsClient

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_OWNER = "MinMinhMin"
REPO_NAME = "KTRFILSA"
REPO_BRANCH = "K_check"
GITHUB_SECRET_NAME = "GITHUB_TOKEN"

HF_REPO_ID = "MinMinMinMin/KL"
HF_REPO_TYPE = "model"
HF_REPO_SUBDIR = "state"

KAGGLE_DATASET_ROOT = Path("/kaggle/input/datasets/minhvu111/dataset-kl")
WORK_ROOT = Path("/kaggle/working")
REPO_DIR = WORK_ROOT / REPO_NAME
OUTPUT_ROOT = WORK_ROOT / "camera_ready_outputs"

USE_DDP = True
GPU = "0,1"
DEVICE = "cuda:0"
DDP_GPUS = 2
DDP_MASTER_PORT = 29501
BATCH_SIZE = 128
MODEL_SEEDS = "12405,12406,12407"
K_VALUES = "3,4,5,6"
SELECTED_K = 3
KMEANS_SEEDS = 10
BOOTSTRAP_REPEATS = 10
ORDER_SHUFFLES = 20
METRIC_SAMPLE = 5000
TSNE_SAMPLE = 5000
NUM_EPOCHS = None
'''


CLONE = r'''
secrets = UserSecretsClient()
github_token = secrets.get_secret(GITHUB_SECRET_NAME)
if not github_token:
    raise RuntimeError(f"Kaggle Secret '{GITHUB_SECRET_NAME}' is empty.")

if REPO_DIR.exists():
    print(f"Reusing existing clone: {REPO_DIR}")
else:
    clone_url = f"https://x-access-token:{github_token}@github.com/{REPO_OWNER}/{REPO_NAME}.git"
    subprocess.run(["git", "clone", "--branch", REPO_BRANCH, clone_url, str(REPO_DIR)], check=True)
del github_token
if "clone_url" in globals():
    del clone_url
print("Repository ready:", REPO_DIR)
'''


INSTALL = r'''
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt"), "huggingface_hub"],
    check=True,
)
print("Dependencies installed")
'''


DATASET_SETUP = r'''
DATASET_TARGET_ROOT = REPO_DIR / "dataset"
DATASET_TARGET_ROOT.mkdir(parents=True, exist_ok=True)

def resolve_dataset_source(root, source_name):
    aliases = [source_name]
    if source_name == "XES3G5M0":
        aliases.append("XES3G5M")
    # Kaggle normally exposes the requested path verbatim, but the mounted
    # dataset can also contain one extra wrapper directory. Search both the
    # requested root and Kaggle's input root so a path-layout difference does
    # not turn into a dangling symlink failure later.
    search_roots = [root, Path("/kaggle/input"), root.parent]
    matches = {}
    for search_root in search_roots:
        if not search_root.exists():
            continue
        direct = search_root / source_name
        if direct.is_dir() and (direct / "preprocessed_df.csv").is_file():
            matches[direct.resolve()] = direct
        for processed in search_root.rglob("preprocessed_df.csv"):
            if processed.parent.name in aliases:
                matches[processed.parent.resolve()] = processed.parent
    if not matches:
        raise FileNotFoundError(f"Could not find {source_name}/preprocessed_df.csv under {root}")
    return sorted(matches.values(), key=lambda path: len(str(path)))[0]

if DATASET != "ALL_DATASETS":
    dataset_mapping = {
        "ASSISTMENT2009": "ASSISTMENT2009",
        "ASSISTMENT2017": "ASSISTMENT2017",
        "XES3G5M": "XES3G5M0",
    }
    source = resolve_dataset_source(KAGGLE_DATASET_ROOT, dataset_mapping[DATASET])
    destination = DATASET_TARGET_ROOT / DATASET
    if destination.is_symlink() and not destination.exists():
        destination.unlink()
    if not destination.exists():
        try:
            destination.symlink_to(source, target_is_directory=True)
        except (OSError, NotImplementedError):
            shutil.copytree(source, destination, dirs_exist_ok=True)
    elif not (destination / "preprocessed_df.csv").exists():
        # An existing empty directory is safe to populate; a non-empty
        # directory without the processed file is almost certainly stale and
        # should fail loudly instead of silently mixing datasets.
        if any(destination.iterdir()):
            raise RuntimeError(
                f"Dataset destination exists but is incomplete: {destination}"
            )
        shutil.copytree(source, destination, dirs_exist_ok=True)
    if not (destination / "preprocessed_df.csv").exists():
        raise FileNotFoundError(f"Missing processed data: {destination / 'preprocessed_df.csv'}")
    print(DATASET, "->", source)
'''


HF_RESTORE = r'''
from huggingface_hub import HfApi, login
from camera_ready_state import (
    build_phase_manifest,
    restore_hf_state,
    upload_hf_state,
    validate_analysis_outputs,
    validate_model_seed_outputs,
    validate_primary_outputs,
    write_phase_manifest,
)

hf_token = UserSecretsClient().get_secret("HF_TOKEN")
if not hf_token:
    raise RuntimeError("Kaggle Secret 'HF_TOKEN' is empty.")
login(token=hf_token, add_to_git_credential=False)
hf_api = HfApi()
del hf_token

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
if RESTORE_STATE:
    restore_hf_state(
        hf_api,
        HF_REPO_ID,
        HF_REPO_TYPE,
        WORK_ROOT / "hf_snapshot",
        OUTPUT_ROOT,
    )
print("State ready:", OUTPUT_ROOT)

def sync_phase_state(completed_jobs):
    manifest = build_phase_manifest(
        dataset=DATASET,
        phase=PHASE,
        repo_branch=REPO_BRANCH,
        completed_jobs=completed_jobs,
        settings={
            "model_seeds": MODEL_SEEDS,
            "k_values": [3, 4, 5, 6],
            "selected_k": SELECTED_K,
            "gpu": GPU,
            "ddp_gpus": DDP_GPUS,
            "batch_size": BATCH_SIZE,
        },
    )
    write_phase_manifest(OUTPUT_ROOT / DATASET / "phase_manifest.json", manifest)
    upload_hf_state(hf_api, HF_REPO_ID, HF_REPO_TYPE, OUTPUT_ROOT)
'''


PRIMARY_RUN = r'''
completed_jobs = []

def run_primary_job(slug):
    command = [
        sys.executable, "run_phase.py",
        "--dataset", DATASET,
        "--phase", "primary",
        "--run-slug", slug,
        "--output-dir", str(OUTPUT_ROOT),
        "--python", sys.executable,
        "--device", DEVICE,
        "--gpu", GPU,
        "--ddp-gpus", str(DDP_GPUS),
        "--ddp-master-port", str(DDP_MASTER_PORT),
        "--model-seeds", MODEL_SEEDS,
        "--kmeans-seeds", str(KMEANS_SEEDS),
        "--bootstrap-repeats", str(BOOTSTRAP_REPEATS),
        "--order-shuffles", str(ORDER_SHUFFLES),
        "--k-values", K_VALUES,
        "--selected-k", str(SELECTED_K),
        "--metric-sample", str(METRIC_SAMPLE),
        "--tsne-sample", str(TSNE_SAMPLE),
        "--batch-size", str(BATCH_SIZE),
    ]
    if USE_DDP:
        command.append("--ddp")
    if NUM_EPOCHS is not None:
        command.extend(["--num-epochs", str(NUM_EPOCHS)])
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_DIR, check=True)
    completed_jobs.append(slug)
    sync_phase_state(completed_jobs)

for slug in RUN_SLUGS:
    run_primary_job(slug)
print("Primary phase complete:", DATASET)
'''


FINISH_RUN = r'''
validate_primary_outputs(OUTPUT_ROOT, DATASET)
completed_jobs = ["primary"]

def run_model_seed(seed):
    command = [
        sys.executable, "run_phase.py",
        "--dataset", DATASET,
        "--phase", "model_seeds",
        "--target-seed", str(seed),
        "--output-dir", str(OUTPUT_ROOT),
        "--python", sys.executable,
        "--device", DEVICE,
        "--gpu", GPU,
        "--ddp-gpus", str(DDP_GPUS),
        "--ddp-master-port", str(DDP_MASTER_PORT),
        "--model-seeds", MODEL_SEEDS,
        "--kmeans-seeds", str(KMEANS_SEEDS),
        "--bootstrap-repeats", str(BOOTSTRAP_REPEATS),
        "--order-shuffles", str(ORDER_SHUFFLES),
        "--k-values", K_VALUES,
        "--selected-k", str(SELECTED_K),
        "--metric-sample", str(METRIC_SAMPLE),
        "--tsne-sample", str(TSNE_SAMPLE),
        "--batch-size", str(BATCH_SIZE),
    ]
    if USE_DDP:
        command.append("--ddp")
    if NUM_EPOCHS is not None:
        command.extend(["--num-epochs", str(NUM_EPOCHS)])
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_DIR, check=True)

for target_seed in [12406, 12407]:
    run_model_seed(target_seed)
    completed_jobs.append(f"model_seed_{target_seed}")
    sync_phase_state(completed_jobs)

validate_model_seed_outputs(OUTPUT_ROOT, DATASET, [12406, 12407])
analysis_command = [
    sys.executable, "run_phase.py",
    "--dataset", DATASET,
    "--phase", "analysis",
    "--output-dir", str(OUTPUT_ROOT),
    "--python", sys.executable,
    "--device", DEVICE,
    "--gpu", GPU,
    "--ddp-gpus", str(DDP_GPUS),
    "--ddp-master-port", str(DDP_MASTER_PORT),
    "--model-seeds", MODEL_SEEDS,
    "--kmeans-seeds", str(KMEANS_SEEDS),
    "--bootstrap-repeats", str(BOOTSTRAP_REPEATS),
    "--order-shuffles", str(ORDER_SHUFFLES),
    "--k-values", K_VALUES,
    "--selected-k", str(SELECTED_K),
    "--metric-sample", str(METRIC_SAMPLE),
    "--tsne-sample", str(TSNE_SAMPLE),
    "--batch-size", str(BATCH_SIZE),
]
if USE_DDP:
    analysis_command.append("--ddp")
print("Running:", " ".join(analysis_command), flush=True)
subprocess.run(analysis_command, cwd=REPO_DIR, check=True)
validate_analysis_outputs(OUTPUT_ROOT, DATASET)
completed_jobs.append("analysis")
sync_phase_state(completed_jobs)
print("Finish phase complete:", DATASET)
'''


FINAL = r'''
from huggingface_hub import HfApi, login
from camera_ready_state import (
    package_outputs,
    restore_hf_state,
    validate_final_dataset_outputs,
    validate_model_seed_outputs,
    validate_primary_outputs,
)

hf_token = UserSecretsClient().get_secret("HF_TOKEN")
if not hf_token:
    raise RuntimeError("Kaggle Secret 'HF_TOKEN' is empty.")
login(token=hf_token, add_to_git_credential=False)
hf_api = HfApi()
del hf_token

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
restore_hf_state(hf_api, HF_REPO_ID, HF_REPO_TYPE, WORK_ROOT / "hf_snapshot", OUTPUT_ROOT)
for dataset in ["ASSISTMENT2009", "ASSISTMENT2017", "XES3G5M"]:
    validate_primary_outputs(OUTPUT_ROOT, dataset)
    validate_model_seed_outputs(OUTPUT_ROOT, dataset, [12406, 12407])
    validate_final_dataset_outputs(
        OUTPUT_ROOT,
        dataset,
        {"k_values": [3, 4, 5, 6], "selected_k": 3},
    )

zip_path = WORK_ROOT / "camera_ready_outputs.zip"
package_outputs(OUTPUT_ROOT, zip_path)
HF_REPO_SUBDIR = "final"
upload_path = f"{HF_REPO_SUBDIR}/{zip_path.name}"
HfApi().upload_file(
    path_or_fileobj=str(zip_path),
    path_in_repo=upload_path,
    repo_id=HF_REPO_ID,
    repo_type=HF_REPO_TYPE,
)
print("Uploaded:", upload_path)
'''


def cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def write_notebook(path, title, dataset, phase, restore_state, run_slugs=None):
    cells = [
        cell("markdown", f"# {title}\n\nResumable camera-ready phase for {dataset}."),
        cell("code", COMMON + f'\nDATASET = "{dataset}"\nPHASE = "{phase}"\nRESTORE_STATE = {restore_state}\n'),
        cell("code", CLONE),
        cell("code", INSTALL),
        cell("code", DATASET_SETUP),
    ]
    # The final notebook restores once in FINAL before validation and
    # packaging. Avoid downloading the complete HF state a second time.
    if phase != "final":
        cells.append(cell("code", HF_RESTORE))
    if phase == "primary":
        slugs = run_slugs or []
        cells.append(cell("code", f"RUN_SLUGS = {slugs!r}\n" + PRIMARY_RUN))
    elif phase == "finish":
        cells.append(cell("code", FINISH_RUN))
    else:
        cells.append(cell("code", FINAL))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    write_notebook(
        ROOT / "01_assistment2009_primary.ipynb",
        "ASSISTMENT2009 primary",
        "ASSISTMENT2009",
        "primary",
        False,
        ["original", "soft", "soft_kmeans", "soft_separation_temporal", "full"],
    )
    write_notebook(
        ROOT / "02_assistment2009_finish.ipynb",
        "ASSISTMENT2009 finish",
        "ASSISTMENT2009",
        "finish",
        True,
    )
    write_notebook(
        ROOT / "03_assistment2017_primary.ipynb",
        "ASSISTMENT2017 primary",
        "ASSISTMENT2017",
        "primary",
        True,
        ["original", "soft", "soft_kmeans", "soft_separation_temporal", "full"],
    )
    write_notebook(
        ROOT / "04_assistment2017_finish.ipynb",
        "ASSISTMENT2017 finish",
        "ASSISTMENT2017",
        "finish",
        True,
    )
    write_notebook(
        ROOT / "05_xes3g5m_primary.ipynb",
        "XES3G5M primary",
        "XES3G5M",
        "primary",
        True,
        ["original", "soft", "soft_kmeans", "soft_separation_temporal", "full"],
    )
    write_notebook(
        ROOT / "06_xes3g5m_finish.ipynb",
        "XES3G5M finish",
        "XES3G5M",
        "finish",
        True,
    )
    write_notebook(
        ROOT / "07_final_package_upload.ipynb",
        "Final camera-ready package",
        "ALL_DATASETS",
        "final",
        True,
    )


if __name__ == "__main__":
    main()
