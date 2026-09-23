"""Local state, prerequisite, manifest, and packaging helpers for Kaggle phases."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PRIMARY_SLUGS = (
    "original",
    "soft",
    "soft_kmeans",
    "soft_separation_temporal",
    "full",
)
K_VALUES = (3, 4, 5, 6)


def state_root(output_root):
    return Path(output_root)


def restore_state(snapshot_root, output_root):
    snapshot_root = Path(snapshot_root)
    output_root = Path(output_root)
    source = snapshot_root / "state" if (snapshot_root / "state").exists() else snapshot_root
    if not source.exists():
        raise FileNotFoundError(f"HF state directory does not exist: {source}")
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output_root, dirs_exist_ok=True)


def _run_root(output_root, dataset, slug, seed):
    output_root = state_root(output_root)
    if slug == "soft_kmeans" and seed != 12405:
        return output_root / dataset / "experiment_2" / "model_seed_runs" / f"seed_{seed}"
    return output_root / dataset / "experiment_1" / "runs" / slug / f"seed_{seed}"


def _required_run_outputs(output_root, dataset, slug, seed):
    run_root = _run_root(output_root, dataset, slug, seed)
    checkpoint_dir = run_root / "checkpoints" / "cl4kt" / dataset
    latent_dir = run_root / "latent_clustering" / dataset / "official"
    paths = list(checkpoint_dir.glob("params_*"))
    paths += [
        latent_dir / "latent_features.npz",
        latent_dir / "cluster_metrics.csv",
        latent_dir / "kmeans_model.pkl",
    ]
    paths += [latent_dir / f"cluster_assignments_k{k}.csv" for k in K_VALUES]
    paths += [latent_dir / f"kmeans_model_k{k}.pkl" for k in K_VALUES]
    return paths


def _validate_paths(paths, description):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {description}: {missing}")
    return [Path(path) for path in paths]


def validate_primary_outputs(output_root, dataset):
    outputs = []
    for slug in PRIMARY_SLUGS:
        paths = _required_run_outputs(output_root, dataset, slug, 12405)
        outputs.extend(_validate_paths(paths, f"primary outputs for {dataset}/{slug}/seed_12405"))
    return outputs


def validate_model_seed_outputs(output_root, dataset, seeds):
    outputs = []
    for seed in seeds:
        paths = _required_run_outputs(output_root, dataset, "soft_kmeans", int(seed))
        outputs.extend(
            _validate_paths(
                paths,
                f"model-seed outputs for {dataset}/soft_kmeans/seed_{seed}",
            )
        )
    return outputs


def validate_analysis_outputs(output_root, dataset):
    manifest_path = state_root(output_root) / dataset / "phase_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset") != dataset:
            raise ValueError(
                f"phase manifest dataset {manifest.get('dataset')!r} does not match {dataset!r}"
            )
    root = state_root(output_root) / dataset
    required = [
        root / "experiment_1" / "k_sensitivity" / "cluster_metrics.csv",
        root / "experiment_1" / "k_sensitivity" / "interpretability_summary.csv",
        root / "experiment_1" / "k_sensitivity" / "interpretability_feature_effects.csv",
        root / "experiment_1" / "k_sensitivity" / "interpretability_sensitivity.md",
        root / "experiment_2" / "kmeans_seed_metrics.csv",
        root / "experiment_2" / "model_seed_metrics.csv",
        root / "experiment_3" / "cluster_interpretation_summary.csv",
        root / "experiment_4" / "trajectory_summary.csv",
    ]
    return _validate_paths(required, f"analysis outputs for {dataset}")


def validate_final_dataset_outputs(output_root, dataset, expected_settings=None):
    """Validate the completed manifest and analysis tree for final packaging."""
    manifest_path = state_root(output_root) / dataset / "phase_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing completed phase manifest for final packaging: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != dataset:
        raise ValueError(
            f"Final manifest dataset {manifest.get('dataset')!r} does not match {dataset!r}"
        )
    completed_jobs = manifest.get("completed_jobs", [])
    if "analysis" not in completed_jobs:
        raise ValueError(
            f"Final manifest for {dataset} is incomplete; missing completed job 'analysis'"
        )
    if expected_settings:
        validate_manifest_settings(manifest, expected_settings)
    return validate_analysis_outputs(output_root, dataset)


def build_phase_manifest(*, dataset, phase, repo_branch, completed_jobs, settings):
    return {
        "dataset": dataset,
        "phase": phase,
        "repo_branch": repo_branch,
        "completed_jobs": list(completed_jobs),
        "settings": dict(settings),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_phase_manifest(path, manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_manifest_settings(manifest, expected):
    for key, value in expected.items():
        actual = manifest.get(key)
        if actual is None and isinstance(manifest.get("settings"), dict):
            actual = manifest["settings"].get(key)
        if actual != value:
            raise ValueError(f"Manifest setting {key!r} is {actual!r}, expected {value!r}")


def restore_hf_state(api, repo_id, repo_type, snapshot_root, output_root):
    downloader = getattr(api, "snapshot_download", None)
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    downloader(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(snapshot_root),
        allow_patterns=["state/**", "manifests/**"],
    )
    restore_state(snapshot_root, output_root)


def upload_hf_state(api, repo_id, repo_type, output_root, path_in_repo="state"):
    return api.upload_folder(
        folder_path=str(output_root),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )


def package_outputs(output_root, zip_path):
    output_root = Path(output_root)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if not path.is_file():
                continue
            if ".git" in path.parts or path.name in {"HF_TOKEN", "GITHUB_TOKEN"}:
                continue
            archive.write(path, path.relative_to(output_root.parent))
    return zip_path
