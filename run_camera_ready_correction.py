#!/usr/bin/env python3
"""Run the isolated 45-job Camera-Ready correction campaign.

The existing training entry points are deliberately left unchanged.  This
launcher reads ``configs/camera_ready_correction.yaml`` and writes every
configuration/seed combination to its own directory:

    <output-root>/<dataset>/experiment_1/runs/<slug>/seed_<seed>/

It is resumable: a complete checkpoint is reused for training and a complete
K=3/4/5/6 extraction is reused for post-hoc clustering unless ``--force`` is
provided.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import yaml

from run_experiments import checkpoint_path, extraction_command, training_command


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "camera_ready_correction.yaml"


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    slug: str
    seed: int
    losses: str | None
    run_dir: Path


def _as_unique_ints(values, name):
    result = [int(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate values")
    return result


def load_camera_ready_config(config_path: Path | str = DEFAULT_CONFIG) -> dict:
    """Load and validate the standalone correction campaign configuration."""

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Camera-Ready config does not exist: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = document.get("camera_ready_correction")
    if not isinstance(settings, dict):
        raise ValueError(
            f"{path} must contain a camera_ready_correction mapping"
        )

    required = {
        "output_root",
        "datasets",
        "model_seeds",
        "k_values",
        "selected_k",
        "metric_sample",
        "tsne_sample",
        "batch_size",
        "eval_batch_size",
        "num_epochs",
        "device",
        "gpu",
        "ddp",
        "ddp_gpus",
        "ddp_master_port",
        "cpu_threads",
        "cpu_affinity",
        "skip_tsne",
        "configurations",
    }
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(f"Missing camera_ready_correction settings: {missing}")

    settings = dict(settings)
    settings["datasets"] = [str(value) for value in settings["datasets"]]
    settings["model_seeds"] = _as_unique_ints(
        settings["model_seeds"], "model_seeds"
    )
    settings["k_values"] = _as_unique_ints(settings["k_values"], "k_values")
    settings["selected_k"] = int(settings["selected_k"])
    if settings["selected_k"] not in settings["k_values"]:
        raise ValueError("selected_k must be one of k_values")
    if len(settings["model_seeds"]) != 3:
        raise ValueError("The correction campaign requires exactly three model seeds")
    if len(settings["datasets"]) != 3:
        raise ValueError("The correction campaign requires exactly three datasets")

    configurations = []
    seen_slugs = set()
    for item in settings["configurations"]:
        if not isinstance(item, dict) or "slug" not in item or "losses" not in item:
            raise ValueError("Each configuration must contain slug and losses")
        slug = str(item["slug"])
        if slug in seen_slugs:
            raise ValueError(f"Duplicate configuration slug: {slug}")
        seen_slugs.add(slug)
        losses = item["losses"]
        configurations.append(
            {"slug": slug, "losses": None if losses is None else str(losses)}
        )
    if len(configurations) != 5:
        raise ValueError("The correction campaign requires exactly five configurations")
    settings["configurations"] = configurations
    settings["config_path"] = path

    output_root = Path(settings["output_root"])
    settings["output_root"] = (
        output_root if output_root.is_absolute() else PROJECT_ROOT / output_root
    ).resolve()
    return settings


def iter_run_specs(settings: dict, datasets=None, slugs=None, seeds=None):
    selected_datasets = list(datasets or settings["datasets"])
    selected_slugs = set(slugs or [item["slug"] for item in settings["configurations"]])
    selected_seeds = list(seeds or settings["model_seeds"])
    losses_by_slug = {
        item["slug"]: item["losses"] for item in settings["configurations"]
    }
    for dataset in selected_datasets:
        for item in settings["configurations"]:
            if item["slug"] not in selected_slugs:
                continue
            for seed in selected_seeds:
                run_dir = (
                    settings["output_root"]
                    / dataset
                    / "experiment_1"
                    / "runs"
                    / item["slug"]
                    / f"seed_{seed}"
                )
                yield RunSpec(
                    dataset=dataset,
                    slug=item["slug"],
                    seed=int(seed),
                    losses=losses_by_slug[item["slug"]],
                    run_dir=run_dir,
                )


def _runtime_args(settings, dataset):
    return SimpleNamespace(
        python=sys.executable,
        config=str(settings["config_path"]),
        dataset=dataset,
        gpu=str(settings["gpu"]),
        batch_size=int(settings["batch_size"]),
        num_epochs=(
            None if settings["num_epochs"] is None else int(settings["num_epochs"])
        ),
        ddp=bool(settings["ddp"]),
        ddp_gpus=int(settings["ddp_gpus"]),
        ddp_master_port=int(settings["ddp_master_port"]),
        device=str(settings["device"]),
        k_values=list(settings["k_values"]),
        selected_k=int(settings["selected_k"]),
        metric_sample=int(settings["metric_sample"]),
        tsne_sample=int(settings["tsne_sample"]),
    )


def build_training_command(settings, dataset, seed, losses, run_dir: Path):
    """Build the same main.py command as the accepted pipeline, but isolated."""

    return training_command(
        _runtime_args(settings, dataset),
        run_dir,
        int(seed),
        losses,
    )


def build_extraction_command(settings, dataset, seed, run_dir: Path, checkpoint: Path):
    command = extraction_command(
        _runtime_args(settings, dataset),
        run_dir,
        int(seed),
        checkpoint,
    )
    if settings.get("skip_tsne", False):
        command.append("--skip_tsne")
    return command


def _latent_dir(spec: RunSpec):
    return spec.run_dir / "latent_clustering" / spec.dataset / "official"


def _required_latent_files(spec: RunSpec, k_values):
    latent_dir = _latent_dir(spec)
    return [
        latent_dir / "latent_features.npz",
        latent_dir / "latent_metadata.csv",
        latent_dir / "cluster_metrics.csv",
        latent_dir / "kmeans_model.pkl",
        *[
            latent_dir / f"cluster_assignments_k{k}.csv"
            for k in k_values
        ],
        *[latent_dir / f"kmeans_model_k{k}.pkl" for k in k_values],
    ]


def _checkpoint(spec: RunSpec):
    return checkpoint_path(spec.run_dir, spec.dataset)


def runtime_environment(settings):
    """Build a conservative child environment without changing model settings."""

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    threads = str(int(settings["cpu_threads"]))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = threads
    environment["TOKENIZERS_PARALLELISM"] = "false"
    # The command receives --gpu explicitly; do not inherit an unrelated
    # CUDA_VISIBLE_DEVICES mapping from the parent shell.
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    return environment


def _run_command(command, log_path: Path, settings, dry_run=False):
    command_to_run = list(command)
    affinity = str(settings.get("cpu_affinity", "")).strip()
    if affinity:
        taskset = shutil.which("taskset") or "taskset"
        command_to_run = [taskset, "-c", affinity, *command_to_run]
        if not dry_run and shutil.which("taskset") is None:
            raise FileNotFoundError("cpu_affinity is configured but taskset is unavailable")
    printable = " ".join(str(value) for value in command_to_run)
    print(printable, flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command_to_run,
            cwd=PROJECT_ROOT,
            env=runtime_environment(settings),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_spec(spec: RunSpec, settings: dict, phase: str, force=False, dry_run=False):
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _checkpoint(spec)
    if phase in {"train", "all"} and (force or checkpoint is None):
        _run_command(
            build_training_command(
                settings, spec.dataset, spec.seed, spec.losses, spec.run_dir
            ),
            spec.run_dir / "launcher_train.log",
            settings=settings,
            dry_run=dry_run,
        )
        if not dry_run:
            checkpoint = _checkpoint(spec)
            if checkpoint is None:
                raise FileNotFoundError(
                    f"Training produced no checkpoint for {spec.dataset}/{spec.slug}/seed_{spec.seed}"
                )
    elif phase == "extract" and checkpoint is None:
        raise FileNotFoundError(
            f"Extraction requires a checkpoint for {spec.dataset}/{spec.slug}/seed_{spec.seed}"
        )

    if phase in {"extract", "all"}:
        if checkpoint is None:
            checkpoint = _checkpoint(spec)
        if checkpoint is None:
            if dry_run:
                checkpoint = spec.run_dir / "checkpoints" / "cl4kt" / spec.dataset / "params_PENDING"
            else:
                raise FileNotFoundError(
                    f"No checkpoint available for {spec.dataset}/{spec.slug}/seed_{spec.seed}"
                )
        complete = all(
            path.exists()
            for path in _required_latent_files(spec, settings["k_values"])
        )
        if force or not complete:
            _run_command(
                build_extraction_command(
                    settings, spec.dataset, spec.seed, spec.run_dir, checkpoint
                ),
                spec.run_dir / "launcher_extract.log",
                settings=settings,
                dry_run=dry_run,
            )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--phase", choices=["train", "extract", "all"], default="all"
    )
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--run-slug", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--gpu")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--ddp-gpus", type=int)
    parser.add_argument("--ddp-master-port", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    settings = load_camera_ready_config(args.config)
    if args.gpu is not None:
        settings["gpu"] = args.gpu
    if args.ddp:
        settings["ddp"] = True
    if args.ddp_gpus is not None:
        settings["ddp_gpus"] = args.ddp_gpus
    if args.ddp_master_port is not None:
        settings["ddp_master_port"] = args.ddp_master_port
    if settings["ddp"] and int(settings["ddp_gpus"]) > 1 and "," not in str(settings["gpu"]):
        raise ValueError("DDP with multiple processes requires --gpu 0,1 (or another comma-separated mapping)")

    specs = list(
        iter_run_specs(
            settings,
            datasets=args.dataset,
            slugs=args.run_slug,
            seeds=args.seed,
        )
    )
    print(
        f"Camera-Ready correction campaign: {len(specs)} run(s), phase={args.phase}, "
        f"output={settings['output_root']}"
    )
    if not specs:
        raise ValueError("No runs selected")

    for index, spec in enumerate(specs, start=1):
        print(
            f"\n[{index}/{len(specs)}] {spec.dataset} / {spec.slug} / seed_{spec.seed}",
            flush=True,
        )
        run_spec(
            spec,
            settings,
            args.phase,
            force=args.force,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        manifest = {
            "config": str(settings["config_path"]),
            "output_root": str(settings["output_root"]),
            "datasets": sorted({spec.dataset for spec in specs}),
            "run_slugs": sorted({spec.slug for spec in specs}),
            "seeds": sorted({spec.seed for spec in specs}),
            "k_values": settings["k_values"],
            "selected_k": settings["selected_k"],
            "phase": args.phase,
            "run_count": len(specs),
        }
        manifest_path = settings["output_root"] / "camera_ready_correction_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
