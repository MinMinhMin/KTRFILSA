#!/usr/bin/env python3
"""Run one resumable camera-ready phase/job for one dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from extract_and_cluster import parse_k_values
from camera_ready_state import (
    validate_model_seed_outputs,
    validate_primary_outputs,
)
from run_experiments import (
    EXPERIMENT_1_RUNS,
    Pipeline,
    ensure_model_run,
    parse_seeds,
    resolved_config_path,
    source_root,
    training_command,
    validate_inputs,
    write_run_config,
)


PRIMARY_SLUGS = {slug for slug, _, _ in EXPERIMENT_1_RUNS}
MODEL_SEED_VALUES = {12406, 12407}


def dataset_output_dir(output_root, dataset):
    return Path(output_root).resolve() / dataset


@dataclass(frozen=True)
class PhasePlan:
    phase: str
    dataset: str
    run_slug: str | None = None
    target_seed: int | None = None
    requires_primary_outputs: bool = False
    requires_model_seed_outputs: bool = False


def _split_values(values):
    result = []
    for value in values or []:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=["primary", "model_seeds", "analysis"],
    )
    parser.add_argument(
        "--run-slug",
        action="append",
        help="Experiment 1 slug; repeat or provide comma-separated values.",
    )
    parser.add_argument("--target-seed", type=int)
    parser.add_argument("--output-dir", default="paper_experiment_results")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--ddp-gpus", type=int, default=2)
    parser.add_argument("--ddp-master-port", type=int, default=29501)
    parser.add_argument("--model-seeds", default="12405,12406,12407")
    parser.add_argument("--kmeans-seeds", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=10)
    parser.add_argument("--order-shuffles", type=int, default=20)
    parser.add_argument("--k-values", default="3,4,5,6")
    parser.add_argument("--selected-k", type=int, default=3)
    parser.add_argument("--metric-sample", type=int, default=5000)
    parser.add_argument("--tsne-sample", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_phase_plan(args):
    run_slugs = _split_values(args.run_slug)
    if args.phase == "primary":
        if not run_slugs:
            raise ValueError("primary phase requires at least one --run-slug")
        unsupported = sorted(set(run_slugs) - PRIMARY_SLUGS)
        if unsupported:
            raise ValueError(
                "Unsupported primary run slug(s): " + ", ".join(unsupported)
            )
        if args.target_seed not in (None, 12405):
            raise ValueError("primary phase only supports target seed 12405")
        if len(run_slugs) > 1:
            raise ValueError(
                "Run one primary slug per invocation so its output can be uploaded"
            )
        return PhasePlan(
            phase=args.phase,
            dataset=args.dataset,
            run_slug=run_slugs[0],
            target_seed=12405,
        )

    if args.phase == "model_seeds":
        if args.run_slug and set(run_slugs) != {"soft_kmeans"}:
            raise ValueError("model_seeds phase only supports --run-slug soft_kmeans")
        if args.target_seed not in MODEL_SEED_VALUES:
            raise ValueError(
                "model_seeds phase requires --target-seed 12406 or 12407"
            )
        return PhasePlan(
            phase=args.phase,
            dataset=args.dataset,
            run_slug="soft_kmeans",
            target_seed=args.target_seed,
            requires_primary_outputs=True,
        )

    if run_slugs or args.target_seed is not None:
        raise ValueError("analysis phase does not accept --run-slug or --target-seed")
    return PhasePlan(
        phase=args.phase,
        dataset=args.dataset,
        requires_primary_outputs=True,
        requires_model_seed_outputs=True,
    )


def _latent_dir(output_root, dataset, slug, seed):
    if slug == "soft_kmeans" and seed != 12405:
        run_root = output_root / "experiment_2" / "model_seed_runs" / f"seed_{seed}"
    else:
        run_root = output_root / "experiment_1" / "runs" / slug / f"seed_{seed}"
    return run_root / "latent_clustering" / dataset / "official"


def _checkpoint_dir(output_root, dataset, slug, seed):
    if slug == "soft_kmeans" and seed != 12405:
        run_root = output_root / "experiment_2" / "model_seed_runs" / f"seed_{seed}"
    else:
        run_root = output_root / "experiment_1" / "runs" / slug / f"seed_{seed}"
    return run_root / "checkpoints" / "cl4kt" / dataset


def _validate_latent_artifacts(output_root, dataset, slug, seed):
    latent_dir = _latent_dir(output_root, dataset, slug, seed)
    required = [
        latent_dir / "latent_features.npz",
        latent_dir / "cluster_metrics.csv",
        latent_dir / "kmeans_model.pkl",
    ] + [latent_dir / f"cluster_assignments_k{k}.csv" for k in (3, 4, 5, 6)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing latent prerequisites for {dataset}/{slug}/seed_{seed}: {missing}"
        )


def validate_phase_prerequisites(args, plan):
    output_root = Path(args.output_dir).resolve()
    if plan.requires_primary_outputs:
        validate_primary_outputs(output_root, plan.dataset)
    if plan.requires_model_seed_outputs:
        validate_model_seed_outputs(output_root, plan.dataset, [12406, 12407])


def _base_run_args(args):
    args.k_values = parse_k_values(args.k_values)
    args.model_seeds = ",".join(str(seed) for seed in parse_seeds(args.model_seeds))
    args.dataset = args.dataset
    return args


def _run_analysis(args):
    command = [
        args.python,
        str(source_root() / "run_experiments.py"),
        "--dataset",
        args.dataset,
        "--output-dir",
        str(dataset_output_dir(args.output_dir, args.dataset)),
        "--config",
        args.config,
        "--python",
        args.python,
        "--device",
        args.device,
        "--gpu",
        args.gpu,
        "--model-seeds",
        args.model_seeds,
        "--ddp-gpus",
        str(args.ddp_gpus),
        "--ddp-master-port",
        str(args.ddp_master_port),
        "--kmeans-seeds",
        str(args.kmeans_seeds),
        "--bootstrap-repeats",
        str(args.bootstrap_repeats),
        "--order-shuffles",
        str(args.order_shuffles),
        "--k-values",
        ",".join(map(str, args.k_values)),
        "--selected-k",
        str(args.selected_k),
        "--metric-sample",
        str(args.metric_sample),
        "--tsne-sample",
        str(args.tsne_sample),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.num_epochs is not None:
        command.extend(["--num-epochs", str(args.num_epochs)])
    if args.ddp:
        command.append("--ddp")
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    print("[analysis]", " ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=source_root(), check=True)


def execute_phase(args):
    args = _base_run_args(args)
    plan = build_phase_plan(args)
    overall_output_root = Path(args.output_dir).resolve()
    validate_inputs(args)
    validate_phase_prerequisites(args, plan)
    if plan.phase == "analysis":
        _run_analysis(args)
        return plan

    args.output_dir = str(dataset_output_dir(overall_output_root, args.dataset))
    seeds = parse_seeds(args.model_seeds)
    pipeline = Pipeline(args)
    write_run_config(pipeline.root, args, seeds)

    if plan.phase == "primary":
        losses = dict((slug, losses) for slug, _, losses in EXPERIMENT_1_RUNS)[
            plan.run_slug
        ]
        run_dir = (
            pipeline.root
            / "experiment_1"
            / "runs"
            / plan.run_slug
            / "seed_12405"
        )
        ensure_model_run(
            pipeline,
            run_dir,
            12405,
            losses,
            f"experiment_1/{plan.run_slug}/seed_12405",
        )
        return plan

    run_dir = (
        pipeline.root
        / "experiment_2"
        / "model_seed_runs"
        / f"seed_{plan.target_seed}"
    )
    ensure_model_run(
        pipeline,
        run_dir,
        plan.target_seed,
        "soft,kmeans",
        f"experiment_2/model_seed_{plan.target_seed}",
    )
    return plan


def main(argv=None):
    args = parse_args(argv)
    execute_phase(args)


if __name__ == "__main__":
    main()
