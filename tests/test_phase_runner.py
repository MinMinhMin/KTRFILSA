import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_parse_phase_args_accepts_primary_job_and_ddp_settings():
    from run_phase import parse_args

    args = parse_args(
        [
            "--dataset",
            "ASSISTMENT2009",
            "--phase",
            "primary",
            "--run-slug",
            "soft_kmeans",
            "--gpu",
            "0,1",
            "--ddp",
            "--ddp-gpus",
            "2",
            "--target-seed",
            "12405",
        ]
    )

    assert args.dataset == "ASSISTMENT2009"
    assert args.phase == "primary"
    assert args.run_slug == ["soft_kmeans"]
    assert args.gpu == "0,1"
    assert args.ddp is True
    assert args.ddp_gpus == 2
    assert args.target_seed == 12405


def test_phase_plan_rejects_unknown_primary_slug():
    from run_phase import build_phase_plan, parse_args

    args = parse_args(
        [
            "--dataset",
            "XES3G5M",
            "--phase",
            "primary",
            "--run-slug",
            "not_a_paper_run",
        ]
    )

    with pytest.raises(ValueError, match="Unsupported primary run slug"):
        build_phase_plan(args)


def test_model_seed_phase_requires_primary_outputs():
    from run_phase import build_phase_plan, parse_args

    args = parse_args(
        [
            "--dataset",
            "ASSISTMENT2017",
            "--phase",
            "model_seeds",
            "--target-seed",
            "12406",
        ]
    )

    plan = build_phase_plan(args)

    assert plan.phase == "model_seeds"
    assert plan.requires_primary_outputs is True
    assert plan.target_seed == 12406
    assert plan.run_slug == "soft_kmeans"


def test_analysis_phase_requires_primary_and_model_seed_outputs():
    from run_phase import build_phase_plan, parse_args

    args = parse_args(
        [
            "--dataset",
            "XES3G5M",
            "--phase",
            "analysis",
        ]
    )

    plan = build_phase_plan(args)

    assert plan.phase == "analysis"
    assert plan.requires_primary_outputs is True
    assert plan.requires_model_seed_outputs is True
    assert plan.run_slug is None


def test_dataset_output_dir_nests_pipeline_under_shared_state_root(tmp_path):
    from run_phase import dataset_output_dir

    assert dataset_output_dir(tmp_path, "XES3G5M") == tmp_path / "XES3G5M"
