import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_camera_ready_configuration_declares_complete_45_job_matrix():
    from run_camera_ready_correction import (
        iter_run_specs,
        load_camera_ready_config,
    )

    settings = load_camera_ready_config(
        ROOT / "configs" / "camera_ready_correction.yaml"
    )
    specs = list(iter_run_specs(settings))

    assert len(specs) == 45
    assert {spec.dataset for spec in specs} == {
        "ASSISTMENT2009",
        "ASSISTMENT2017",
        "XES3G5M",
    }
    assert {spec.slug for spec in specs} == {
        "original",
        "soft",
        "soft_kmeans",
        "soft_separation_temporal",
        "full",
    }
    assert {spec.seed for spec in specs} == {12405, 12406, 12407}
    assert len({spec.run_dir for spec in specs}) == 45


def test_camera_ready_training_command_isolated_from_existing_output(tmp_path):
    from run_camera_ready_correction import (
        build_training_command,
        load_camera_ready_config,
    )

    settings = load_camera_ready_config(
        ROOT / "configs" / "camera_ready_correction.yaml"
    )
    run_dir = tmp_path / "ASSISTMENT2009" / "experiment_1" / "runs" / "full" / "seed_12407"
    command = build_training_command(
        settings,
        dataset="ASSISTMENT2009",
        seed=12407,
        losses="soft,kmeans,separation,temporal",
        run_dir=run_dir,
    )

    assert command[command.index("--config") + 1].endswith(
        "configs/camera_ready_correction.yaml"
    )
    assert command[command.index("--seed") + 1] == "12407"
    assert command[command.index("--checkpoint_dir") + 1] == str(
        run_dir / "checkpoints"
    )
    assert "--use_joint_training_module" in command
    assert "--ddp" not in command


def test_camera_ready_analysis_discovers_all_five_configs_and_three_seeds(tmp_path):
    from camera_ready_correction_analysis import (
        correction_run_dir,
        expected_latent_files,
    )

    run_dir = correction_run_dir(
        tmp_path, "XES3G5M", "soft_kmeans", 12406
    )
    assert run_dir == (
        tmp_path
        / "XES3G5M"
        / "experiment_1"
        / "runs"
        / "soft_kmeans"
        / "seed_12406"
    )
    files = expected_latent_files(
        run_dir / "latent_clustering" / "XES3G5M" / "official",
        [3, 4, 5, 6],
    )
    assert files[0].name == "latent_features.npz"
    assert files[-1].name == "kmeans_model_k6.pkl"


def test_camera_ready_analysis_rejects_missing_full_seed_artifact(tmp_path):
    from camera_ready_correction_analysis import validate_artifact_tree

    with pytest.raises(FileNotFoundError, match="seed_12407"):
        validate_artifact_tree(
            tmp_path,
            "ASSISTMENT2017",
            ["full"],
            [12407],
            [3, 4, 5, 6],
        )


def test_analysis_cli_separates_input_and_output_roots():
    from camera_ready_correction_analysis import _parse_args

    args = _parse_args([
        "--source-root",
        "paper_result",
        "--analysis-output-root",
        "camera_ready_correction_outputs",
    ])

    assert args.source_root == "paper_result"
    assert args.analysis_output_root == "camera_ready_correction_outputs"


def test_analysis_cli_keeps_legacy_output_root_behavior():
    from camera_ready_correction_analysis import _parse_args

    args = _parse_args(["--output-root", "legacy_results"])

    assert args.legacy_output_root == "legacy_results"


def test_camera_ready_runtime_limits_preserve_reproducibility_and_reserve_cpu():
    from run_camera_ready_correction import (
        load_camera_ready_config,
        runtime_environment,
    )

    settings = load_camera_ready_config(
        ROOT / "configs" / "camera_ready_correction.yaml"
    )

    assert settings["gpu"] == "0"
    assert settings["ddp"] is False
    assert settings["batch_size"] == 512
    assert settings["num_epochs"] == 300
    assert settings["metric_sample"] == 5000
    assert settings["eval_batch_size"] == 2048
    assert settings["k_values"] == [3, 4, 5, 6]
    assert settings["selected_k"] == 3
    assert settings["cpu_threads"] == 4
    assert settings["cpu_affinity"] == "0-3"

    environment = runtime_environment(settings)
    assert environment["OMP_NUM_THREADS"] == "4"
    assert environment["MKL_NUM_THREADS"] == "4"
    assert environment["OPENBLAS_NUM_THREADS"] == "4"


def test_camera_ready_fast_profile_keeps_tsne_and_only_increases_safe_eval_work():
    from run_camera_ready_correction import (
        build_extraction_command,
        iter_run_specs,
        load_camera_ready_config,
    )

    settings = load_camera_ready_config(
        ROOT / "configs" / "camera_ready_correction.yaml"
    )
    spec = next(iter_run_specs(settings))
    command = build_extraction_command(
        settings,
        spec.dataset,
        spec.seed,
        spec.run_dir,
        spec.run_dir / "checkpoint",
    )

    assert settings["skip_tsne"] is False
    assert "--skip_tsne" not in command
    assert settings["metric_sample"] == 5000
    assert settings["selected_k"] == 3
