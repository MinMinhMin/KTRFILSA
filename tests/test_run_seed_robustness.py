from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_parse_csv_values_accepts_commas_and_rejects_duplicates():
    from run_seed_robustness import parse_csv_values

    assert parse_csv_values("12405, 12406 12407", cast=int) == [12405, 12406, 12407]

    with pytest.raises(ValueError, match="duplicate"):
        parse_csv_values("12405,12405", cast=int)


def test_aggregate_metrics_uses_sample_std_across_model_seeds():
    from run_seed_robustness import aggregate_metrics

    per_seed = pd.DataFrame(
        [
            {
                "dataset": "XES3G5M",
                "configuration": "original",
                "model_seed": 12405,
                "k": 3,
                "silhouette": 0.10,
                "dbi": 2.0,
                "ch": 100.0,
                "auc": 0.80,
            },
            {
                "dataset": "XES3G5M",
                "configuration": "original",
                "model_seed": 12406,
                "k": 3,
                "silhouette": 0.30,
                "dbi": 1.0,
                "ch": 200.0,
                "auc": 0.82,
            },
            {
                "dataset": "XES3G5M",
                "configuration": "original",
                "model_seed": 12407,
                "k": 3,
                "silhouette": 0.20,
                "dbi": 3.0,
                "ch": 300.0,
                "auc": 0.78,
            },
        ]
    )

    summary = aggregate_metrics(per_seed)
    row = summary.iloc[0]

    assert row["n_seeds"] == 3
    assert row["silhouette_mean"] == pytest.approx(0.20)
    assert row["silhouette_std"] == pytest.approx(0.10)
    assert row["dbi_mean"] == pytest.approx(2.0)
    assert row["ch_mean"] == pytest.approx(200.0)
    assert row["auc_mean"] == pytest.approx(0.80)
    assert row["auc_std"] == pytest.approx(0.02)


def test_format_summary_has_camera_ready_mean_plus_std_cells():
    from run_seed_robustness import aggregate_metrics, format_summary_cells

    per_seed = pd.DataFrame(
        [
            {
                "dataset": "ASSISTMENT2009",
                "configuration": "soft_kmeans",
                "model_seed": seed,
                "k": 3,
                "silhouette": value,
                "dbi": 1.0,
                "ch": 1000.0,
                "auc": 0.81,
            }
            for seed, value in [(12405, 0.2), (12406, 0.4), (12407, 0.3)]
        ]
    )

    row = format_summary_cells(aggregate_metrics(per_seed)).iloc[0]

    assert row["method"] == "Soft + KMeans"
    assert row["silhouette"] == "0.3000 ± 0.1000"
    assert row["dbi"] == "1.0000 ± 0.0000"
    assert row["ch"] == "1000.00 ± 0.00"
    assert row["auc"] == "0.8100 ± 0.0000"


def test_build_runner_command_selects_same_seed_set_for_both_methods():
    from run_seed_robustness import build_runner_command

    command = build_runner_command(
        python_executable="python",
        config_path=Path("configs/camera_ready_correction.yaml"),
        datasets=["XES3G5M"],
        configurations=["original", "soft_kmeans"],
        seeds=[12405, 12406, 12407],
        gpu="0,1",
        ddp=True,
        ddp_gpus=2,
        ddp_master_port=29501,
        force=False,
    )

    assert command[:3] == ["python", "run_camera_ready_correction.py", "--config"]
    assert command.count("--run-slug") == 2
    assert command.count("--seed") == 3
    assert "--dataset" in command
    assert "--ddp" in command
    assert "--ddp-gpus" in command
    assert "--force" not in command


def test_paper_config_is_preserved_and_orchestration_is_derived(tmp_path):
    import yaml

    from run_seed_robustness import _temporary_config_with_output_root

    paper_path = ROOT / "configs" / "paper.yaml"
    generated_path = _temporary_config_with_output_root(
        paper_path,
        tmp_path / "outputs",
        tmp_path,
    )
    original = yaml.safe_load(paper_path.read_text(encoding="utf-8"))
    generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))

    assert generated["cl4kt_config"] == original["cl4kt_config"]
    assert generated["train_config"] == original["train_config"]
    orchestration = generated["camera_ready_correction"]
    assert orchestration["batch_size"] == 512
    assert orchestration["num_epochs"] == 300
    assert orchestration["mask_prob"] == 0.2
    assert orchestration["permute_prob"] == 0.3
