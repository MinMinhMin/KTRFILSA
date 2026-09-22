import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_synthetic_fixture(tmp_path):
    from extract_and_cluster import evaluate_kmeans

    dataset_dir = tmp_path / "dataset" / "XES3G5M"
    dataset_dir.mkdir(parents=True)
    rows = []
    for user_id in range(4):
        for step in range(6):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": step % 3,
                    "correct": int((user_id + step) % 2 == 0),
                    "skill_id": step % 2,
                    "split": "test",
                    "source_item_id": str(step % 3),
                    "source_skill_id": str(step % 2),
                }
            )
    data_path = dataset_dir / "preprocessed_df.csv"
    pd.DataFrame(rows).to_csv(data_path, sep="\t", index=False)

    config_path = tmp_path / "paper.yaml"
    config_path.write_text(
        "dataset_path: '"
        + str(tmp_path / "dataset")
        + "'\ntrain_config:\n  seq_len: 6\n  sequence_option: recent\n",
        encoding="utf-8",
    )
    clustering_dir = tmp_path / "clustering" / "XES3G5M" / "official"
    clustering_dir.mkdir(parents=True)
    rng = np.random.RandomState(3)
    features = np.vstack(
        [
            np.array([0.0, 0.0]) + 0.05 * rng.randn(6, 2),
            np.array([3.0, 3.0]) + 0.05 * rng.randn(6, 2),
            np.array([6.0, 0.0]) + 0.05 * rng.randn(6, 2),
            np.array([9.0, 3.0]) + 0.05 * rng.randn(6, 2),
        ]
    ).astype("float32")
    metadata = pd.DataFrame(
        {
            "feature_index": np.arange(24),
            "sequence_index": np.repeat(np.arange(4), 6),
            "user_id": np.repeat(np.arange(4), 6),
            "step_index": np.tile(np.arange(6), 4),
            "item_id": np.tile(np.arange(6), 4),
            "skill_id": np.tile(np.arange(6), 4) % 2,
            "correct": [row["correct"] for row in rows],
        }
    )
    metadata.to_csv(clustering_dir / "latent_metadata.csv", index=False)
    evaluate_kmeans(
        features,
        features,
        metadata,
        clustering_dir,
        [3, 4],
        13,
        selected_k=3,
        metric_sample=0,
    )
    return data_path, clustering_dir, config_path


def test_sensitivity_analysis_writes_all_k_tables_and_report(tmp_path):
    from analyze_cluster_sensitivity import run_sensitivity

    data_path, clustering_dir, config_path = make_synthetic_fixture(tmp_path)
    output_dir = tmp_path / "sensitivity"
    run_sensitivity(
        preprocessed_csv=data_path,
        clustering_dir=clustering_dir,
        config_path=config_path,
        output_dir=output_dir,
        k_values=[3, 4],
        selected_k=3,
        timestamp_unit="ordinal",
    )

    assert set(pd.read_csv(output_dir / "cluster_metrics.csv")["k"]) == {3, 4}
    summary = pd.read_csv(output_dir / "interpretability_summary.csv")
    assert set(summary["k"]) == {3, 4}
    assert {
        "mean_correct",
        "avg_skill_difficulty",
        "pedagogical_label_suggestion",
    }.issubset(summary.columns)
    effects = pd.read_csv(output_dir / "interpretability_feature_effects.csv")
    assert effects["feature"].notna().all()
    assert (
        effects.loc[
            effects["feature"] == "median_time_gap_sec", "status"
        ].iloc[0]
        == "missing"
    )
    report = (output_dir / "interpretability_sensitivity.md").read_text()
    assert "K=3" in report and "Silhouette" in report
    assert "descriptive" in report.lower()
