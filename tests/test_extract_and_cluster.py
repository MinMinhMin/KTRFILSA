import joblib
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_parse_k_values_rejects_duplicates_and_non_positive_values():
    from extract_and_cluster import parse_k_values

    assert parse_k_values("3, 4,5,6") == [3, 4, 5, 6]
    with pytest.raises(ValueError, match="duplicate"):
        parse_k_values("3,3,4")
    with pytest.raises(ValueError, match="positive"):
        parse_k_values("0,3")


def test_evaluate_kmeans_writes_all_k_and_keeps_selected_k_canonical(tmp_path):
    from extract_and_cluster import evaluate_kmeans

    rng = np.random.RandomState(7)
    centers = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
    features = np.vstack([center + 0.05 * rng.randn(12, 2) for center in centers])
    metadata = pd.DataFrame(
        {
            "feature_index": np.arange(len(features)),
            "sequence_index": np.zeros(len(features), dtype=int),
            "user_id": np.repeat([1, 2, 3], 12),
            "step_index": np.tile(np.arange(12), 3),
        }
    )

    evaluate_kmeans(
        features,
        features,
        metadata,
        tmp_path,
        [3, 4, 5, 6],
        11,
        selected_k=3,
        metric_sample=0,
    )

    metrics = pd.read_csv(tmp_path / "cluster_metrics.csv")
    assert set(metrics["k"]) == {3, 4, 5, 6}
    assert int(metrics.loc[metrics["selected_k"], "k"].iloc[0]) == 3
    for k in [3, 4, 5, 6]:
        assert (tmp_path / f"kmeans_model_k{k}.pkl").exists()
        assert (tmp_path / f"cluster_assignments_k{k}.csv").exists()
        assert (tmp_path / f"representatives_k{k}.csv").exists()
    bundle = joblib.load(tmp_path / "kmeans_model.pkl")
    assert bundle["best_k"] == 3
    assert len(pd.read_csv(tmp_path / "cluster_assignments_k3.csv")) == len(features)
