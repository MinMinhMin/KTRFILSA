import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PRIMARY_SLUGS = [
    "original",
    "soft",
    "soft_kmeans",
    "soft_separation_temporal",
    "full",
]


def _write(path, content="ok"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _create_primary(root, dataset, slug, seed=12405):
    run_root = root / dataset / "experiment_1" / "runs" / slug / f"seed_{seed}"
    _write(run_root / "checkpoints" / "cl4kt" / dataset / "params_46")
    latent = run_root / "latent_clustering" / dataset / "official"
    for filename in [
        "latent_features.npz",
        "cluster_metrics.csv",
        "kmeans_model.pkl",
        "cluster_assignments_k3.csv",
        "cluster_assignments_k4.csv",
        "cluster_assignments_k5.csv",
        "cluster_assignments_k6.csv",
        "kmeans_model_k3.pkl",
        "kmeans_model_k4.pkl",
        "kmeans_model_k5.pkl",
        "kmeans_model_k6.pkl",
    ]:
        _write(latent / filename)


def test_restore_state_merges_without_deleting_existing_dataset(tmp_path):
    from camera_ready_state import restore_state, state_root

    snapshot = tmp_path / "snapshot" / "state"
    output = tmp_path / "outputs"
    _write(snapshot / "ASSISTMENT2009" / "existing.csv")
    _write(output / "ASSISTMENT2017" / "keep.csv")

    restore_state(snapshot.parent, output)

    assert state_root(output) == output
    assert (output / "ASSISTMENT2009" / "existing.csv").exists()
    assert (output / "ASSISTMENT2017" / "keep.csv").exists()


def test_validate_primary_outputs_reports_missing_k_artifact(tmp_path):
    from camera_ready_state import validate_primary_outputs

    root = tmp_path / "outputs"
    for slug in PRIMARY_SLUGS:
        _create_primary(root, "XES3G5M", slug)
    missing = (
        root
        / "XES3G5M"
        / "experiment_1"
        / "runs"
        / "soft"
        / "seed_12405"
        / "latent_clustering"
        / "XES3G5M"
        / "official"
        / "cluster_assignments_k6.csv"
    )
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="XES3G5M.*soft.*k6"):
        validate_primary_outputs(root, "XES3G5M")


def test_validate_model_seed_outputs_requires_requested_seeds(tmp_path):
    from camera_ready_state import validate_model_seed_outputs

    root = tmp_path / "outputs"
    for seed in [12406, 12407]:
        run_root = root / "ASSISTMENT2009" / "experiment_2" / "model_seed_runs" / f"seed_{seed}"
        _write(run_root / "checkpoints" / "cl4kt" / "ASSISTMENT2009" / "params_46")
        latent = run_root / "latent_clustering" / "ASSISTMENT2009" / "official"
        for filename in [
            "latent_features.npz",
            "cluster_metrics.csv",
            "kmeans_model.pkl",
            "cluster_assignments_k3.csv",
            "cluster_assignments_k4.csv",
            "cluster_assignments_k5.csv",
            "cluster_assignments_k6.csv",
            "kmeans_model_k3.pkl",
            "kmeans_model_k4.pkl",
            "kmeans_model_k5.pkl",
            "kmeans_model_k6.pkl",
        ]:
            _write(latent / filename)

    (root / "ASSISTMENT2009" / "experiment_2" / "model_seed_runs" / "seed_12407").rename(
        root / "ASSISTMENT2009" / "experiment_2" / "model_seed_runs" / "seed_12408"
    )
    with pytest.raises(FileNotFoundError, match="seed_12407"):
        validate_model_seed_outputs(root, "ASSISTMENT2009", [12406, 12407])


def test_validate_analysis_outputs_rejects_wrong_dataset_manifest(tmp_path):
    from camera_ready_state import validate_analysis_outputs

    root = tmp_path / "outputs"
    manifest = root / "ASSISTMENT2017" / "phase_manifest.json"
    _write(
        manifest,
        json.dumps(
            {
                "dataset": "ASSISTMENT2009",
                "phase": "analysis",
                "selected_k": 3,
                "k_values": [3, 4, 5, 6],
            }
        ),
    )

    with pytest.raises(ValueError, match="dataset"):
        validate_analysis_outputs(root, "ASSISTMENT2017")


def test_phase_manifest_round_trip_and_settings_validation(tmp_path):
    from camera_ready_state import (
        build_phase_manifest,
        validate_manifest_settings,
        write_phase_manifest,
    )

    manifest = build_phase_manifest(
        dataset="XES3G5M",
        phase="primary",
        repo_branch="K_check",
        completed_jobs=["original"],
        settings={"k_values": [3, 4, 5, 6], "selected_k": 3},
    )
    path = tmp_path / "phase_manifest.json"
    write_phase_manifest(path, manifest)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest_settings(
        loaded,
        {"dataset": "XES3G5M", "k_values": [3, 4, 5, 6], "selected_k": 3},
    )
    with pytest.raises(ValueError, match="selected_k"):
        validate_manifest_settings(loaded, {"selected_k": 4})


def test_final_validation_requires_completed_analysis_manifest(tmp_path):
    from camera_ready_state import (
        build_phase_manifest,
        validate_final_dataset_outputs,
        write_phase_manifest,
    )

    root = tmp_path / "outputs"
    dataset_root = root / "ASSISTMENT2009"
    write_phase_manifest(
        dataset_root / "phase_manifest.json",
        build_phase_manifest(
            dataset="ASSISTMENT2009",
            phase="finish",
            repo_branch="K_check",
            completed_jobs=["primary"],
            settings={"k_values": [3, 4, 5, 6], "selected_k": 3},
        ),
    )

    with pytest.raises(ValueError, match="analysis"):
        validate_final_dataset_outputs(root, "ASSISTMENT2009")
