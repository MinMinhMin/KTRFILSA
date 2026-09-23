from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_multiseed_cli_defaults_and_custom_roots():
    from run_multiseed_k_sensitivity import _parse_args

    args = _parse_args(
        [
            "--source-root",
            "/tmp/paper-results",
            "--output-root",
            "/tmp/k-sensitivity",
            "--dataset-root",
            "/tmp/data",
            "--datasets",
            "ASSISTMENT2009",
            "XES3G5M",
            "--seeds",
            "12405",
            "12407",
            "--k-values",
            "3,4,5,6",
        ]
    )

    assert args.source_root == "/tmp/paper-results"
    assert args.output_root == "/tmp/k-sensitivity"
    assert args.dataset_root == "/tmp/data"
    assert args.datasets == ["ASSISTMENT2009", "XES3G5M"]
    assert args.seeds == [12405, 12407]
    assert args.k_values == "3,4,5,6"


def test_multiseed_paths_support_primary_and_extra_seed_layouts():
    from run_multiseed_k_sensitivity import build_paths

    primary = build_paths(
        source_root=Path("/source"),
        output_root=Path("/output"),
        dataset_root=Path("/data"),
        dataset="ASSISTMENT2009",
        seed=12405,
        primary_seed=12405,
        primary_run_template="experiment_1/runs/soft_kmeans/seed_{seed}",
        extra_run_template="experiment_2/model_seed_runs/seed_{seed}",
        latent_template="latent_clustering/{dataset}/official",
        output_template="{dataset}/soft_kmeans/seed_{seed}",
        data_template="{dataset}/preprocessed_df.csv",
    )
    extra = build_paths(
        source_root=Path("/source"),
        output_root=Path("/output"),
        dataset_root=Path("/data"),
        dataset="ASSISTMENT2009",
        seed=12406,
        primary_seed=12405,
        primary_run_template="experiment_1/runs/soft_kmeans/seed_{seed}",
        extra_run_template="experiment_2/model_seed_runs/seed_{seed}",
        latent_template="latent_clustering/{dataset}/official",
        output_template="{dataset}/soft_kmeans/seed_{seed}",
        data_template="{dataset}/preprocessed_df.csv",
    )

    assert primary.clustering_dir == Path(
        "/source/ASSISTMENT2009/experiment_1/runs/soft_kmeans/seed_12405/latent_clustering/ASSISTMENT2009/official"
    )
    assert extra.clustering_dir == Path(
        "/source/ASSISTMENT2009/experiment_2/model_seed_runs/seed_12406/latent_clustering/ASSISTMENT2009/official"
    )
    assert primary.preprocessed_csv == Path("/data/ASSISTMENT2009/preprocessed_df.csv")
    assert extra.output_dir == Path("/output/ASSISTMENT2009/soft_kmeans/seed_12406")
