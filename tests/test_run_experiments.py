import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_extraction_command_requests_all_k_and_selected_k():
    from run_experiments import extraction_command

    args = SimpleNamespace(
        config="configs/paper.yaml",
        dataset="XES3G5M",
        gpu="0",
        k_values=[3, 4, 5, 6],
        selected_k=3,
        metric_sample=5000,
        tsne_sample=5000,
        device="cpu",
        python=sys.executable,
    )
    command = extraction_command(args, Path("run"), 12405, Path("checkpoint"))
    assert command[command.index("--k_values") + 1] == "3,4,5,6"
    assert command[command.index("--selected_k") + 1] == "3"


def test_run_config_does_not_replace_explicit_selected_k(tmp_path):
    from run_experiments import write_run_config

    args = SimpleNamespace(
        output_dir=str(tmp_path),
        dataset="XES3G5M",
        model_seeds="12405",
        kmeans_seeds=10,
        bootstrap_repeats=10,
        order_shuffles=20,
        metric_sample=5000,
        device="cpu",
        gpu="0",
        batch_size=512,
        num_epochs=None,
        k_values=[3, 4, 5, 6],
        selected_k=3,
    )
    write_run_config(tmp_path, args, [12405])
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["posthoc_k_values"] == [3, 4, 5, 6]
    assert config["selected_k"] == 3
