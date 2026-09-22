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


def test_training_command_uses_torchrun_when_ddp_is_enabled():
    from run_experiments import training_command

    args = SimpleNamespace(
        python=sys.executable,
        config="configs/paper.yaml",
        dataset="XES3G5M",
        gpu="0,1",
        batch_size=128,
        num_epochs=1,
        ddp=True,
        ddp_gpus=2,
        ddp_master_port=29501,
    )
    command = training_command(args, Path("run"), 12405, "soft,kmeans")

    assert command[:4] == [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    assert command[command.index("--nproc_per_node") + 1] == "2"
    assert command[command.index("--master_port") + 1] == "29501"
    assert command[command.index("--gpu") + 1] == "0,1"
    assert "--ddp" in command


def test_training_command_remains_single_process_by_default():
    from run_experiments import training_command

    args = SimpleNamespace(
        python=sys.executable,
        config="configs/paper.yaml",
        dataset="XES3G5M",
        gpu="0",
        batch_size=512,
        num_epochs=None,
        ddp=False,
        ddp_gpus=1,
        ddp_master_port=29501,
    )
    command = training_command(args, Path("run"), 12405, None)

    assert command[:2] == [sys.executable, "main.py"]
    assert "--ddp" not in command
