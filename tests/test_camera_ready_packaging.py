import sys
import zipfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeHFAPI:
    def __init__(self, source):
        self.source = Path(source)
        self.snapshot_calls = []
        self.upload_calls = []

    def snapshot_download(self, **kwargs):
        self.snapshot_calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        source = self.source / "state"
        for path in source.rglob("*"):
            if path.is_file():
                destination = target / "state" / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
        return str(target)

    def upload_folder(self, **kwargs):
        self.upload_calls.append(kwargs)
        return {"path": kwargs["path_in_repo"]}


def test_restore_and_upload_hf_state_use_state_subtree(tmp_path):
    from camera_ready_state import restore_hf_state, upload_hf_state

    remote = tmp_path / "remote"
    (remote / "state" / "ASSISTMENT2009").mkdir(parents=True)
    (remote / "state" / "ASSISTMENT2009" / "checkpoint.bin").write_bytes(b"x")
    api = FakeHFAPI(remote)
    output = tmp_path / "outputs"
    snapshot = tmp_path / "snapshot"

    restore_hf_state(api, "MinMinMinMin/KL", "model", snapshot, output)
    upload_hf_state(api, "MinMinMinMin/KL", "model", output)

    assert (output / "ASSISTMENT2009" / "checkpoint.bin").exists()
    assert api.snapshot_calls[0]["repo_id"] == "MinMinMinMin/KL"
    assert api.upload_calls[0]["path_in_repo"] == "state"
    assert api.upload_calls[0]["repo_type"] == "model"


def test_package_outputs_contains_all_datasets_and_excludes_credentials(tmp_path):
    from camera_ready_state import package_outputs

    output = tmp_path / "outputs"
    for dataset in ["ASSISTMENT2009", "ASSISTMENT2017", "XES3G5M"]:
        path = output / dataset / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dataset, encoding="utf-8")
    (output / ".git").mkdir(parents=True)
    (output / ".git" / "config").write_text("secret", encoding="utf-8")
    (output / "HF_TOKEN").write_text("secret", encoding="utf-8")

    archive = tmp_path / "camera_ready_outputs.zip"
    package_outputs(output, archive)

    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
    assert "outputs/ASSISTMENT2009/manifest.json" in names
    assert "outputs/ASSISTMENT2017/manifest.json" in names
    assert "outputs/XES3G5M/manifest.json" in names
    assert not any(".git" in name or "HF_TOKEN" in name for name in names)
