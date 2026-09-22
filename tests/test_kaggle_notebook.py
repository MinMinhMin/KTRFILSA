import json
from pathlib import Path


def test_camera_ready_notebook_contains_required_secrets_mapping_and_upload():
    notebook = json.loads(
        Path("notebooks/camera_ready_kaggle_hf.ipynb").read_text()
    )
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for token in [
        "REPO_OWNER",
        "REPO_NAME",
        "REPO_BRANCH",
        'GITHUB_SECRET_NAME = "GITHUB_TOKEN"',
        'HF_REPO_ID = "MinMinMinMin/KL"',
        'HF_REPO_TYPE = "model"',
        "HF_TOKEN",
        "XES3G5M0",
        "--k-values",
        "--selected-k",
        "HfApi().upload_file",
    ]:
        assert token in source
    assert 'github_token = "' not in source
    assert "DATASET_TARGET_ROOT.mkdir(parents=True, exist_ok=True)" in source
    assert "resolve_dataset_source" in source
    assert "shutil.copytree" in source
