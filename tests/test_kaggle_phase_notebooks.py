import ast
import json
from pathlib import Path


NOTEBOOK_ROOT = Path("notebooks")
NOTEBOOKS = [
    "01_assistment2009_primary.ipynb",
    "02_assistment2009_finish.ipynb",
    "03_assistment2017_primary.ipynb",
    "04_assistment2017_finish.ipynb",
    "05_xes3g5m_primary.ipynb",
    "06_xes3g5m_finish.ipynb",
    "07_final_package_upload.ipynb",
]


def notebook_source(name):
    notebook = json.loads((NOTEBOOK_ROOT / name).read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"{name}:cell_{index}")
    return source


def test_exactly_seven_phase_notebooks_exist_and_compile():
    assert sorted(path.name for path in NOTEBOOK_ROOT.glob("0*.ipynb")) == NOTEBOOKS
    for name in NOTEBOOKS:
        source = notebook_source(name)
        assert "REPO_OWNER" in source
        assert "GITHUB_SECRET_NAME = \"GITHUB_TOKEN\"" in source
        assert 'HF_REPO_ID = "MinMinMinMin/KL"' in source
        assert 'HF_REPO_TYPE = "model"' in source
        assert "/kaggle/input/datasets/minhvu111/dataset-kl" in source
        assert 'GPU = "0,1"' in source
        assert 'DEVICE = "cuda:0"' in source
        assert "DDP_GPUS = 2" in source
        assert 'MODEL_SEEDS = "12405,12406,12407"' in source
        assert "K_VALUES = \"3,4,5,6\"" in source
        assert "SELECTED_K = 3" in source


def test_primary_notebooks_run_one_job_and_upload_state():
    for name in [
        "01_assistment2009_primary.ipynb",
        "03_assistment2017_primary.ipynb",
        "05_xes3g5m_primary.ipynb",
    ]:
        source = notebook_source(name)
        assert '"--phase", "primary"' in source
        assert '"--run-slug", slug' in source
        assert "upload_hf_state" in source
        assert "validate_primary_outputs" in source


def test_finish_notebooks_run_both_model_seeds_then_analysis():
    for name in [
        "02_assistment2009_finish.ipynb",
        "04_assistment2017_finish.ipynb",
        "06_xes3g5m_finish.ipynb",
    ]:
        source = notebook_source(name)
        assert '"--phase", "model_seeds"' in source
        assert "for target_seed in [12406, 12407]" in source
        assert '"--phase", "analysis"' in source
        assert "validate_model_seed_outputs" in source
        assert "upload_hf_state" in source


def test_final_notebook_validates_all_datasets_and_uses_requested_upload_form():
    source = notebook_source("07_final_package_upload.ipynb")
    for dataset in ["ASSISTMENT2009", "ASSISTMENT2017", "XES3G5M"]:
        assert dataset in source
    assert "validate_final_dataset_outputs" in source
    assert "package_outputs" in source
    assert "HfApi().upload_file" in source
    assert 'HF_REPO_SUBDIR = "final"' in source
    assert 'upload_path = f"{HF_REPO_SUBDIR}/{zip_path.name}"' in source
