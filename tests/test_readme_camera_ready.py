from pathlib import Path


def test_readme_documents_k_sensitivity_and_kaggle_upload():
    text = Path("README.md").read_text()
    for phrase in [
        "--k-values",
        "--selected-k",
        "Silhouette",
        "Davies",
        "Calinski",
        "interpretability_sensitivity.md",
        "MinMinMinMin/KL",
        "HF_TOKEN",
    ]:
        assert phrase in text


def test_readme_documents_seven_phase_notebooks_and_hf_state_resume():
    text = Path("README.md").read_text()
    for notebook in [
        "01_assistment2009_primary.ipynb",
        "02_assistment2009_finish.ipynb",
        "03_assistment2017_primary.ipynb",
        "04_assistment2017_finish.ipynb",
        "05_xes3g5m_primary.ipynb",
        "06_xes3g5m_finish.ipynb",
        "07_final_package_upload.ipynb",
    ]:
        assert notebook in text
    for phrase in [
        "run_phase.py",
        "state/",
        "final/",
        "upload_folder",
        "12405,12406,12407",
        "BATCH_SIZE = 128",
    ]:
        assert phrase in text
