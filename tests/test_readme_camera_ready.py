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
