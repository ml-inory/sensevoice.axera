import os

# Speed up hf download using mirror url
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

current_file_path = os.path.dirname(__file__)
REPO_ROOT = "AXERA-TECH"
CACHE_PATH = os.path.join(current_file_path, "models")


def download_model(model_name: str) -> str:
    """
    Download model from AXERA-TECH's huggingface space.

    model_name: str
        Available model names could be checked on https://huggingface.co/AXERA-TECH.

    Returns:
        str: Path to model_name

    """
    os.makedirs(CACHE_PATH, exist_ok=True)

    model_path = os.path.join(CACHE_PATH, model_name)
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        snapshot_download(
            repo_id=f"{REPO_ROOT}/{model_name}",
            local_dir=os.path.join(CACHE_PATH, model_name),
        )

    return model_path
