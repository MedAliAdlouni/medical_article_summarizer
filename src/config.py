import os
import torch


def get_default_data_path() -> str:
    # Allow override via env var; else fallback to a local data path
    return os.environ.get(
        "MAS_DATA_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
    )


conf = {
    "dataset": {
        "path": get_default_data_path(),
        "subset": "train",
    },
    "architecture": "encoder_decoder",
    "model": "facebook/bart-large-cnn",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


