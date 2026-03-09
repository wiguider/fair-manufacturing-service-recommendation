from pathlib import Path
import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent
