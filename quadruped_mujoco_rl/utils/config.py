from pathlib import Path
from typing import Any

import yaml

from quadruped_mujoco_rl.utils.paths import resolve_project_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    return config
