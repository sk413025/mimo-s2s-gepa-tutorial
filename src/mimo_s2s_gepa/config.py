from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else base / value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config["_config_path"] = str(config_path)
    if "data_path" in config:
        config["data_path"] = str(resolve_path(config["data_path"]))
    if "task_path" in config:
        config["task_path"] = str(resolve_path(config["task_path"]))
    config["output_dir"] = str(resolve_path(config.get("output_dir", "outputs")))
    return config
