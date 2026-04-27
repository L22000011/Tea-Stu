from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_cli_config(config: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    merged = dict(config)
    for key, value in kwargs.items():
        if value is not None:
            merged[key] = value
    return merged

