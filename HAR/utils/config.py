from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to load HAR config files.") from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
