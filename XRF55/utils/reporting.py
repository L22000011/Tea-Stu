from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_yaml(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        if yaml is None:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        else:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def append_csv_row(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: str | Path, rows: List[Dict[str, Any]], title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(f"# {title}\n\nNo rows were produced.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [f"# {title}", "", "|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(header, "")) for header in headers) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_dependencies(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
        )
        content = result.stdout if result.stdout else result.stderr
    except Exception as exc:
        content = f"Failed to collect dependencies: {exc}\n"
    path.write_text(content, encoding="utf-8")


def plot_history(path: str | Path, history: Iterable[Dict[str, Any]]) -> None:
    rows = list(history)
    path = Path(path)
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        path.with_suffix(".txt").write_text("matplotlib is not installed; plot was skipped.\n", encoding="utf-8")
        return

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_acc = [float(row["val_acc"]) for row in rows if row.get("val_acc") not in ("NA", None, "")]
    val_epochs = [int(row["epoch"]) for row in rows if row.get("val_acc") not in ("NA", None, "")]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    if val_acc:
        plt.plot(val_epochs, val_acc, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("XRF55 training history")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
