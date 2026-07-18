from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.modality import ALL_MODALITIES, canonicalize_modalities


SANITIZE_COUNTS: Dict[str, int] = {}


def _record_sanitize_event(name: str, path: Path | None = None) -> None:
    SANITIZE_COUNTS[name] = SANITIZE_COUNTS.get(name, 0) + 1
    count = SANITIZE_COUNTS[name]
    if count <= 5 or count % 100 == 0:
        suffix = f": {path}" if path is not None else ""
        print(f"[XRF55Dataset] Sanitized {name} #{count}{suffix}")


def _finite_array(data: np.ndarray, name: str, path: Path) -> np.ndarray:
    if not np.isfinite(data).all():
        _record_sanitize_event(f"{name}_nan_inf", path)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data.astype(np.float32)


def _safe_sequence(data: np.ndarray, expected_channels: int, name: str, path: Path) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim == 1:
        data = data[None, :]
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    if data.shape[0] != expected_channels and data.ndim == 2 and data.shape[1] == expected_channels:
        data = data.T

    if data.shape[0] != expected_channels:
        flat = data.reshape(-1)
        steps = int(np.ceil(flat.size / expected_channels))
        padded = np.zeros(expected_channels * steps, dtype=np.float32)
        keep = min(flat.size, padded.size)
        padded[:keep] = flat[:keep]
        data = padded.reshape(expected_channels, steps)
        _record_sanitize_event(f"{name}_reshaped", path)

    return _finite_array(data, name, path)


def _safe_mmwave(data: np.ndarray, path: Path) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim == 2:
        data = np.repeat(data[None, :, :], 17, axis=0)
        _record_sanitize_event("mmwave_2d_repeated", path)
    elif data.ndim == 3:
        if data.shape[0] == 17:
            pass
        elif data.shape[-1] == 17:
            data = np.transpose(data, (2, 0, 1))
        else:
            channels = min(17, data.shape[0])
            fixed = np.zeros((17, data.shape[-2], data.shape[-1]), dtype=np.float32)
            fixed[:channels] = data[:channels]
            data = fixed
            _record_sanitize_event("mmwave_channels_fixed", path)
    elif data.ndim == 4:
        if data.shape[0] == 1 and data.shape[1] == 17:
            data = data[0]
        elif data.shape[-1] == 17:
            data = np.transpose(data[0], (2, 0, 1))
        else:
            data = data.reshape(17, data.shape[-2], data.shape[-1])
            _record_sanitize_event("mmwave_4d_reshaped", path)
    else:
        flat = data.reshape(-1)
        side = int(np.ceil(np.sqrt(max(flat.size / 17, 1))))
        padded = np.zeros(17 * side * side, dtype=np.float32)
        padded[: min(flat.size, padded.size)] = flat[: min(flat.size, padded.size)]
        data = padded.reshape(17, side, side)
        _record_sanitize_event("mmwave_fallback_cube", path)
    return _finite_array(data, "mmwave", path)


class XRF55Database:
    def __init__(self, data_root: str | Path, scene: str = "dml", semantic_vector_path: str | Path | None = None) -> None:
        root = Path(data_root)
        if not root.exists():
            raise FileNotFoundError(f"XRF55 dataset root does not exist: {root}")

        candidate_root = root / "XRF_dataset"
        self.data_root = candidate_root if candidate_root.exists() else root
        self.scene = scene
        self.semantic_vector_path = Path(semantic_vector_path) if semantic_vector_path is not None else None
        self.word_vectors = self._load_word_vectors()
        print(f"[XRF55Database] Dataset root: {self.data_root}")
        if self.semantic_vector_path is not None:
            print(f"[XRF55Database] Semantic vectors: {self.semantic_vector_path}")

    def _load_word_vectors(self) -> np.ndarray | None:
        if self.semantic_vector_path is None:
            return None
        if not self.semantic_vector_path.exists():
            print(f"[XRF55Database] Semantic vector file not found, semantic loss disabled: {self.semantic_vector_path}")
            return None
        vectors = np.load(self.semantic_vector_path).astype(np.float32)
        print(f"[XRF55Database] Loaded semantic vectors: shape={vectors.shape}")
        return vectors


class XRF55Dataset(Dataset):
    def __init__(
        self,
        database: XRF55Database,
        split: str,
        modalities: Iterable[str],
    ) -> None:
        self.database = database
        self.split = split
        self.modalities = canonicalize_modalities(modalities)
        self.samples = self._build_index()

    def _split_file(self) -> Path:
        suffix = "train" if self.split == "train" else "val"
        path = self.database.data_root / f"{self.database.scene}_{suffix}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        return path

    def _base_data_dir(self) -> Path:
        subset = "train_data" if self.split == "train" else "test_data"
        path = self.database.data_root / f"{self.database.scene}_new_data" / subset
        if not path.exists():
            raise FileNotFoundError(f"Data directory not found: {path}")
        return path

    def _build_index(self) -> List[Dict[str, Any]]:
        txt_path = self._split_file()
        base_dir = self._base_data_dir()
        lines = txt_path.read_text(encoding="utf-8").splitlines()
        samples: List[Dict[str, Any]] = []
        print(f"[XRF55Dataset] Building {self.split} index | samples={len(lines)} | modalities={self.modalities}")
        for idx, line in enumerate(lines, start=1):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                raise ValueError(f"Invalid split line in {txt_path}: {line}")
            file_name = parts[0]
            label = int(parts[2]) - 1
            item = {
                "sample_id": file_name,
                "label": label,
                "modalities": self.modalities,
            }
            for modality in self.modalities:
                folder = "WiFi" if modality == "wifi" else "RFID" if modality == "rfid" else "mmWave"
                item[f"{modality}_path"] = base_dir / folder / f"{file_name}.npy"
            samples.append(item)
            if idx == 1 or idx % 200 == 0 or idx == len(lines):
                print(f"[XRF55Dataset] Indexed sample {idx}/{len(lines)}: {file_name}")
        return samples

    def _load_semantic_target(self, label: int) -> np.ndarray:
        vectors = self.database.word_vectors
        if vectors is None:
            return np.zeros((1024,), dtype=np.float32)
        if label < 0 or label >= len(vectors):
            raise IndexError(f"Label {label} is out of range for semantic vectors with shape {vectors.shape}.")
        return vectors[label].astype(np.float32)

    def _read_modality(self, path: Path, modality: str) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Missing modality file: {path}")
        data = np.load(path, allow_pickle=False)
        if modality == "wifi":
            return _safe_sequence(data, expected_channels=270, name="wifi", path=path)
        if modality == "rfid":
            return _safe_sequence(data, expected_channels=23, name="rfid", path=path)
        if modality == "mmwave":
            return _safe_mmwave(data, path=path)
        raise ValueError(f"Unsupported modality: {modality}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.samples[index]
        inputs = {}
        for modality in self.modalities:
            inputs[modality] = torch.from_numpy(self._read_modality(item[f"{modality}_path"], modality))
        target = int(item["label"])
        return {
            "inputs": inputs,
            "target": torch.tensor(target, dtype=torch.long),
            "semantic_target": torch.from_numpy(self._load_semantic_target(target)),
            "meta": {
                "sample_id": item["sample_id"],
                "split": self.split,
            },
        }


def build_datasets(dataset_root: str | Path, config: Dict[str, Any]) -> tuple[XRF55Dataset, XRF55Dataset]:
    scene = config.get("scene", "dml")
    semantic_path = config.get("semantic_vector_path")
    if semantic_path is None:
        semantic_root = Path(__file__).resolve().parents[2] / "origin-XRF55" / "word2vec" / "bert_new_sentence_large_uncased.npy"
        semantic_path = str(semantic_root)
    database = XRF55Database(dataset_root, scene=scene, semantic_vector_path=semantic_path)
    modalities = config.get("modality", ALL_MODALITIES)
    train_dataset = XRF55Dataset(database, split="train", modalities=modalities)
    val_dataset = XRF55Dataset(database, split="val", modalities=modalities)
    return train_dataset, val_dataset


def build_dataloaders(dataset_root: str | Path, config: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    train_dataset, val_dataset = build_datasets(dataset_root, config)
    loader_cfg = config.get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 32))
    num_workers = int(loader_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(
        f"[Data] Dataloaders ready | train_samples={len(train_dataset)} | "
        f"val_samples={len(val_dataset)} | batch_size={batch_size} | workers={num_workers}"
    )
    return train_loader, val_loader
