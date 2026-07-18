from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.modality import ALL_MODALITIES, canonicalize_modalities

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import scipy.io as scio
except ModuleNotFoundError:
    scio = None


ALL_SUBJECTS = [
    "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10",
    "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20",
    "S21", "S22", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30",
    "S31", "S32", "S33", "S34", "S35", "S36", "S37", "S38", "S39", "S40",
]

ALL_ACTIONS = [
    "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09",
    "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18",
    "A19", "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27",
]

PROTOCOL_ACTIONS = {
    "protocol1": ["A02", "A03", "A04", "A05", "A13", "A14", "A17", "A18", "A19", "A20", "A21", "A22", "A23", "A27"],
    "protocol2": ["A01", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A15", "A16", "A24", "A25", "A26"],
    "protocol3": ALL_ACTIONS,
}

SANITIZE_COUNTS: Dict[str, int] = {}


def _load_manifest_keys(manifest_path: str | Path | None) -> set[tuple[str, str, str, int]] | None:
    if not manifest_path:
        return None
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {path}")
    keys: set[tuple[str, str, str, int]] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            keys.add((row["scene"], row["subject"], row["action"], int(row["idx"])))
    print(f"[MMFiDataset] Loaded manifest filter | path={path} | samples={len(keys)}")
    return keys


def _record_sanitize_event(name: str, path: Path | None = None) -> None:
    SANITIZE_COUNTS[name] = SANITIZE_COUNTS.get(name, 0) + 1
    count = SANITIZE_COUNTS[name]
    if count <= 5 or count % 100 == 0:
        suffix = f": {path}" if path is not None else ""
        print(f"[MMFiDataset] Sanitized {name} #{count}{suffix}")


def _finite_array(data: np.ndarray, name: str, path: Path) -> np.ndarray:
    if not np.isfinite(data).all():
        _record_sanitize_event(f"{name}_nan_inf", path)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def _safe_point_cloud(data: np.ndarray, dims: int, name: str, path: Path) -> np.ndarray:
    if data.size == 0:
        _record_sanitize_event(f"{name}_empty", path)
        return np.zeros((1, dims), dtype=np.float32)
    if data.size % dims != 0:
        valid_size = (data.size // dims) * dims
        _record_sanitize_event(f"{name}_truncated", path)
        data = data[:valid_size]
    if data.size == 0:
        _record_sanitize_event(f"{name}_empty_after_truncate", path)
        return np.zeros((1, dims), dtype=np.float32)
    data = data.reshape(-1, dims).astype(np.float32)
    data = _finite_array(data, name, path)
    if data.shape[0] == 0:
        _record_sanitize_event(f"{name}_zero_points", path)
        data = np.zeros((1, dims), dtype=np.float32)
    return data


def _safe_minmax(data: np.ndarray, name: str, path: Path) -> np.ndarray:
    data = _finite_array(data.astype(np.float32), name, path)
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    denom = max_val - min_val
    if denom < 1e-6:
        _record_sanitize_event(f"{name}_constant", path)
        return np.zeros_like(data, dtype=np.float32)
    return ((data - min_val) / denom).astype(np.float32)


def subject_to_scene(subject: str) -> str:
    idx = int(subject[1:])
    if 1 <= idx <= 10:
        return "E01"
    if 11 <= idx <= 20:
        return "E02"
    if 21 <= idx <= 30:
        return "E03"
    if 31 <= idx <= 40:
        return "E04"
    raise ValueError(f"Unknown MM-Fi subject: {subject}")


def decode_split(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    protocol = config.get("protocol", "protocol3")
    actions = PROTOCOL_ACTIONS.get(protocol, ALL_ACTIONS)
    split_name = config.get("split_to_use", "random_split")
    modalities = canonicalize_modalities(config.get("modality", ALL_MODALITIES))

    train_form: Dict[str, List[str]] = {}
    val_form: Dict[str, List[str]] = {}

    if split_name == "random_split":
        split_cfg = config["random_split"]
        ratio = float(split_cfg.get("ratio", 0.8))
        seed = int(split_cfg.get("random_seed", 0))
        for offset, action in enumerate(actions):
            rng = np.random.default_rng(seed + offset)
            indices = rng.permutation(len(ALL_SUBJECTS))
            cut = int(np.floor(ratio * len(ALL_SUBJECTS)))
            train_subjects = np.array(ALL_SUBJECTS)[indices[:cut]].tolist()
            val_subjects = np.array(ALL_SUBJECTS)[indices[cut:]].tolist()
            for subject in train_subjects:
                train_form.setdefault(subject, []).append(action)
            for subject in val_subjects:
                val_form.setdefault(subject, []).append(action)
    elif split_name == "cross_scene_split":
        train_scenes = set(config["cross_scene_split"]["train_dataset"].get("scenes") or ["E01", "E02", "E03"])
        val_scenes = set(config["cross_scene_split"]["val_dataset"].get("scenes") or ["E04"])
        for subject in ALL_SUBJECTS:
            scene = subject_to_scene(subject)
            if scene in train_scenes:
                train_form[subject] = list(actions)
            if scene in val_scenes:
                val_form[subject] = list(actions)
    elif split_name == "cross_subject_split":
        split_cfg = config["cross_subject_split"]
        for subject in split_cfg["train_dataset"]["subjects"]:
            train_form[subject] = list(actions)
        for subject in split_cfg["val_dataset"]["subjects"]:
            val_form[subject] = list(actions)
    else:
        split_cfg = config["manual_split"]
        for subject in split_cfg["train_dataset"]["subjects"]:
            train_form[subject] = list(split_cfg["train_dataset"]["actions"])
        for subject in split_cfg["val_dataset"]["subjects"]:
            val_form[subject] = list(split_cfg["val_dataset"]["actions"])

    return {
        "train": {"modalities": modalities, "split": "training", "data_form": train_form},
        "val": {"modalities": modalities, "split": "validation", "data_form": val_form},
    }


class MMFiDatabase:
    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"MM-Fi dataset root does not exist: {self.data_root}")
        print(f"[MMFiDatabase] Dataset root: {self.data_root}")


class MMFiDataset(Dataset):
    def __init__(
        self,
        database: MMFiDatabase,
        modalities: Iterable[str],
        split: str,
        data_form: Dict[str, List[str]],
        manifest_keys: set[tuple[str, str, str, int]] | None = None,
        vk_override_root: str | Path | None = None,
    ) -> None:
        self.database = database
        self.modalities = canonicalize_modalities(modalities)
        self.split = split
        self.data_form = data_form
        self.manifest_keys = manifest_keys
        self.vk_override_root = Path(vk_override_root) if vk_override_root else None
        self.data_list = self._load_data_list()

    def _modality_folder(self, modality: str) -> str:
        if modality == "vk":
            return "rgb"
        if modality == "wifi-csi":
            return "wifi-csi"
        return modality

    def _load_data_list(self) -> List[Dict[str, Any]]:
        data_info: List[Dict[str, Any]] = []
        root = self.database.data_root
        total_actions = sum(len(actions) for actions in self.data_form.values())
        action_counter = 0
        print(
            f"[MMFiDataset] Building {self.split} index | "
            f"subjects={len(self.data_form)} | actions={total_actions} | modalities={self.modalities}"
        )
        for subject, actions in self.data_form.items():
            scene = subject_to_scene(subject)
            for action in actions:
                action_counter += 1
                action_root = root / scene / subject / action
                mmwave_dir = action_root / "mmwave"
                if not mmwave_dir.exists():
                    print(f"[MMFiDataset] Skip missing mmwave directory: {mmwave_dir}")
                    continue
                modality_files: Dict[str, List[str]] = {}
                for modality in self.modalities:
                    folder = self._modality_folder(modality)
                    modality_dir = action_root / folder
                    if not modality_dir.exists():
                        raise FileNotFoundError(f"Missing modality directory: {modality_dir}")
                    modality_files[modality] = sorted(os.listdir(modality_dir))
                frame_list = modality_files["mmwave"]
                if action_counter == 1 or action_counter % 25 == 0 or action_counter == total_actions:
                    print(
                        f"[MMFiDataset] Indexed action {action_counter}/{total_actions}: "
                        f"{scene}/{subject}/{action}, frames={len(frame_list)}"
                    )
                for idx, mmwave_file in enumerate(frame_list):
                    frame_idx = int(mmwave_file.split(".")[0].split("frame")[1]) - 1
                    item: Dict[str, Any] = {
                        "modalities": self.modalities,
                        "scene": scene,
                        "subject": subject,
                        "action": action,
                        "gt_path": action_root / "ground_truth.npy",
                        "idx": frame_idx,
                    }
                    for modality in self.modalities:
                        folder = self._modality_folder(modality)
                        modality_dir = action_root / folder
                        files = modality_files[modality]
                        file_index = idx if modality == "mmwave" else frame_idx
                        if modality == "vk" and self.vk_override_root is not None:
                            item[f"{modality}_path"] = (
                                self.vk_override_root / scene / subject / action / f"frame{frame_idx + 1:03d}.npy"
                            )
                        else:
                            item[f"{modality}_path"] = modality_dir / files[file_index]
                    if self.manifest_keys is not None:
                        key = (scene, subject, action, frame_idx)
                        if key not in self.manifest_keys:
                            continue
                    data_info.append(item)
        print(f"[MMFiDataset] Finished {self.split} index | samples={len(data_info)}")
        return data_info

    def _read_frame(self, path: Path, modality: str) -> np.ndarray:
        if modality == "vk":
            data = np.load(path).astype(np.float32)
            data = _finite_array(data, "vk", path)
            if data.shape != (17, 2):
                _record_sanitize_event("vk_bad_shape", path)
                fixed = np.zeros((17, 2), dtype=np.float32)
                flat = data.reshape(-1, 2) if data.size >= 2 else np.zeros((0, 2), dtype=np.float32)
                keep = min(17, flat.shape[0])
                if keep > 0:
                    fixed[:keep] = flat[:keep]
                data = fixed
            return data
        if modality == "depth":
            if cv2 is None:
                raise ModuleNotFoundError("OpenCV is required to read MM-Fi depth images.")
            data = cv2.imread(str(path))
            if data is None:
                raise FileNotFoundError(f"Failed to load depth image: {path}")
            data = data.astype(np.float32)
            data = _finite_array(data, "depth", path)
            if data.ndim != 3:
                _record_sanitize_event("depth_bad_shape", path)
                if data.ndim == 2:
                    data = np.repeat(data[:, :, None], 3, axis=2)
                else:
                    raise ValueError(f"Unsupported depth image shape {data.shape}: {path}")
            if data.shape[2] == 1:
                data = np.repeat(data, 3, axis=2)
            if data.shape[2] > 3:
                data = data[:, :, :3]
            if float(np.max(data) - np.min(data)) < 1e-6:
                _record_sanitize_event("depth_constant", path)
            return data
        if modality == "lidar":
            with open(path, "rb") as handle:
                raw = handle.read()
            data = np.frombuffer(raw, dtype=np.float64).copy()
            return _safe_point_cloud(data, 3, "lidar", path)
        if modality == "mmwave":
            with open(path, "rb") as handle:
                raw = handle.read()
            data = np.frombuffer(raw, dtype=np.float64).copy()
            return _safe_point_cloud(data, 5, "mmwave", path)
        if modality == "wifi-csi":
            if scio is None:
                raise ModuleNotFoundError("SciPy is required to read MM-Fi WiFi-CSI .mat files.")
            data = scio.loadmat(path)["CSIamp"].astype(np.float32)
            data[np.isinf(data)] = np.nan
            for idx in range(data.shape[-1]):
                plane = data[:, :, idx]
                valid = plane[np.isfinite(plane)]
                fill_value = float(valid.mean()) if valid.size else 0.0
                plane[~np.isfinite(plane)] = fill_value
            return _safe_minmax(data, "wifi_csi", path)
        raise ValueError(f"Unsupported modality: {modality}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data_list[index]
        gt = np.load(item["gt_path"]).astype(np.float32)
        sample = {
            "meta": {
                "scene": item["scene"],
                "subject": item["subject"],
                "action": item["action"],
                "idx": item["idx"],
            },
            "target": torch.from_numpy(np.nan_to_num(gt[item["idx"]], nan=0.0, posinf=0.0, neginf=0.0)),
            "inputs": {},
        }
        for modality in self.modalities:
            sample["inputs"][modality] = self._read_frame(item[f"{modality}_path"], modality)
        return sample


def collate_mmfi_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    targets = torch.stack([item["target"] for item in batch]).float()
    inputs: Dict[str, torch.Tensor] = {}

    if "vk" in batch[0]["inputs"]:
        inputs["vk"] = torch.tensor(np.stack([item["inputs"]["vk"] for item in batch]), dtype=torch.float32)
    if "depth" in batch[0]["inputs"]:
        depth = np.stack([item["inputs"]["depth"] for item in batch])
        inputs["depth"] = torch.tensor(depth, dtype=torch.float32).permute(0, 3, 1, 2)
    if "mmwave" in batch[0]["inputs"]:
        mmwave = []
        for item in batch:
            tensor = torch.tensor(item["inputs"]["mmwave"], dtype=torch.float32)
            if tensor.numel() == 0 or tensor.size(0) == 0:
                _record_sanitize_event("mmwave_empty_collate")
                tensor = torch.zeros(1, 5, dtype=torch.float32)
            mmwave.append(tensor)
        inputs["mmwave"] = torch.nn.utils.rnn.pad_sequence(mmwave, batch_first=True)
    if "lidar" in batch[0]["inputs"]:
        lidar = []
        for item in batch:
            tensor = torch.tensor(item["inputs"]["lidar"], dtype=torch.float32)
            if tensor.numel() == 0 or tensor.size(0) == 0:
                _record_sanitize_event("lidar_empty_collate")
                tensor = torch.zeros(1, 3, dtype=torch.float32)
            lidar.append(tensor)
        inputs["lidar"] = torch.nn.utils.rnn.pad_sequence(lidar, batch_first=True)
    if "wifi-csi" in batch[0]["inputs"]:
        wifi = np.stack([item["inputs"]["wifi-csi"] for item in batch])
        inputs["wifi-csi"] = torch.tensor(wifi, dtype=torch.float32)

    return {
        "inputs": inputs,
        "target": targets,
        "meta": [item["meta"] for item in batch],
    }


def build_datasets(dataset_root: str | Path, config: Dict[str, Any]) -> Tuple[MMFiDataset, MMFiDataset]:
    database = MMFiDatabase(dataset_root)
    split = decode_split(config)
    manifest_keys = _load_manifest_keys(config.get("manifest_path"))
    vk_override_root = config.get("vk_override_root")
    if vk_override_root:
        print(f"[MMFiDataset] VK override root: {vk_override_root}")
    train_dataset = MMFiDataset(database, manifest_keys=manifest_keys, vk_override_root=vk_override_root, **split["train"])
    val_dataset = MMFiDataset(database, manifest_keys=manifest_keys, vk_override_root=vk_override_root, **split["val"])
    return train_dataset, val_dataset


def build_dataloaders(
    dataset_root: str | Path,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    print("[Data] Building train/validation datasets...")
    train_dataset, val_dataset = build_datasets(dataset_root, config)
    loader_cfg = config.get("loader", {})
    batch_size = int(loader_cfg.get("batch_size", 16))
    num_workers = int(loader_cfg.get("num_workers", 0))
    seed = int(config.get("init_rand_seed", 0))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_mmfi_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_mmfi_batch,
    )
    print(
        f"[Data] Dataloaders ready | train_samples={len(train_dataset)} | "
        f"val_samples={len(val_dataset)} | batch_size={batch_size} | workers={num_workers}"
    )
    return train_loader, val_loader
