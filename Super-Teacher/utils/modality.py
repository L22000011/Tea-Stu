from __future__ import annotations

import itertools
import random
from typing import Iterable, List, Sequence


ALL_MODALITIES = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
HPE_MODALITIES = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
HAR_MODALITIES = ["vk", "depth", "lidar", "mmwave"]
NONVISUAL_HPE_MODALITIES = ["depth", "lidar", "mmwave", "wifi-csi"]
NONVISUAL_HAR_MODALITIES = ["depth", "lidar", "mmwave"]

LEGACY_TO_CANONICAL = {
    "rgb": "vk",
    "vk": "vk",
    "visual_keypoint": "vk",
    "visual-keypoint": "vk",
    "depth": "depth",
    "lidar": "lidar",
    "mmwave": "mmwave",
    "radar": "mmwave",
    "wifi": "wifi-csi",
    "wifi-csi": "wifi-csi",
    "csi": "wifi-csi",
}


def canonicalize_modality(name: str) -> str:
    key = name.lower()
    if key not in LEGACY_TO_CANONICAL:
        raise ValueError(f"Unknown modality: {name}")
    return LEGACY_TO_CANONICAL[key]


def canonicalize_modalities(modalities: Iterable[str]) -> List[str]:
    result: List[str] = []
    for name in modalities:
        canonical = canonicalize_modality(name)
        if canonical not in result:
            result.append(canonical)
    return result


def all_nonempty_combinations(reference: Sequence[str] = ALL_MODALITIES) -> List[List[str]]:
    combos: List[List[str]] = []
    for size in range(1, len(reference) + 1):
        for combo in itertools.combinations(reference, size):
            combos.append(list(combo))
    return combos


def sample_missing_modalities(
    available: Sequence[str],
    drop_counts: Sequence[int],
    rng: random.Random | None = None,
) -> List[str]:
    generator = rng if rng is not None else random
    available = list(available)
    if not available:
        raise ValueError("At least one available modality is required.")
    valid = [count for count in drop_counts if 0 <= count < len(available)]
    if not valid:
        valid = [0]
    drop_count = generator.choice(valid)
    dropped = set(generator.sample(available, drop_count))
    selected = [name for name in available if name not in dropped]
    if not selected:
        selected = [generator.choice(available)]
    return selected
