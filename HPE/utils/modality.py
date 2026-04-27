import itertools
import random
from typing import Dict, Iterable, List, Sequence


ALL_MODALITIES = ["vk", "depth", "lidar", "mmwave", "wifi-csi"]
NONVISUAL_MODALITIES = ["depth", "lidar", "mmwave", "wifi-csi"]

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
    result = []
    for name in modalities:
        canonical = canonicalize_modality(name)
        if canonical not in result:
            result.append(canonical)
    return result


def modality_names_to_list(names: Sequence[str], reference: Sequence[str] = ALL_MODALITIES) -> List[bool]:
    selected = set(canonicalize_modalities(names))
    return [name in selected for name in reference]


def modality_list_to_names(mask: Sequence[bool], reference: Sequence[str] = ALL_MODALITIES) -> List[str]:
    return [name for name, enabled in zip(reference, mask) if enabled]


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
    valid_drop_counts = [d for d in drop_counts if 0 <= d < len(available)]
    if not valid_drop_counts:
        valid_drop_counts = [0]
    drop_count = generator.choice(valid_drop_counts)
    dropped = set(generator.sample(available, drop_count))
    selected = [name for name in available if name not in dropped]
    if not selected:
        selected = [generator.choice(available)]
    return selected


def batch_modality_summary(modality_counts: Dict[str, int], selected: Sequence[str]) -> None:
    key = "+".join(selected)
    modality_counts[key] = modality_counts.get(key, 0) + 1

