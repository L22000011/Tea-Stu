from __future__ import annotations

import itertools
import random
from typing import Iterable, List, Sequence


ALL_MODALITIES = ["wifi", "rfid", "mmwave"]

LEGACY_TO_CANONICAL = {
    "wifi": "wifi",
    "wi-fi": "wifi",
    "rfid": "rfid",
    "mmwave": "mmwave",
    "mm-wave": "mmwave",
    "radar": "mmwave",
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


def summarize_available_counts(modality_sets: Iterable[str], total_modalities: int = len(ALL_MODALITIES)) -> List[dict]:
    buckets: dict[int, list[str]] = {}
    for value in modality_sets:
        size = len([part for part in value.split("+") if part.strip()])
        buckets.setdefault(size, []).append(value)
    rows = []
    for size in sorted(buckets):
        rows.append(
            {
                "available_modalities": size,
                "missing_modalities": total_modalities - size,
                "combination_count": len(buckets[size]),
                "combinations": ";".join(buckets[size]),
            }
        )
    return rows
