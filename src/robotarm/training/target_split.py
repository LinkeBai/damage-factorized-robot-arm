"""Immutable provisional Reach target split for five-joint experiments."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET_SPLIT = (
    _ROOT / "config" / "splits" / "reach_targets_5dof_provisional_v1.yaml"
)


@dataclass(frozen=True)
class ReachTarget:
    target_id: str
    xyz: tuple[float, float, float]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.xyz, dtype=np.float64)


@dataclass(frozen=True)
class ReachTargetSplit:
    version: str
    status: str
    calibration: tuple[ReachTarget, ...]
    validation: tuple[ReachTarget, ...]
    evaluation: tuple[ReachTarget, ...]
    source_path: Path
    sha256: str


def load_target_split(
    path: str | Path = DEFAULT_TARGET_SPLIT,
) -> ReachTargetSplit:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    def targets(key: str) -> tuple[ReachTarget, ...]:
        return tuple(
            ReachTarget(
                target_id=str(item["id"]),
                xyz=tuple(float(value) for value in item["xyz"]),
            )
            for item in data[key]
        )

    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    split = ReachTargetSplit(
        version=str(data["version"]),
        status=str(data["status"]),
        calibration=targets("calibration"),
        validation=targets("validation"),
        evaluation=targets("evaluation"),
        source_path=path.resolve(),
        sha256=hashlib.sha256(canonical).hexdigest(),
    )
    id_sets = [
        {target.target_id for target in group}
        for group in (split.calibration, split.validation, split.evaluation)
    ]
    if id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2]:
        raise ValueError("target IDs must be disjoint across splits")
    coordinates = [
        {target.xyz for target in group}
        for group in (split.calibration, split.validation, split.evaluation)
    ]
    if (
        coordinates[0] & coordinates[1]
        or coordinates[0] & coordinates[2]
        or coordinates[1] & coordinates[2]
    ):
        raise ValueError("target coordinates must be disjoint across splits")
    return split
