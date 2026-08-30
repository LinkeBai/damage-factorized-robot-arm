"""Immutable topology-residual splits for five-joint G1 simulation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from robotarm.envs.damage import D1, D2, D3, D4, D5, DamageConfig
from robotarm.envs.residual_physics import ResidualPhysicsConfig, residual_profile

SplitName = Literal["train", "validation", "test"]
_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_G1_SPLIT = _ROOT / "config" / "splits" / "g1_5dof_v1.yaml"


def damage_from_name(name: str) -> DamageConfig:
    table = {
        "intact": DamageConfig.intact,
        "D1": D1,
        "D2": D2,
        "D3": D3,
        "D4": D4,
        "D5": D5,
    }
    try:
        return table[name]()
    except KeyError:
        raise KeyError(f"unknown topology {name!r}; choose from {sorted(table)}") from None


@dataclass(frozen=True)
class DomainSpec:
    topology: str
    residual_name: str
    split: SplitName

    @property
    def domain_id(self) -> str:
        return f"{self.topology}__{self.residual_name}"

    @property
    def damage(self) -> DamageConfig:
        return damage_from_name(self.topology)

    @property
    def residual(self) -> ResidualPhysicsConfig:
        return residual_profile(self.residual_name)


@dataclass(frozen=True)
class G1Protocol:
    version: str
    dof: int
    calibration_shots: tuple[int, ...]
    train: tuple[DomainSpec, ...]
    validation: tuple[DomainSpec, ...]
    test: tuple[DomainSpec, ...]
    source_path: Path
    sha256: str

    @property
    def all_domains(self) -> tuple[DomainSpec, ...]:
        return self.train + self.validation + self.test


def _canonical_hash(data: dict[str, object]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_g1_protocol(path: str | Path = DEFAULT_G1_SPLIT) -> G1Protocol:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if int(data["dof"]) != 5:
        raise ValueError("G1 split must declare dof: 5")

    def specs(key: SplitName) -> tuple[DomainSpec, ...]:
        return tuple(
            DomainSpec(str(topology), str(residual), key)
            for topology, residual in data[key]
        )

    protocol = G1Protocol(
        version=str(data["version"]),
        dof=int(data["dof"]),
        calibration_shots=tuple(int(value) for value in data["calibration_shots"]),
        train=specs("train"),
        validation=specs("validation"),
        test=specs("test"),
        source_path=path.resolve(),
        sha256=_canonical_hash(data),
    )
    ids = [
        {domain.domain_id for domain in split}
        for split in (protocol.train, protocol.validation, protocol.test)
    ]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("train/validation/test domain combinations must be disjoint")
    return protocol


def build_g1_protocol() -> G1Protocol:
    """Backward-compatible name for loading the immutable G1 split."""
    return load_g1_protocol()
