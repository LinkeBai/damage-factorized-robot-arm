"""Continuous residual-physics configurations for simulation deployments.

Topology describes *which* joint is locked. This module describes the
continuous effects that remain unknown after that diagnosis: actuator loss,
damping/friction variation, command latency, deadband, sensor noise and tool
payload. These values are simulation factors, not real-arm measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

N_JOINTS = 5


@dataclass(frozen=True)
class ResidualPhysicsConfig:
    name: str = "nominal"
    actuator_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    damping_scale: float = 1.0
    friction_scale: float = 1.0
    armature_scale: float = 1.0
    backlash: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    control_delay_steps: int = 0
    action_deadband: float = 0.0
    observation_noise_std: float = 0.0
    payload_mass_delta_kg: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.actuator_scale) != N_JOINTS:
            raise ValueError(f"actuator_scale must have {N_JOINTS} values")
        if any(value <= 0 for value in self.actuator_scale):
            raise ValueError("actuator_scale values must be positive")
        if len(self.backlash) != N_JOINTS:
            raise ValueError(f"backlash must have {N_JOINTS} values")
        if any(value < 0 for value in self.backlash):
            raise ValueError("backlash values must be non-negative")
        for field_name in ("damping_scale", "friction_scale", "armature_scale"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.control_delay_steps < 0:
            raise ValueError("control_delay_steps must be non-negative")
        if not 0.0 <= self.action_deadband < 1.0:
            raise ValueError("action_deadband must be in [0, 1)")
        if self.observation_noise_std < 0 or self.payload_mass_delta_kg < 0:
            raise ValueError("noise and payload delta must be non-negative")

    @property
    def actuator_scale_array(self) -> np.ndarray:
        return np.asarray(self.actuator_scale, dtype=np.float64)

    @property
    def backlash_array(self) -> np.ndarray:
        return np.asarray(self.backlash, dtype=np.float64)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


RESIDUAL_PROFILES: dict[str, ResidualPhysicsConfig] = {
    "nominal": ResidualPhysicsConfig(),
    "weak_motor": ResidualPhysicsConfig(
        name="weak_motor",
        actuator_scale=(0.78, 0.82, 0.75, 0.80, 0.85),
        seed=11,
    ),
    "high_damping": ResidualPhysicsConfig(
        name="high_damping",
        damping_scale=2.0,
        friction_scale=1.5,
        seed=12,
    ),
    "delay_1": ResidualPhysicsConfig(
        name="delay_1",
        control_delay_steps=1,
        seed=13,
    ),
    "noisy_deadband": ResidualPhysicsConfig(
        name="noisy_deadband",
        action_deadband=0.04,
        observation_noise_std=0.002,
        seed=21,
    ),
    "mixed_composition": ResidualPhysicsConfig(
        name="mixed_composition",
        actuator_scale=(0.78, 0.82, 0.75, 0.80, 0.85),
        damping_scale=2.0,
        friction_scale=1.5,
        control_delay_steps=1,
        action_deadband=0.04,
        observation_noise_std=0.002,
        seed=25,
    ),
    "mixed_unseen": ResidualPhysicsConfig(
        name="mixed_unseen",
        actuator_scale=(0.68, 0.74, 0.70, 0.72, 0.78),
        damping_scale=1.7,
        friction_scale=2.0,
        armature_scale=1.3,
        backlash=(0.03, 0.03, 0.03, 0.03, 0.03),
        control_delay_steps=2,
        action_deadband=0.06,
        observation_noise_std=0.003,
        payload_mass_delta_kg=0.03,
        seed=31,
    ),
}


def residual_profile(name: str) -> ResidualPhysicsConfig:
    try:
        return RESIDUAL_PROFILES[name]
    except KeyError:
        raise KeyError(
            f"unknown residual profile {name!r}; choose from {sorted(RESIDUAL_PROFILES)}"
        ) from None
