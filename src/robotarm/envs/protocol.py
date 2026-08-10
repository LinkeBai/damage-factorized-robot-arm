"""RobotEnv protocol — unified interface across simulation and real hardware.

Per PROJECT-PLAN-V4 §4.7, training code must not depend directly on the
MuJoCo or Feetech SDK APIs; it only talks to this protocol. This lets us run
the same training, calibration, and evaluation code on either the simulated
arm (``MujocoArmEnv``) or the real one (``FeetechArmEnv``).

The protocol is structural (typing.Protocol) rather than an ABC so that both
concrete environments can be lighter, but the contract itself is explicit.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Damage configuration
# ---------------------------------------------------------------------------
@runtime_checkable
class DamageConfig(Protocol):
    """What is broken in the robot for a given deployment.

    Fields (defined by the concrete dataclass in ``damage.py``):
        joint_mask: np.ndarray[int, (6,)]  -- 1 if the joint is locked, else 0
        lock_angle: np.ndarray[float, (6,)] -- absolute lock angle per joint
        (a joint that is not locked leaves this entry, but it is ignored)
    """

    joint_mask: npt.NDArray[np.int64]
    lock_angle: npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Observation / transition
# ---------------------------------------------------------------------------
@runtime_checkable
class Observation(Protocol):
    """One observation returned by the environment.

    Standardized to dict-of-arrays with at least these keys:
        state: np.ndarray[float, (S,)]   -- proprioception (joint positions/vel)
        target: np.ndarray[float, (D,)]  -- task target (e.g. 3D reach point)
        (optionally) image*: np.ndarray[uint8, (H, W, 3)] -- RGB, if obs_type uses pixels
    """

    state: npt.NDArray[np.float64]
    target: npt.NDArray[np.float64]


@runtime_checkable
class StepResult(Protocol):
    """Result of a single ``step``.

    Keys (dict-of-arrays, mutable):
        observation, reward, success, done
    """

    observation: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.uint8] | npt.NDArray[np.int64]]
    reward: float
    success: bool
    done: bool


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------
@runtime_checkable
class RobotEnv(Protocol):
    """Every arm environment (sim or real) must implement this."""

    def reset(self, *, target: npt.NDArray[np.float64], damage_config: DamageConfig | None = None) -> Observation:
        """Reset to a given target (and optionally an injected damage).

        Returns initial observation.
        """
        ...

    def step(self, action: npt.NDArray[np.float64]) -> StepResult:
        """Apply a joint command, advance one control step, return result."""
        ...

    def emergency_stop(self) -> None:
        """Immediately halt the arm in a safe state. Must always work."""
        ...

    def close(self) -> None:
        """Release resources (sim model handle, serial port, subprocess...)."""
        ...

    @property
    def action_dim(self) -> int:
        """Dimensionality of the action vector."""
        ...

    @property
    def observation_dim(self) -> int:
        """Dimensionality of the proprioception/state vector."""
        ...
