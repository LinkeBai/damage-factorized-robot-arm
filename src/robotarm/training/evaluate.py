"""Evaluation harness for a policy on the sim arm (PROJECT-PLAN-V4 §5, §6.1).

Runs a policy against a set of targets, resetting the ``MujocoArmEnv`` per
episode (optionally under a damage config), and reports the plan's primary
metrics for Reach (§5.1): success rate, final distance, and time-to-reach.
The policy is any callable ``(state) -> action`` normalized to [-1,1]; this
keeps the harness policy-agnostic so it can score a scripted baseline, a
random controller, or (later) the DFWM world-model policy identically.

The success threshold is the plan's engineering starting value (5 cm); it is
recalibrated at G0 from camera / kinematics error — see config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

from robotarm.envs.mujoco_env import MujocoArmEnv

Policy = Callable[[npt.NDArray], npt.NDArray]  # state(12,) -> action(6,)


@dataclass
class ReachMetrics:
    """Aggregate Reach metrics over an evaluation run."""

    success_rate: float
    mean_final_distance: float
    mean_time_to_reach: float  # steps

    def as_dict(self) -> dict[str, float]:
        return {
            "success_rate": self.success_rate,
            "mean_final_distance": self.mean_final_distance,
            "mean_time_to_reach": self.mean_time_to_reach,
        }


def _rand_policy(rng: np.random.Generator) -> Policy:
    def policy(state: npt.NDArray) -> npt.NDArray:
        return rng.uniform(-1.0, 1.0, size=5)
    return policy


def _zero_policy(state: npt.NDArray) -> npt.NDArray:
    return np.zeros(5)


def evaluate_reach(
    env: MujocoArmEnv,
    targets: npt.NDArray,
    policy: Policy,
    *,
    max_steps: int = 200,
    tolerance: float = 0.05,
    damage_config=None,
    rng: np.random.Generator | None = None,
) -> ReachMetrics:
    """Evaluate ``policy`` on each ``target``; return aggregate metrics."""
    rng = rng or np.random.default_rng(0)
    successes = 0
    final_dists = []
    times = []

    for t in targets:
        obs = env.reset(target=t, damage_config=damage_config)
        success = False
        reached_at = max_steps
        for step in range(max_steps):
            action = np.asarray(policy(obs["state"]), dtype=np.float64)
            res = env.step(action)
            obs = res["observation"]
            dist = float(np.linalg.norm(env.ee_pos() - t))
            if res["success"] or dist <= tolerance:
                success = True
                reached_at = step + 1
                break
        successes += int(success)
        final_dists.append(dist if not success else 0.0)
        # time-to-reach = final step when success, else max (not reached).
        times.append(reached_at if success else max_steps)

    return ReachMetrics(
        success_rate=successes / len(targets),
        mean_final_distance=float(np.mean(final_dists)),
        mean_time_to_reach=float(np.mean(times)),
    )


__all__ = ["Policy", "ReachMetrics", "evaluate_reach", "_rand_policy", "_zero_policy"]
