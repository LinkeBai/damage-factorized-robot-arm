"""Fast original-arm closed-loop screen for the analytic SFET carrier.

This isolates the hard-subspace minimum-change transport using the live
end-effector task Jacobian.  It is a necessary carrier ablation, not evidence
that the frozen IPWM Jacobian or three-trial Broyden adaptation already works.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from robotarm.integrations.tdmpc2_env import OriginalArmPushEnv
from robotarm.models.structured_fault_effect_transport import SFETConfig, StructuredFaultEffectTransport
from robotarm.training.controllers import joint_reference_action
from run_original_arm_bc_sanity import damage_from_name


def task_jacobian(env: OriginalArmPushEnv) -> np.ndarray:
    arm = env._env
    jacp = np.zeros((3, arm.model.nv), dtype=np.float64)
    jacr = np.zeros((3, arm.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(arm.model, arm.data, jacp, jacr, arm._ee_site)
    return jacp[:2, arm._qvel_adr]


def transported_action(env: OriginalArmPushEnv, nominal: np.ndarray, ridge: float) -> np.ndarray:
    locked = tuple(env.damage.locked)
    full = task_jacobian(env)
    desired = full @ nominal
    carrier = StructuredFaultEffectTransport(
        full, locked=locked, config=SFETConfig(ridge=ridge, action_limit=1.0)
    )
    masked = nominal.copy()
    masked[list(locked)] = 0.0
    current = carrier.jacobian @ masked
    return carrier.repair(masked, desired, current).astype(np.float32)


def intact_reference_action(env: OriginalArmPushEnv) -> np.ndarray:
    """Return the full intact-reference command before applying the known lock."""
    if env._unaware_seed_references is None:
        raise RuntimeError("unaware references are unavailable")
    phase = 0 if env._step < 0.4 * env.max_episode_steps else 1
    state = env._env._observe()["state"][:10]
    return joint_reference_action(
        state, env._unaware_seed_references[phase], locked_joints=()
    ).astype(np.float32)


def evaluate(method: str, damage: str, seed: int, episodes: int, ridge: float) -> dict:
    env = OriginalArmPushEnv(
        seed=seed, seed_policy="directional_unaware", damage=damage_from_name(damage)
    )
    successes, distances, contacts, violations = [], [], [], []
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            had_contact = False
            info = None
            for _ in range(env.max_episode_steps):
                nominal = intact_reference_action(env)
                if method == "hard_mask":
                    action = nominal.copy()
                    action[list(env.damage.locked)] = 0.0
                elif method == "analytic_transport":
                    action = transported_action(env, nominal, ridge)
                elif method == "oracle_ik":
                    action = env.directional_action(fault_aware=True)
                else:
                    raise ValueError(method)
                violations.append(float(np.max(np.abs(action[list(env.damage.locked)]))))
                _, _, done, info = env.step(action)
                had_contact |= bool(info["contact"])
                if done:
                    break
            assert info is not None
            successes.append(bool(info["success"]))
            distances.append(float(info["goal_distance_m"]))
            contacts.append(had_contact)
    finally:
        env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "terminal_goal_distance_m_mean": float(np.mean(distances)),
        "contact_episode_rate": float(np.mean(contacts)),
        "max_locked_action_abs": float(np.max(violations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for damage in ("D2", "D3"):
        for seed in (7, 17, 27):
            for method in ("hard_mask", "analytic_transport", "oracle_ik"):
                result = evaluate(method, damage, seed + 10_000, args.episodes, args.ridge)
                row = {"damage": damage, "seed": seed, "method": method, **result}
                rows.append(row)
                print(row, flush=True)
    payload = {
        "diagnostic": "sfet_original_arm_analytic_carrier_screen",
        "development_only": True,
        "not_yet_tested": ["three-trial Broyden", "frozen IPWM response", "ridge/BC", "HCAR", "action regret"],
        "episodes_per_cell": args.episodes,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
