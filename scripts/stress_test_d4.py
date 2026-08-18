"""Quick stress test: IK+PD and WM-hybrid on harder D4 + mixed_unseen domains."""
from __future__ import annotations

import numpy as np
import torch

from robotarm.envs.damage import D4
from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.envs.residual_physics import residual_profile
from robotarm.training.controllers import joint_reference_action, solve_reach_reference
from robotarm.training.control_eval import infer_dfwm_context
from robotarm.training.g1_mechanism import train_mechanism_models
from robotarm.training.sim_data import collect_controller_domains
from robotarm.training.sim_protocol import DomainSpec, build_g1_protocol
from robotarm.training.target_split import load_target_split


def test_controller(name, env, target, locked, locked_tuple, reference, action_fn):
    obs = env.reset(target=target, damage_config=env._damage)
    for step in range(300):
        action = action_fn(obs, reference, locked_tuple)
        result = env.step(action)
        obs = result["observation"]
        distance = float(np.linalg.norm(env.ee_pos() - target))
        if distance <= 0.05:
            return True, step + 1, distance
    return False, 300, distance


def main():
    seed = 7
    torch.manual_seed(seed); np.random.seed(seed)
    split = load_target_split()
    cal = tuple(t.as_array() for t in split.calibration)
    targets = tuple(t.as_array() for t in split.evaluation)
    ranges = MujocoArmEnv().joint_ranges
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build stress domains: D4 + mixed_unseen
    protocol = build_g1_protocol()
    stress_domains = [
        DomainSpec("D4", "mixed_unseen", "test"),
        DomainSpec("D4", "mixed_composition", "test"),
        DomainSpec("D2", "mixed_unseen", "test"),
        DomainSpec("D3", "mixed_unseen", "test"),
    ]

    # Train models on standard protocol
    train = collect_controller_domains(
        protocol.train, trajectories_per_domain=2, steps=100,
        seed=seed * 10_000, targets=cal
    )
    models = train_mechanism_models(
        protocol.train, train, ranges, epochs=60, device=device
    )

    for domain in stress_domains:
        damage = domain.damage
        residual = domain.residual
        locked = {i: damage.lock_angle_of(i) for i in damage.locked}
        locked_tuple = tuple(damage.locked)

        # IK+PD baseline
        ik_successes = 0
        ik_steps = []
        for ti, target in enumerate(targets):
            env = MujocoArmEnv(residual_physics=residual)
            ref, ik_err = solve_reach_reference(
                target, env.joint_ranges, locked_joints=locked
            )
            env._damage = damage

            def ik_action(obs, ref, locked_tuple):
                return joint_reference_action(obs["state"], ref, locked_joints=locked_tuple)

            ok, steps, dist = test_controller(
                "ik", env, target, locked, locked_tuple, ref, ik_action
            )
            if ok:
                ik_successes += 1
                ik_steps.append(steps)
            print(f"IK+PD {domain.domain_id} target_{ti}: {'OK' if ok else 'FAIL'} steps={steps} dist={dist:.4f}", flush=True)

        # WM-hybrid with K=5 calibration
        calibration = collect_controller_domains(
            (domain,), trajectories_per_domain=5, steps=100,
            seed=seed * 100_000, targets=cal
        )
        context = infer_dfwm_context(
            models, domain, calibration, ranges,
            shots=5, latent_steps=30, device=device
        )
        wm = models.dfwm_world_model; wm.eval()

        wm_successes = 0
        wm_steps = []
        for ti, target in enumerate(targets):
            env = MujocoArmEnv(residual_physics=residual)
            ref, ik_err = solve_reach_reference(
                target, env.joint_ranges, locked_joints=locked
            )
            env._damage = damage

            def wm_action(obs, reference, locked_tuple):
                base = joint_reference_action(
                    obs["state"], reference, locked_joints=locked_tuple
                )
                state_t = torch.as_tensor(
                    obs["state"], dtype=torch.float32, device=device
                ).reshape(1, -1)
                action_t = torch.as_tensor(
                    base, dtype=torch.float32, device=device
                ).reshape(1, -1)
                with torch.no_grad():
                    pred, _ = wm.step(
                        state_t, action_t, context.reshape(1, -1), None
                    )
                    q_pred = pred["mean"][0, :5].cpu().numpy()
                correction = np.clip(0.3 * (reference - q_pred), -0.3, 0.3)
                correction[list(locked_tuple)] = 0.0
                return np.clip(base + correction, -1.0, 1.0)

            ok, steps, dist = test_controller(
                "wm", env, target, locked, locked_tuple, ref, wm_action
            )
            if ok:
                wm_successes += 1
                wm_steps.append(steps)
            print(f"WM-Hyb {domain.domain_id} target_{ti}: {'OK' if ok else 'FAIL'} steps={steps} dist={dist:.4f}", flush=True)

        print(f"{domain.domain_id}: IK={ik_successes}/{len(targets)} avg={np.mean(ik_steps):.1f} | WM={wm_successes}/{len(targets)} avg={np.mean(wm_steps):.1f}", flush=True)


if __name__ == "__main__":
    main()
