"""Batch trajectory collection using MuJoCo Warp for GPU-parallel simulation.

For training trajectories with random excitation, runs all trajectories for
a given physics profile in parallel on GPU using mujoco-warp.

Usage:
  from scripts.collect_warp import collect_push_domains_warp
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from src.robotarm.training.sim_data import SimTrajectory
from src.robotarm.training.sim_protocol import DomainSpec
from src.robotarm.envs.residual_physics import ResidualPhysicsConfig

PUSH_XML = "sim/assets/arm_push.xml"
CTRL_SCALE = np.array([1.5, 1.8, 2.4, 1.8, 3.0], dtype=np.float32)

_warp_initialized = False
_cached_base_model: mujoco.MjModel | None = None


def _ensure_warp():
    global _warp_initialized
    if not _warp_initialized:
        wp.init()
        _warp_initialized = True


def _apply_residual_to_model(m: mujoco.MjModel, residual: ResidualPhysicsConfig) -> None:
    """Apply residual physics parameters to a MuJoCo model in-place."""
    # arm_push.xml orders two block slide DoFs before the five arm DoFs.
    m.dof_damping[2:7] *= residual.damping_scale
    m.dof_frictionloss[2:7] *= residual.friction_scale
    m.dof_armature[2:7] *= residual.armature_scale
    if residual.payload_mass_delta_kg > 0:
        tool_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tool")
        old_mass = float(m.body_mass[tool_id])
        new_mass = old_mass + residual.payload_mass_delta_kg
        m.body_mass[tool_id] = new_mass
        if old_mass > 0:
            m.body_inertia[tool_id] *= new_mass / old_mass
        mujoco.mj_setConst(m, mujoco.MjData(m))


def _active_probe_actions(step: int, nworld: int, locked_per_world: list) -> np.ndarray:
    """Deterministic active probing: cycle through joints, expose motor/damping/delay."""
    actions = np.zeros((nworld, 5), dtype=np.float32)
    for w in range(nworld):
        active_joint = step // 20 % 5
        sign = 1.0 if (step // 10) % 2 == 0 else -1.0
        actions[w, active_joint] = 0.7 * sign
        actions[w, (active_joint + 2) % 5] = 0.25 * np.sin(2.0 * np.pi * step / 25.0)
        for j in locked_per_world[w]:
            if j < 5:
                actions[w, j] = 0.0
    return actions


def collect_push_domains_warp(
    domains: tuple[DomainSpec, ...],
    *,
    trajectories_per_domain: int,
    steps: int,
    seed: int,
    block_initial_xy: np.ndarray | None = None,
    excitation: str = "active",  # "active" or "random"
) -> list[SimTrajectory]:
    """Batch-collect random-excitation trajectories using GPU parallel simulation.

    Groups domains by physics profile and runs all trajectories for each
    physics group in a single GPU batch. Within each batch, different worlds
    can have different joint locks (damage configs) because those are applied
    via ctrl zeroing, not model parameters.

    Delay, deadband, and observation noise are applied post-hoc on CPU.
    """
    _ensure_warp()

    if block_initial_xy is None:
        block_initial_xy = np.array([0.24, 0.10])

    # Group domains by residual physics name (same physics -> same model)
    from collections import defaultdict
    physics_groups: dict[str, list[tuple[int, DomainSpec]]] = defaultdict(list)
    for i, domain in enumerate(domains):
        physics_groups[domain.residual_name].append((i, domain))

    all_trajs: dict[int, SimTrajectory] = {}

    for physics_name, indexed_domains in physics_groups.items():
        # Build model with this physics profile
        m_base = mujoco.MjModel.from_xml_path(PUSH_XML)
        residual = indexed_domains[0][1].residual
        _apply_residual_to_model(m_base, residual)

        # nworld = number of domains in this group × trajectories_per_domain
        n_domains_in_group = len(indexed_domains)
        nworld = n_domains_in_group * trajectories_per_domain

        wm = mjw.put_model(m_base)
        wd = mjw.make_data(m_base, nworld=nworld)

        # Set initial qpos/qvel to zero (reset)
        qpos_np = np.zeros((nworld, m_base.nq), dtype=np.float32)
        qvel_np = np.zeros((nworld, m_base.nv), dtype=np.float32)

        # Simulator order is [block_x, block_y, j1..j5].  The world-model
        # contract is [j1..j5, v1..v5, block_xy, block_vxy].
        block_qpos_adr = np.array([0, 1])
        arm_qpos_adr = np.arange(2, 7)
        arm_qvel_adr = np.arange(2, 7)
        block_origin = m_base.body("block").pos[:2].copy()
        for w in range(nworld):
            qpos_np[w, block_qpos_adr] = block_initial_xy - block_origin

        wd.qpos = wp.from_numpy(qpos_np, dtype=wp.float32, device="cuda")
        wd.qvel = wp.from_numpy(qvel_np, dtype=wp.float32, device="cuda")

        # Forward to initialize
        mjw.fwd_kinematics(wm, wd)
        wp.synchronize()

        # Random action generation per world
        rng = np.random.default_rng(seed)
        all_states = np.zeros((nworld, steps + 1, m_base.nq + m_base.nv), dtype=np.float32)
        all_actions = np.zeros((nworld, steps, m_base.nu), dtype=np.float32)
        all_applied_actions = np.zeros_like(all_actions)

        def canonical_state(qpos, qvel):
            return np.concatenate((
                qpos[:, arm_qpos_adr], qvel[:, arm_qvel_adr],
                qpos[:, block_qpos_adr] + block_origin[None, :],
                qvel[:, block_qpos_adr]), axis=1)

        # Locked joint indices per world
        locked_per_world = []
        lock_angles_per_world = []
        for domain_idx, (_, domain) in enumerate(indexed_domains):
            for traj_idx in range(trajectories_per_domain):
                w = domain_idx * trajectories_per_domain + traj_idx
                locked_per_world.append(list(domain.damage.locked))
                lock_angles_per_world.append([
                    domain.damage.lock_angle_of(j) for j in domain.damage.locked
                ])

        # Match MujocoArmEnv.reset: locked joints start exactly at lock angle.
        for w, locked in enumerate(locked_per_world):
            for local, joint in enumerate(locked):
                qpos_np[w, arm_qpos_adr[joint]] = lock_angles_per_world[w][local]
                qvel_np[w, arm_qvel_adr[joint]] = 0.0
        wd.qpos = wp.from_numpy(qpos_np, dtype=wp.float32, device="cuda")
        wd.qvel = wp.from_numpy(qvel_np, dtype=wp.float32, device="cuda")
        mjw.fwd_kinematics(wm, wd)
        all_states[:, 0, :] = canonical_state(qpos_np, qvel_np)

        delay = residual.control_delay_steps
        action_history = [np.zeros((nworld, m_base.nu), dtype=np.float32)
                          for _ in range(delay)]
        previous_velocity_sign = np.zeros((nworld, 5), dtype=np.int8)

        # Simulate
        actions = np.zeros((nworld, m_base.nu), dtype=np.float32)
        for step in range(steps):
            if excitation == "active":
                actions = _active_probe_actions(step, nworld, locked_per_world)
            else:
                # Random excitation
                excitation_noise = rng.uniform(-0.45, 0.45, size=(nworld, m_base.nu)).astype(np.float32)
                actions = 0.75 * actions + 0.25 * excitation_noise
                for w, locked in enumerate(locked_per_world):
                    for j in locked:
                        if j < m_base.nu:
                            actions[w, j] = 0.0

            commanded = actions.copy()
            all_actions[:, step, :] = commanded

            filtered = np.where(
                np.abs(commanded) < residual.action_deadband, 0.0, commanded)
            action_history.append(filtered.copy())
            applied = action_history.pop(0) if delay else action_history.pop()

            # Match the CPU backlash approximation on velocity reversal.
            if np.any(residual.backlash_array > 0):
                qvel_full = wd.qvel.numpy().copy()
                qvel_before = qvel_full[:, arm_qvel_adr].copy()
                current_sign = np.where(qvel_before > 1e-4, 1,
                                        np.where(qvel_before < -1e-4, -1, 0))
                reversal = ((current_sign != 0) & (previous_velocity_sign != 0)
                            & (current_sign != previous_velocity_sign))
                applied[reversal & (residual.backlash_array[None, :] > 0)] = 0.0
                qvel_before[reversal & (residual.backlash_array[None, :] > 0)] *= 0.5
                qvel_full[:, arm_qvel_adr] = qvel_before
                wd.qvel = wp.from_numpy(qvel_full, dtype=wp.float32, device="cuda")
                previous_velocity_sign = np.where(
                    current_sign != 0, current_sign, previous_velocity_sign)

            all_applied_actions[:, step, :] = applied

            ctrl = applied * CTRL_SCALE[None, :] * residual.actuator_scale_array[None, :]
            ctrl = np.clip(ctrl, m_base.actuator_ctrlrange[:, 0],
                           m_base.actuator_ctrlrange[:, 1]).astype(np.float32)
            ctrl_wp = wp.from_numpy(ctrl, dtype=wp.float32, device="cuda")
            wd.ctrl = ctrl_wp
            mjw.step(wm, wd)

            qpos_t = wd.qpos.numpy().copy()
            qvel_t = wd.qvel.numpy().copy()
            # A zero command is not a mechanical lock.  Re-pin after every
            # integration step, exactly as MujocoArmEnv._apply_damage does.
            for w, locked in enumerate(locked_per_world):
                for local, joint in enumerate(locked):
                    qpos_t[w, arm_qpos_adr[joint]] = lock_angles_per_world[w][local]
                    qvel_t[w, arm_qvel_adr[joint]] = 0.0
            wd.qpos = wp.from_numpy(qpos_t, dtype=wp.float32, device="cuda")
            wd.qvel = wp.from_numpy(qvel_t, dtype=wp.float32, device="cuda")
            mjw.fwd_kinematics(wm, wd)
            all_states[:, step + 1, :] = canonical_state(qpos_t, qvel_t)

        wp.synchronize()

        # Apply observation noise post-hoc
        if residual.observation_noise_std > 0:
            noise = rng.normal(0, residual.observation_noise_std, all_states.shape).astype(np.float32)
            all_states += noise

        # Build SimTrajectory objects
        for domain_idx, (orig_idx, domain) in enumerate(indexed_domains):
            for traj_idx in range(trajectories_per_domain):
                w = domain_idx * trajectories_per_domain + traj_idx
                traj = SimTrajectory(
                    domain_id=domain.domain_id,
                    states=torch.from_numpy(all_states[w]).float(),
                    actions=torch.from_numpy(all_actions[w]).float(),
                    applied_actions=torch.from_numpy(all_applied_actions[w]).float(),
                    metadata={
                        "tool_block_contact_steps": 0,  # not tracked in warp mode
                        "block_displacement_m": float(np.linalg.norm(
                            all_states[w, -1, 10:12] - all_states[w, 0, 10:12]
                        )),
                        "warp_collected": True,
                    },
                )
                all_trajs[orig_idx * trajectories_per_domain + traj_idx] = traj

    # Return in original order
    return [all_trajs[i] for i in sorted(all_trajs.keys())]
