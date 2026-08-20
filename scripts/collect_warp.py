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

_warp_initialized = False
_cached_base_model: mujoco.MjModel | None = None


def _ensure_warp():
    global _warp_initialized
    if not _warp_initialized:
        wp.init()
        _warp_initialized = True


def _apply_residual_to_model(m: mujoco.MjModel, residual: ResidualPhysicsConfig) -> None:
    """Apply residual physics parameters to a MuJoCo model in-place."""
    # Actuator gear (strength scaling)
    for i, scale in enumerate(residual.actuator_scale):
        if i < m.nu:
            m.actuator_gear[i, 0] *= scale

    # Damping
    m.dof_damping[:] *= residual.damping_scale

    # Friction (geom-level)
    m.geom_friction[:, 0] *= residual.friction_scale


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

        # Set block initial position for each world
        # block qpos indices: last 7 (3 pos + 4 quat) if freejoint
        block_qpos_start = m_base.nq - 7  # assume block has freejoint at end
        for w in range(nworld):
            qpos_np[w, block_qpos_start] = block_initial_xy[0]
            qpos_np[w, block_qpos_start + 1] = block_initial_xy[1]
            qpos_np[w, block_qpos_start + 2] = 0.025  # block z height
            qpos_np[w, block_qpos_start + 3] = 1.0    # quat w

        wd.qpos = wp.from_numpy(qpos_np, dtype=wp.float32, device="cuda")
        wd.qvel = wp.from_numpy(qvel_np, dtype=wp.float32, device="cuda")

        # Forward to initialize
        mjw.fwd_kinematics(wm, wd)
        wp.synchronize()

        # Random action generation per world
        rng = np.random.default_rng(seed)
        all_states = np.zeros((nworld, steps + 1, m_base.nq + m_base.nv), dtype=np.float32)
        all_actions = np.zeros((nworld, steps, m_base.nu), dtype=np.float32)

        # Capture initial states
        qpos_init = wd.qpos.numpy().copy()
        qvel_init = wd.qvel.numpy().copy()
        all_states[:, 0, :m_base.nq] = qpos_init
        all_states[:, 0, m_base.nq:] = qvel_init

        # Locked joint indices per world
        locked_per_world = []
        for domain_idx, (_, domain) in enumerate(indexed_domains):
            for traj_idx in range(trajectories_per_domain):
                w = domain_idx * trajectories_per_domain + traj_idx
                locked_per_world.append(list(domain.damage.locked))

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

            all_actions[:, step, :] = actions

            ctrl_wp = wp.from_numpy(actions.copy(), dtype=wp.float32, device="cuda")
            wd.ctrl = ctrl_wp
            mjw.step(wm, wd)

            qpos_t = wd.qpos.numpy().copy()
            qvel_t = wd.qvel.numpy().copy()
            all_states[:, step + 1, :m_base.nq] = qpos_t
            all_states[:, step + 1, m_base.nq:] = qvel_t

        wp.synchronize()

        # Apply control delay post-hoc (shift actions)
        delay = residual.control_delay_steps
        if delay > 0:
            delayed_actions = np.zeros_like(all_actions)
            delayed_actions[:, delay:, :] = all_actions[:, :-delay, :]
            all_actions = delayed_actions

        # Apply deadband post-hoc
        if residual.action_deadband > 0:
            db = residual.action_deadband
            all_actions = np.where(np.abs(all_actions) < db, 0.0, all_actions)

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
                    applied_actions=torch.from_numpy(all_actions[w]).float(),
                    metadata={
                        "tool_block_contact_steps": 0,  # not tracked in warp mode
                        "block_displacement_m": float(np.linalg.norm(
                            all_states[w, -1, block_qpos_start:block_qpos_start + 2] -
                            all_states[w, 0, block_qpos_start:block_qpos_start + 2]
                        )),
                        "warp_collected": True,
                    },
                )
                all_trajs[orig_idx * trajectories_per_domain + traj_idx] = traj

    # Return in original order
    return [all_trajs[i] for i in sorted(all_trajs.keys())]
