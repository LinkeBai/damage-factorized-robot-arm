"""Prepare one genuine model-selected real Push trajectory without moving hardware.

The frozen selective IPWM ranks a small, pre-bounded family of joint-reference
paths.  Every candidate, score, model input and hash is archived.  The chosen
reference is exported for the existing fail-closed STS3215 executor.  This is
open-loop model selection, not receding-horizon control.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robotarm.deployment.fixed_raw_trajectory import (
    VALID_CONDITIONS, load_fixed_raw_trajectory, load_safety_envelope,
    locked_indices_for_condition,
)
from robotarm.deployment.real_calibration import radians_to_ticks, ticks_to_radians
from robotarm.envs.fk import (
    BASE_HEIGHT, SHOULDER_X, L_UPPER, L_FOREARM, WRIST_OFFSET,
    J5_TO_J6_Y, J5_TO_J6_Z, TCP_OFFSET_X,
    forward_kinematics, forward_pose, inverse_kinematics,
)
from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.selective_intervention_rollout import SelectiveInterventionRollout
from robotarm.models.topology_graph_world_model import TopologyGraphConfig


SELECTION_RULE = "minimum_predicted_terminal_object_to_goal_distance"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.loads(response.read())


def interpolate_waypoints(times: np.ndarray, q: np.ndarray, count: int) -> np.ndarray:
    sample_times = np.linspace(times[0], times[-1], count)
    return np.column_stack([np.interp(sample_times, times, q[:, j]) for j in range(5)])


def _batch_transform(q: torch.Tensor, axis: str | None = None,
                     translation: tuple[float, float, float] | None = None) -> torch.Tensor:
    batch = q.shape[0]
    matrix = torch.eye(4, dtype=q.dtype, device=q.device).expand(batch, 4, 4).clone()
    if translation is not None:
        matrix[:, :3, 3] = torch.as_tensor(translation, dtype=q.dtype, device=q.device)
    elif axis == "y":
        c, s = torch.cos(q), torch.sin(q)
        matrix[:, 0, 0], matrix[:, 0, 2] = c, s
        matrix[:, 2, 0], matrix[:, 2, 2] = -s, c
    elif axis == "z":
        c, s = torch.cos(q), torch.sin(q)
        matrix[:, 0, 0], matrix[:, 0, 1] = c, -s
        matrix[:, 1, 0], matrix[:, 1, 1] = s, c
    return matrix


def forward_pose_batched(q: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of the exact provisional GenkiArm TCP pose for N x 5 angles."""
    if q.ndim != 2 or q.shape[1] != 5:
        raise ValueError("q must have shape (N,5)")
    zero = q[:, 0] * 0
    chain = _batch_transform(zero, translation=(0, 0, BASE_HEIGHT))
    operations = (
        _batch_transform(q[:, 0], axis="z"),
        _batch_transform(zero, translation=(SHOULDER_X, 0, 0)),
        _batch_transform(q[:, 1], axis="y"),
        _batch_transform(zero, translation=(0, 0, L_UPPER)),
        _batch_transform(q[:, 2], axis="y"),
        _batch_transform(zero, translation=(0, 0, L_FOREARM)),
        _batch_transform(q[:, 3], axis="y"),
        _batch_transform(zero, translation=(0, 0, WRIST_OFFSET)),
        _batch_transform(q[:, 4], axis="z"),
        _batch_transform(zero, translation=(0, J5_TO_J6_Y, J5_TO_J6_Z)),
        _batch_transform(zero, translation=(TCP_OFFSET_X, 0, 0)),
    )
    for operation in operations:
        chain = torch.bmm(chain, operation)
    return chain


def forward_kinematics_batched(q: torch.Tensor) -> torch.Tensor:
    """Return TCP positions for an ``N x 5`` joint-angle tensor."""
    return forward_pose_batched(q)[:, :3, 3]


def contact_geometry_metrics(
    qrefs: np.ndarray,
    task_axis_xy: np.ndarray,
) -> dict[str, np.ndarray]:
    """Measure trajectory-level push geometry without claiming object motion.

    The initial horizontal tool x-axis is oriented toward the task axis before
    comparison, so the metric is invariant to which gripper face is designated
    as the pushing face.  All quantities concern the TCP only; contact and
    object displacement still require simulation or real observations.
    """
    qrefs = np.asarray(qrefs, dtype=np.float64)
    if qrefs.ndim != 3 or qrefs.shape[2] != 5:
        raise ValueError("qrefs must have shape (candidate,horizon,5)")
    flat = torch.as_tensor(qrefs.reshape(-1, 5), dtype=torch.float64)
    poses = forward_pose_batched(flat).numpy().reshape(*qrefs.shape[:2], 4, 4)
    positions = poses[..., :3, 3]
    tool_x = poses[..., :3, 0]
    axis = np.asarray(task_axis_xy, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)

    # Choose the sign of the initial tool axis that points most nearly along
    # the requested task direction.  Rotation is measured from that initial
    # physical face, not from an unattainable ideal orientation.
    initial_face = tool_x[:, :1, :].copy()
    sign = np.where(
        np.sum(initial_face[..., :2] * axis[None, None, :], axis=-1, keepdims=True) >= 0,
        1.0, -1.0,
    )
    face = tool_x * sign
    face /= np.maximum(np.linalg.norm(face, axis=-1, keepdims=True), 1e-12)
    reference = face[:, :1, :]
    cosine = np.clip(np.sum(face * reference, axis=-1), -1.0, 1.0)
    face_rotation_deg = np.degrees(np.arccos(cosine))
    horizontal_face = face[..., :2]
    horizontal_face /= np.maximum(
        np.linalg.norm(horizontal_face, axis=-1, keepdims=True), 1e-12
    )
    axis_cosine = np.clip(
        np.sum(horizontal_face * axis[None, None, :], axis=-1), -1.0, 1.0
    )
    axis_alignment_error_deg = np.degrees(np.arccos(axis_cosine))

    delta = positions - positions[:, :1, :]
    forward = np.sum(delta[..., :2] * axis[None, None, :], axis=-1)
    lateral = np.sum(delta[..., :2] * normal[None, None, :], axis=-1)
    height = delta[..., 2]
    reverse_step = np.maximum(-(np.diff(forward, axis=1)), 0.0)
    return {
        "maximum_face_rotation_deg": np.max(face_rotation_deg, axis=1),
        "maximum_axis_alignment_error_deg": np.max(axis_alignment_error_deg, axis=1),
        "maximum_height_deviation_m": np.max(np.abs(height), axis=1),
        "maximum_lateral_deviation_m": np.max(np.abs(lateral), axis=1),
        "maximum_reverse_step_m": np.max(reverse_step, axis=1) if reverse_step.shape[1] else np.zeros(len(qrefs)),
        "terminal_forward_tcp_travel_m": forward[:, -1],
    }


@torch.no_grad()
def inverse_kinematics_batched(
    targets: np.ndarray, joint_ranges: np.ndarray, q0: np.ndarray,
    locked_indices: tuple[int, ...], device: torch.device,
    max_steps: int = 100, tolerance: float = 5e-4, damping: float = 1e-3,
    joint_motion_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic GPU-batched damped least-squares position IK."""
    dtype = torch.float64
    target = torch.as_tensor(targets, dtype=dtype, device=device)
    q = torch.as_tensor(q0, dtype=dtype, device=device).reshape(1, 5).expand(len(targets), -1).clone()
    limits = torch.as_tensor(joint_ranges, dtype=dtype, device=device)
    locked = set(locked_indices)
    free = [index for index in range(5) if index not in locked]
    weights = np.ones(5) if joint_motion_weights is None else np.asarray(joint_motion_weights, dtype=float)
    if weights.shape != (5,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("joint_motion_weights must contain five finite positive values")
    inverse_free_weights = torch.as_tensor(1.0 / weights[free], dtype=dtype, device=device)
    epsilon = 1e-5
    for _ in range(max_steps):
        position = forward_kinematics_batched(q)
        active_indices = torch.nonzero(
            torch.linalg.vector_norm(target - position, dim=1) > tolerance,
            as_tuple=False,
        ).flatten()
        if active_indices.numel() == 0 or not free:
            break
        q_active = q[active_indices].clone()
        position_active = position[active_indices]
        error = target[active_indices] - position_active
        columns = []
        for joint in free:
            probe = q_active.clone(); probe[:, joint] += epsilon
            columns.append((forward_kinematics_batched(probe) - position_active) / epsilon)
        jacobian = torch.stack(columns, dim=2)
        identity = torch.eye(3, dtype=dtype, device=device).expand(len(active_indices), 3, 3)
        weighted_jacobian_t = jacobian.transpose(1, 2) * inverse_free_weights[None, :, None]
        lhs = torch.bmm(jacobian, weighted_jacobian_t) + damping * identity
        delta = torch.bmm(weighted_jacobian_t, torch.linalg.solve(lhs, error.unsqueeze(2))).squeeze(2)
        q_active[:, free] += torch.clamp(delta, -0.15, 0.15)
        q_active = torch.maximum(torch.minimum(q_active, limits[:, 1]), limits[:, 0])
        for joint in locked_indices:
            q_active[:, joint] = float(q0[joint])
        q[active_indices] = q_active
    errors = torch.linalg.vector_norm(target - forward_kinematics_batched(q), dim=1)
    return q.cpu().numpy(), errors.cpu().numpy()


def equivalent_shared_robot_path(model):
    """Elide a duplicate robot rollout only for the verified narrow architecture.

    Both robot transitions depend only on q/qd/action/mask/angle and identical
    hidden states/weights. Induction therefore makes carrier robot coordinates
    identical to intervention robot coordinates at every step. The intervention
    object branch is retained unchanged. This is an inference optimization only.
    """
    if type(model) is not SelectiveInterventionRollout or not model.analytic_projection:
        raise ValueError("shared robot optimization requires projected selective rollout")
    a, b = model.intervention_model, model.carrier_model
    if type(a) is not BlockTriangularDPWM or type(b) is not BlockTriangularDPWM:
        raise ValueError("unsupported robot architecture")
    disabled = ("contact_conditioned_robot", "independent_object_encoder",
                "reaction_rank", "linear_physical_reaction", "shadow_object_rank",
                "intervention_context_ramp", "intervention_residual_decay",
                "reaction_event_decay")
    if any(getattr(branch, key) for branch in (a, b) for key in disabled):
        raise ValueError("object-conditioned or auxiliary recurrent robot state is unsupported")
    equal_settings = ("cfg", "analytic_projection", "robot_expert_count",
                      "robot_position_delta_scale", "robot_velocity_delta_scale",
                      "kinematic_integration_dt", "kinematic_position_blend")
    if any(getattr(a, key) != getattr(b, key) for key in equal_settings):
        raise ValueError("robot transition settings differ")
    if not a.analytic_projection or model.robot_dim != 2 * a.cfg.dof:
        raise ValueError("robot projection layout differs")
    for name in ("robot_encoder", "robot_message", "robot_update", "robot_temporal",
                 "robot_head", "additional_robot_experts"):
        left, right = getattr(a, name).state_dict(), getattr(b, name).state_dict()
        if left.keys() != right.keys() or any(not torch.equal(left[k], right[k]) for k in left):
            raise ValueError("robot transition weights differ")
    if a.training or b.training:
        raise ValueError("shared robot optimization is inference-only")
    return a


def load_model(checkpoint: Path, config: Path, device: torch.device, *, shared_robot: bool = False):
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    topology = TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"]))
    model = BlockTriangularDPWM(
        topology,
        compact_bridge_object_head=bool(cfg.get("compact_bridge_object_head", False)),
        geometric_object_rank=int(cfg.get("geometric_object_rank", 0)),
        analytic_projection=True,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    carrier = copy.deepcopy(model)
    with torch.no_grad():
        for name in ("geometric_object_head", "global_residual_head", "intervention_object_head"):
            if hasattr(carrier, name):
                for parameter in getattr(carrier, name).parameters():
                    parameter.zero_()
    rollout = SelectiveInterventionRollout(model.eval(), carrier.eval(), analytic_projection=True).to(device).eval()
    return equivalent_shared_robot_path(rollout) if shared_robot else rollout


@torch.no_grad()
def score_reference(model, initial: np.ndarray, qref: np.ndarray, goal: np.ndarray,
                    mask: np.ndarray, angle: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    state = torch.as_tensor(initial, dtype=torch.float32, device=device).reshape(1, -1)
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device).reshape(1, -1)
    angle_t = torch.as_tensor(angle, dtype=torch.float32, device=device).reshape(1, -1)
    hidden, actions, predicted = None, [], []
    for reference in qref:
        q = state[0, :5].detach().cpu().numpy()
        qd = state[0, 5:10].detach().cpu().numpy()
        action = np.clip(5.0 * (reference - q) - 0.5 * qd, -1.0, 1.0)
        action[mask.astype(bool)] = 0.0
        action_t = torch.as_tensor(action, dtype=torch.float32, device=device).reshape(1, -1)
        state, hidden = model.step(state, action_t, mask_t, angle_t, hidden)
        actions.append(action)
        predicted.append(state[0].detach().cpu().numpy())
    trajectory = np.asarray(predicted)
    return float(np.linalg.norm(trajectory[-1, 10:12] - goal)), np.asarray(actions), trajectory


@torch.no_grad()
def score_references_batched(model, initial: np.ndarray, qrefs: np.ndarray,
                             goal: np.ndarray, mask: np.ndarray, angle: np.ndarray,
                             batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score large candidate banks on GPU without candidate replication."""
    device = next(model.parameters()).device
    all_scores, all_actions, all_predictions = [], [], []
    for start in range(0, len(qrefs), batch_size):
        refs = torch.as_tensor(qrefs[start:start + batch_size], dtype=torch.float32, device=device)
        batch = refs.shape[0]
        state = torch.as_tensor(initial, dtype=torch.float32, device=device).reshape(1, -1).expand(batch, -1).clone()
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device).reshape(1, -1).expand(batch, -1)
        angle_t = torch.as_tensor(angle, dtype=torch.float32, device=device).reshape(1, -1).expand(batch, -1)
        hidden, actions, predictions = None, [], []
        for depth in range(refs.shape[1]):
            action = torch.clamp(5.0 * (refs[:, depth] - state[:, :5]) - 0.5 * state[:, 5:10], -1.0, 1.0)
            action = action * (1.0 - mask_t)
            state, hidden = model.step(state, action, mask_t, angle_t, hidden)
            actions.append(action.cpu().numpy())
            predictions.append(state.cpu().numpy())
        all_scores.append(torch.linalg.vector_norm(state[:, 10:12] - torch.as_tensor(goal, device=device), dim=-1).cpu().numpy())
        all_actions.append(np.stack(actions, axis=1))
        all_predictions.append(np.stack(predictions, axis=1))
    return np.concatenate(all_scores), np.concatenate(all_actions), np.concatenate(all_predictions)


@torch.no_grad()
def score_references_terminal_only(
    model, initial: np.ndarray, qrefs: np.ndarray, goal: np.ndarray,
    mask: np.ndarray, angle: np.ndarray, batch_size: int,
) -> np.ndarray:
    """Online IPWM scoring path that retains only terminal candidate costs."""
    device = next(model.parameters()).device
    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)
    scores: list[np.ndarray] = []
    for start in range(0, len(qrefs), batch_size):
        refs = torch.as_tensor(
            qrefs[start:start + batch_size], dtype=torch.float32, device=device
        )
        batch = refs.shape[0]
        state = torch.as_tensor(
            initial, dtype=torch.float32, device=device
        ).reshape(1, -1).expand(batch, -1).clone()
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device).reshape(1, -1).expand(batch, -1)
        angle_t = torch.as_tensor(angle, dtype=torch.float32, device=device).reshape(1, -1).expand(batch, -1)
        hidden = None
        for depth in range(refs.shape[1]):
            action = torch.clamp(
                5.0 * (refs[:, depth] - state[:, :5]) - 0.5 * state[:, 5:10],
                -1.0, 1.0,
            )
            action = action * (1.0 - mask_t)
            state, hidden = model.step(state, action, mask_t, angle_t, hidden)
        scores.append(torch.linalg.vector_norm(state[:, 10:12] - goal_t, dim=-1).cpu().numpy())
    return np.concatenate(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=VALID_CONDITIONS, required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--template-trajectory-id", required=True)
    parser.add_argument("--template-condition", choices=VALID_CONDITIONS,
                        help="Condition encoded in the timing/template CSV; defaults to --condition")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/icra_confirmation_d3_query_selective_w10/seed27/model.pt")
    parser.add_argument("--model-config", type=Path, default=ROOT / "config/experiment/icra_primary_d2d4_eval_strict_3seed_v1.yaml")
    parser.add_argument("--axis-calibration", type=Path, default=ROOT / "results/real_robot/push_axis_current_epoch_20260903.json")
    parser.add_argument("--safety", type=Path, default=ROOT / "hardware/safety_limits.yaml")
    parser.add_argument("--preview-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--offline-observation", type=Path,
        help="Frozen joint/cube observation JSON for preparation only; execution must revalidate live state",
    )
    parser.add_argument("--scales", default="0.90,1.00,1.10")
    parser.add_argument("--scale-range", help="min,max,count; overrides --scales")
    parser.add_argument("--lateral-offsets-px", default="0",
                        help="Comma-separated TCP offsets normal to the measured Push axis")
    parser.add_argument("--lateral-range-px", help="min,max,count; overrides --lateral-offsets-px")
    parser.add_argument("--candidate-source", choices=("exact_ik", "template_local", "constant_pose_j1j2"),
                        default="exact_ik")
    parser.add_argument(
        "--joint-motion-weights", default="1,1,1,1,1",
        help="Five positive IK motion costs for J1..J5; larger values discourage a joint",
    )
    parser.add_argument(
        "--constant-pose-delta-range-deg", default="-30,30,10000",
        help="min,max,count J3 change; J4 receives the exact opposite change",
    )
    parser.add_argument("--execution-duration-scale", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--max-ik-error-m", type=float, default=0.005)
    parser.add_argument("--max-push-face-rotation-deg", type=float, default=10.0)
    parser.add_argument("--max-push-axis-alignment-error-deg", type=float, default=15.0)
    parser.add_argument("--max-push-height-deviation-mm", type=float, default=5.0)
    parser.add_argument("--max-push-lateral-deviation-mm", type=float, default=5.0)
    parser.add_argument("--max-push-reverse-step-mm", type=float, default=0.5)
    parser.add_argument(
        "--push-geometry-policy", choices=("strict", "record"), default="strict",
        help="strict filters candidates; record archives diagnostics without making them a task gate",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        joint_motion_weights = np.asarray(
            [float(value) for value in args.joint_motion_weights.split(",")], dtype=float
        )
    except ValueError as error:
        raise SystemExit("--joint-motion-weights requires five positive numbers") from error
    if (joint_motion_weights.shape != (5,) or not np.all(np.isfinite(joint_motion_weights))
            or np.any(joint_motion_weights <= 0)):
        raise SystemExit("--joint-motion-weights requires five finite positive numbers")

    axis = json.loads(args.axis_calibration.read_text(encoding="utf-8"))
    if axis.get("status") != "PASS":
        raise SystemExit("task-axis calibration is not PASS")
    if args.offline_observation:
        frozen_observation = json.loads(args.offline_observation.read_text(encoding="utf-8"))
        if frozen_observation.get("schema_version") != 1:
            raise SystemExit("offline observation schema_version must be 1")
        joints = {"status": "ok", "current_raw": frozen_observation["joint_raw"]}
        cube = {
            "status": "PASS", "current_px": frozen_observation["object_pixel"],
            "task_goal_px": frozen_observation["goal_pixel"],
        }
        observation_source = "frozen_offline_preparation_requires_live_execution_revalidation"
    else:
        joints = get_json(args.preview_url + "/joint-status")
        cube = get_json(args.preview_url + "/cube-status")
        if joints.get("status") != "ok" or cube.get("status") != "PASS":
            raise SystemExit("fresh joint/cube observation is not ready")
        observation_source = "preview_read_only_status_request"
    safety = load_safety_envelope(args.safety)
    zero = np.asarray([j.zero_raw for j in safety.joints])
    direction = np.asarray([j.direction for j in safety.joints])
    current_raw = np.asarray(joints["current_raw"], dtype=int)
    q0 = ticks_to_radians(current_raw, zero, direction)
    slope = np.asarray(axis["diagnostics"]["base_xy_per_pixel"], dtype=float)
    intercept = np.asarray(axis["diagnostics"]["base_xy_intercept_m"], dtype=float)
    object_xy = float(cube["current_px"][0]) * slope + intercept
    goal_xy = float(cube["task_goal_px"][0]) * slope + intercept
    initial = np.r_[q0, np.zeros(5), object_xy, np.zeros(2)]

    template = load_fixed_raw_trajectory(
        args.template, trajectory_id=args.template_trajectory_id,
        condition=args.template_condition or args.condition,
        safety=safety, maximum_speed_deg_s=5.0,
    )
    if args.execution_duration_scale <= 0.0:
        raise SystemExit("execution-duration-scale must be positive; the executor re-audits speed")
    times = np.asarray([row.time_s for row in template.waypoints], dtype=float)
    times *= args.execution_duration_scale
    template_raw = np.asarray([row.targets_raw for row in template.waypoints], dtype=int)
    template_q = ticks_to_radians(template_raw, zero, direction)
    def values(explicit: str, range_spec: str | None) -> list[float]:
        if range_spec is None:
            return [float(x) for x in explicit.split(",")]
        low, high, count = range_spec.split(",")
        count_i = int(count)
        if count_i < 1:
            raise SystemExit("candidate range count must be positive")
        return np.linspace(float(low), float(high), count_i).tolist()

    scales = values(args.scales, args.scale_range)
    lateral_offsets_px = values(args.lateral_offsets_px, args.lateral_range_px)
    locked_indices = locked_indices_for_condition(args.condition)
    mask = np.zeros(5, dtype=float)
    angle = np.zeros(5, dtype=float)
    for locked in locked_indices:
        mask[locked] = 1.0
        angle[locked] = q0[locked]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint.resolve(), args.model_config.resolve(), device)
    joint_ranges = np.asarray([[j.min_deg, j.max_deg] for j in safety.joints], dtype=float)
    joint_ranges = np.deg2rad(joint_ranges)
    tcp0 = forward_kinematics(q0)
    task_delta = goal_xy - object_xy
    axis_unit = np.asarray(axis["diagnostics"]["unit_axis_base_xy"], dtype=float)
    normal_unit = np.asarray([-axis_unit[1], axis_unit[0]], dtype=float)
    metres_per_pixel = float(axis["diagnostics"]["metres_per_pixel"])
    if args.candidate_source == "constant_pose_j1j2":
        if args.condition != "J1+J2":
            raise SystemExit("constant_pose_j1j2 is defined only for condition J1+J2")
        low, high, count = args.constant_pose_delta_range_deg.split(",")
        delta_degrees = np.linspace(float(low), float(high), int(count)).tolist()
        candidate_parameters = [(delta, 0.0) for delta in delta_degrees]
    else:
        delta_degrees = []
        candidate_parameters = [(scale, lateral) for scale in scales for lateral in lateral_offsets_px]
    qrefs, endpoints, ik_errors, geometric_endpoint_errors = [], [], [], []
    phase = ((times - times[0]) / (times[-1] - times[0]))[:, None]
    if args.candidate_source == "exact_ik":
        target_tcps = np.repeat(tcp0[None, :], len(candidate_parameters), axis=0)
        for index, (scale, lateral_px) in enumerate(candidate_parameters):
            target_tcps[index, :2] += scale * task_delta + lateral_px * metres_per_pixel * normal_unit
        endpoint_array, error_array = inverse_kinematics_batched(
            target_tcps, joint_ranges, q0, locked_indices, device,
            joint_motion_weights=joint_motion_weights,
        )
        candidate_paths = q0[None, None, :] + phase[None, :, :] * (
            endpoint_array[:, None, :] - q0[None, None, :]
        )
        qrefs = np.asarray([
            interpolate_waypoints(times, candidate_q, args.horizon)
            for candidate_q in candidate_paths
        ])
        endpoints = endpoint_array.tolist()
        ik_errors = error_array.tolist()
        geometric_endpoint_errors = error_array.tolist()
    elif args.candidate_source == "template_local":
        for scale, lateral_px in candidate_parameters:
            candidate_q = q0 + scale * (template_q - template_q[0])
            # Small normal offsets are represented by base-yaw changes around
            # the local TCP radius.  This remains a bounded local candidate,
            # not a general 2-D visual-servo controller.
            radius = max(float(np.linalg.norm(tcp0[:2])), 0.1)
            candidate_q[:, 0] += phase[:, 0] * lateral_px * metres_per_pixel / radius
            if locked_indices:
                candidate_q[:, locked_indices] = q0[list(locked_indices)]
            q_end = candidate_q[-1]
            intended = tcp0.copy(); intended[:2] += scale * task_delta + lateral_px * metres_per_pixel * normal_unit
            geometric_error = float(np.linalg.norm(forward_kinematics(q_end) - intended))
            qref = interpolate_waypoints(times, candidate_q, args.horizon)
            qrefs.append(qref)
            endpoints.append(q_end)
            # A template-local candidate is already an explicit joint-space
            # trajectory and does not invoke IK.  Treating its task-endpoint
            # residual as an IK convergence error incorrectly rejects every
            # local extension.  Keep the residual as a separate diagnostic;
            # IPWM scores the actual FK-consistent rollout.
            ik_errors.append(0.0)
            geometric_endpoint_errors.append(geometric_error)
    else:
        # J1/J2/J5 remain exactly fixed. Equal and opposite J3/J4 changes keep
        # q3+q4, and therefore the complete TCP orientation, exactly constant.
        for delta_deg, _ in candidate_parameters:
            delta = np.deg2rad(delta_deg)
            q_end = q0.copy()
            q_end[2] += delta
            q_end[3] -= delta
            candidate_q = q0[None, :] + phase * (q_end - q0)[None, :]
            qrefs.append(interpolate_waypoints(times, candidate_q, args.horizon))
            endpoints.append(q_end)
            ik_errors.append(0.0)
            intended = tcp0.copy(); intended[:2] += task_delta
            geometric_endpoint_errors.append(
                float(np.linalg.norm(forward_kinematics(q_end) - intended))
            )
    qrefs = np.asarray(qrefs)
    scores, actions, predictions = score_references_batched(
        model, initial, qrefs, goal_xy, mask, angle, args.score_batch_size,
    )
    raw_scores = scores.astype(float)
    # The executor enforces the same 5 deg/s limit after selection.  Apply it
    # before ranking as well so IPWM cannot repeatedly select an IK-valid but
    # physically non-executable branch.  Because selected waypoints are linear
    # in elapsed time, the endpoint displacement divided by total duration is
    # the exact maximum speed written to the selected trajectory CSV.
    trajectory_duration_s = float(times[-1] - times[0])
    candidate_max_speeds_deg_s = np.max(
        np.abs(np.degrees(np.asarray(endpoints) - q0[None, :])), axis=1
    ) / trajectory_duration_s
    ik_eligible = np.asarray(ik_errors) <= args.max_ik_error_m
    joint_limit_eligible = np.all(
        (qrefs >= joint_ranges[None, None, :, 0] - 1e-12)
        & (qrefs <= joint_ranges[None, None, :, 1] + 1e-12),
        axis=(1, 2),
    )
    speed_eligible = candidate_max_speeds_deg_s <= 5.0 + 1e-9
    geometry = contact_geometry_metrics(qrefs, axis_unit)
    face_eligible = geometry["maximum_face_rotation_deg"] <= args.max_push_face_rotation_deg + 1e-9
    alignment_eligible = geometry["maximum_axis_alignment_error_deg"] <= args.max_push_axis_alignment_error_deg + 1e-9
    height_eligible = geometry["maximum_height_deviation_m"] <= args.max_push_height_deviation_mm / 1000.0 + 1e-12
    lateral_eligible = geometry["maximum_lateral_deviation_m"] <= args.max_push_lateral_deviation_mm / 1000.0 + 1e-12
    monotonic_eligible = geometry["maximum_reverse_step_m"] <= args.max_push_reverse_step_mm / 1000.0 + 1e-12
    geometry_eligible = face_eligible & alignment_eligible & height_eligible & lateral_eligible & monotonic_eligible
    task_geometry_eligible = (
        geometry_eligible
        if args.push_geometry_policy == "strict"
        else np.ones_like(geometry_eligible, dtype=bool)
    )
    eligible = ik_eligible & joint_limit_eligible & speed_eligible & task_geometry_eligible
    if not np.any(eligible):
        args.output_dir.mkdir(parents=True, exist_ok=False)
        np.savez(args.output_dir / "rejected_candidates.npz",
                 q_reference_rad=qrefs, scores=raw_scores, eligible=eligible,
                 **geometry)
        rejection = {
            "status": "PREFLIGHT_REJECTED_NO_MOTION",
            "condition": args.condition,
            "candidate_count": len(qrefs), "device": str(device),
            "geometry_pass_counts": {
                "face": int(np.sum(face_eligible)),
                "alignment": int(np.sum(alignment_eligible)),
                "height": int(np.sum(height_eligible)),
                "lateral": int(np.sum(lateral_eligible)),
                "monotonic": int(np.sum(monotonic_eligible)),
            },
            "geometry_ranges": {k: {"min": float(np.min(v)), "max": float(np.max(v))}
                                for k, v in geometry.items()},
            "claim_boundary": "Candidate family rejection; not proof of global unreachability or a physical trial.",
        }
        (args.output_dir / "rejection.json").write_text(json.dumps(rejection, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(
            "no candidate passed the joint frozen gates: "
            f"ik={int(np.sum(ik_eligible))}/{len(ik_eligible)}, "
            f"limits={int(np.sum(joint_limit_eligible))}/{len(joint_limit_eligible)}, "
            f"speed={int(np.sum(speed_eligible))}/{len(speed_eligible)}, "
            f"push_geometry={int(np.sum(geometry_eligible))}/{len(geometry_eligible)}, "
            f"all={int(np.sum(eligible))}/{len(eligible)}, "
            f"min_speed={float(np.min(candidate_max_speeds_deg_s)):.6f} deg/s"
        )
    selection_scores = np.where(eligible, raw_scores, np.inf)
    scores = raw_scores.tolist()
    selected = int(np.argmin(selection_scores))
    # Export the exact horizon reference that IPWM scored.  Reconstructing a
    # shorter endpoint-only line here would break decision/execution identity
    # whenever the candidate library contains a non-linear reference path.
    selected_waypoints_q = qrefs[selected]
    selected_times = np.linspace(times[0], times[-1], len(selected_waypoints_q))
    selected_raw = radians_to_ticks(selected_waypoints_q, zero, direction)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    archive = args.output_dir / "ipwm_candidates.npz"
    np.savez_compressed(archive, scales=np.asarray([x[0] for x in candidate_parameters]),
                        lateral_offsets_px=np.asarray([x[1] for x in candidate_parameters]),
                        q_reference_rad=qrefs,
                        model_actions=actions, predicted_states=predictions,
                        predicted_scores=np.asarray(scores), selected_candidate_index=selected,
                        ik_eligible=ik_eligible, speed_eligible=speed_eligible,
                        joint_limit_eligible=joint_limit_eligible,
                        push_geometry_eligible=geometry_eligible,
                        push_geometry_policy=np.asarray(args.push_geometry_policy),
                        face_eligible=face_eligible, alignment_eligible=alignment_eligible,
                        height_eligible=height_eligible,
                        lateral_eligible=lateral_eligible, monotonic_eligible=monotonic_eligible,
                        **geometry,
                        candidate_max_speed_deg_s=candidate_max_speeds_deg_s,
                        selection_eligible=eligible, selection_scores=selection_scores,
                        initial_state=initial, goal_xy_m=goal_xy, mask=mask, lock_angle=angle,
                        ik_endpoint_rad=np.asarray(endpoints), ik_error_m=np.asarray(ik_errors),
                        geometric_endpoint_error_m=np.asarray(geometric_endpoint_errors),
                        exact_fk_start_m=tcp0, task_delta_base_xy_m=task_delta)
    selected_csv = args.output_dir / "selected_trajectory.csv"
    trajectory_id = f"{args.trial_id}_ipwm_selected"
    with selected_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trajectory_id", "condition", "waypoint_index", "time_s", "j1_raw", "j2_raw", "j3_raw", "j4_raw", "j5_raw"])
        for i, (time_s, targets) in enumerate(zip(selected_times, selected_raw)):
            writer.writerow([trajectory_id, args.condition, i, f"{time_s:.3f}", *map(int, targets)])
    # Reparse through the exact executor validator before declaring it executable.
    validated = load_fixed_raw_trajectory(selected_csv, trajectory_id=trajectory_id,
                                           condition=args.condition, safety=safety,
                                           maximum_speed_deg_s=5.0)
    preparation = {
        "schema_version": "real_ipwm_preparation_v1", "trial_id": args.trial_id,
        "method": "selective_ipwm", "planner_mode": "open_loop_sequence",
        "condition": args.condition, "device": str(device),
        "initial_observation": {
            "joint_raw": current_raw.tolist(), "joint_position_rad": q0.tolist(),
            "joint_velocity_rad_s": [0.0] * 5, "object_pixel": cube["current_px"],
            "goal_pixel": cube["task_goal_px"], "object_xy_m": object_xy.tolist(),
            "goal_xy_m": goal_xy.tolist(), "task_axis_calibration_sha256": sha256(args.axis_calibration),
            "telemetry_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timestamp_source": observation_source,
            "offline_observation": (
                None if not args.offline_observation else {
                    "path": str(args.offline_observation.resolve()),
                    "sha256": sha256(args.offline_observation),
                    "execution_revalidation_required": True,
                }
            ),
        },
        "decision": {"candidate_count": len(candidate_parameters),
                     "eligible_candidate_count": int(np.sum(eligible)),
                     "ik_eligible_candidate_count": int(np.sum(ik_eligible)),
                     "joint_limit_eligible_candidate_count": int(np.sum(joint_limit_eligible)),
                     "speed_eligible_candidate_count": int(np.sum(speed_eligible)),
                     "push_geometry_eligible_candidate_count": int(np.sum(geometry_eligible)),
                     "push_geometry_policy": args.push_geometry_policy,
                     "push_geometry_gate": {
                         "maximum_face_rotation_deg": args.max_push_face_rotation_deg,
                         "maximum_axis_alignment_error_deg": args.max_push_axis_alignment_error_deg,
                         "maximum_height_deviation_mm": args.max_push_height_deviation_mm,
                         "maximum_lateral_deviation_mm": args.max_push_lateral_deviation_mm,
                         "maximum_reverse_step_mm": args.max_push_reverse_step_mm,
                         "semantics": (
                             "TCP trajectory gate only; does not prove contact or object displacement; "
                             + ("enforced before ranking" if args.push_geometry_policy == "strict" else "recorded as diagnostic, not enforced")
                         ),
                     },
                     "maximum_speed_deg_s": 5.0,
                     "max_ik_error_m": args.max_ik_error_m,
                     "ik_solver": (
                         "deterministic_gpu_batched_active_set_dls_100_steps"
                         if args.candidate_source == "exact_ik" else "not_applicable_joint_template_local"
                     ),
                     "candidate_scales": [x[0] for x in candidate_parameters],
                     "candidate_lateral_offsets_px": [x[1] for x in candidate_parameters],
                     "predicted_scores": scores, "selected_candidate_index": selected,
                     "selection_rule": SELECTION_RULE, "ik_error_m": ik_errors,
                     "geometric_endpoint_error_m": geometric_endpoint_errors},
        "files": {
            "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
            "model_config": {"path": str(args.model_config.resolve()), "sha256": sha256(args.model_config)},
            "candidate_archive": {"path": str(archive.resolve()), "sha256": sha256(archive)},
            "selected_trajectory": {"path": str(selected_csv.resolve()), "sha256": sha256(selected_csv)},
            "action_bridge": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
            "simulation_model": {"path": str((ROOT / "sim/assets/genkiarm_push.xml").resolve()),
                                 "sha256": sha256(ROOT / "sim/assets/genkiarm_push.xml")},
        },
        "bridge": {"type": "IPWM-ranked exact-GenkiArm-FK/IK joint-reference library",
                   "simulation_model": "genkiarm_push.xml", "safety_audit": "PASS",
                   "clipped_action_count": 0,
                   "candidate_source": args.candidate_source,
                   "joint_motion_weights": joint_motion_weights.tolist(),
                   "candidate_parameter_semantics": (
                       "j3_delta_deg_with_equal_opposite_j4_delta"
                       if args.candidate_source == "constant_pose_j1j2"
                       else "task_displacement_scale_and_lateral_offset_px"
                   ),
                   "execution_duration_scale": args.execution_duration_scale,
                   "maximum_commanded_speed_deg_s": validated.maximum_commanded_speed_deg_s},
        "claim_boundary": "Model-selected open-loop trajectory; not receding-horizon visual feedback and not evidence of superiority.",
    }
    manifest = args.output_dir / "ipwm_preparation_manifest.json"
    manifest.write_text(json.dumps(preparation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output_dir.resolve()),
                      "device": str(device), "candidate_count": len(scores),
                      "score_min_all": min(scores), "score_max_all": max(scores),
                      "score_min_eligible": float(selection_scores[selected]),
                      "eligible_candidate_count": int(np.sum(eligible)),
                      "ik_eligible_candidate_count": int(np.sum(ik_eligible)),
                      "speed_eligible_candidate_count": int(np.sum(speed_eligible)),
                      "selected": selected,
                      "selected_scale": candidate_parameters[selected][0],
                      "selected_lateral_offset_px": candidate_parameters[selected][1],
                      "trajectory_id": trajectory_id,
                      "maximum_speed_deg_s": validated.maximum_commanded_speed_deg_s}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
