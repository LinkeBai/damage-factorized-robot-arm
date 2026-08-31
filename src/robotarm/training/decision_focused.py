"""Decision-focused objectives and metrics for matched action candidates.

The functions operate on world-model terminal state predictions. They are not
an external ranker and introduce no learned parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch

from robotarm.models.contact_geometry import pusher_box_contact_gate


@dataclass(frozen=True)
class PairedCandidateBatch:
    """Matched candidate rollouts grouped by current state and diagnosis."""

    initial_state: torch.Tensor  # [groups, state_dim]
    actions: torch.Tensor  # [groups, candidates, horizon, action_dim]
    true_terminal_object: torch.Tensor  # [groups, candidates, object_cost_dim]
    goal: torch.Tensor  # [groups, object_cost_dim]
    lock_mask: torch.Tensor  # [groups, dof]
    lock_angle: torch.Tensor  # [groups, dof]
    true_contact: torch.Tensor | None = None  # [groups, candidates, horizon]
    true_success: torch.Tensor | None = None  # [groups, candidates]
    true_min_contact_distance: torch.Tensor | None = None  # [groups, candidates, horizon]

    def validate(self) -> None:
        groups, candidates, _, _ = self.actions.shape
        if self.initial_state.shape[0] != groups:
            raise ValueError("each candidate group must have one shared initial state")
        if self.true_terminal_object.shape[:2] != (groups, candidates):
            raise ValueError("true terminal outcomes must match group and candidate axes")
        if self.goal.shape != (groups, self.true_terminal_object.shape[-1]):
            raise ValueError("each candidate group must have one goal")
        if self.lock_mask.shape[0] != groups or self.lock_angle.shape != self.lock_mask.shape:
            raise ValueError("diagnosis tensors must have one matched row per group")
        if self.true_contact is not None and self.true_contact.shape != (
            groups, candidates, self.actions.shape[2]
        ):
            raise ValueError("contact labels must match group, candidate and horizon axes")
        if self.true_success is not None and self.true_success.shape != (groups, candidates):
            raise ValueError("success labels must match group and candidate axes")
        if (self.true_min_contact_distance is not None
                and self.true_min_contact_distance.shape != (
                    groups, candidates, self.actions.shape[2]
                )):
            raise ValueError("distance labels must match group, candidate and horizon axes")


def subset_candidate_batch(
    batch: PairedCandidateBatch,
    indices: torch.Tensor | np.ndarray | list[int] | slice,
) -> PairedCandidateBatch:
    """Select whole candidate groups without breaking within-group pairing."""
    selected = PairedCandidateBatch(
        initial_state=batch.initial_state[indices],
        actions=batch.actions[indices],
        true_terminal_object=batch.true_terminal_object[indices],
        goal=batch.goal[indices],
        lock_mask=batch.lock_mask[indices],
        lock_angle=batch.lock_angle[indices],
        true_contact=(None if batch.true_contact is None else batch.true_contact[indices]),
        true_success=(None if batch.true_success is None else batch.true_success[indices]),
        true_min_contact_distance=(
            None if batch.true_min_contact_distance is None
            else batch.true_min_contact_distance[indices]
        ),
    )
    selected.validate()
    return selected


def load_sequence_candidate_npz(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    allowed_locked_joints: tuple[int, ...] = (1, 3),
    split: str = "train",
    validation_fraction: float = 0.2,
    split_seed: int = 72,
    segment_repeat: int = 10,
    max_groups: int | None = None,
) -> PairedCandidateBatch:
    """Load strict paired sequence data and exclude held-out locks by construction."""
    if split not in {"train", "validation", "all"}:
        raise ValueError("split must be train, validation or all")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if segment_repeat < 1:
        raise ValueError("segment_repeat must be positive")
    data = np.load(Path(path))
    required = {"initial_state", "action_sequence", "locked_joint", "segment_states",
                "group", "goal", "episode"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"candidate archive is missing fields: {sorted(missing)}")
    lock = data["locked_joint"].astype(np.int64)
    episode = data["episode"].astype(np.int64)
    permitted = np.isin(lock, allowed_locked_joints)
    episode_key = lock * 100_000 + episode
    keys = np.unique(episode_key[permitted])
    rng = np.random.default_rng(split_seed)
    rng.shuffle(keys)
    validation_count = max(1, int(round(len(keys) * validation_fraction)))
    selected_keys = (
        keys if split == "all"
        else keys[:validation_count] if split == "validation"
        else keys[validation_count:]
    )
    rows = permitted & np.isin(episode_key, selected_keys)
    groups = np.unique(data["group"][rows])
    if max_groups is not None:
        if max_groups < 1:
            raise ValueError("max_groups must be positive")
        groups = groups[:max_groups]
    if not len(groups):
        raise ValueError(f"no candidate groups remain for split={split}")
    indices = [np.where(rows & (data["group"] == group))[0] for group in groups]
    counts = {len(item) for item in indices}
    if len(counts) != 1:
        raise ValueError("every candidate group must contain the same number of actions")
    grouped = np.stack(indices)
    initial = data["initial_state"][grouped]
    goals = data["goal"][grouped]
    locks = lock[grouped]
    if not np.allclose(initial, initial[:, :1], atol=0.0, rtol=0.0):
        raise ValueError("candidate group contains mismatched initial states")
    if not np.allclose(goals, goals[:, :1], atol=0.0, rtol=0.0):
        raise ValueError("candidate group contains mismatched goals")
    if not np.all(locks == locks[:, :1]):
        raise ValueError("candidate group contains mismatched diagnoses")
    segment_actions = data["action_sequence"][grouped].astype(np.float32)
    actions = np.repeat(segment_actions, segment_repeat, axis=2)
    group_lock = locks[:, 0]
    mask = np.zeros((len(groups), 5), dtype=np.float32)
    mask[np.arange(len(groups)), group_lock] = 1.0
    angle = np.zeros_like(mask)
    angle[np.arange(len(groups)), group_lock] = initial[:, 0, :5][
        np.arange(len(groups)), group_lock
    ]
    batch = PairedCandidateBatch(
        initial_state=torch.as_tensor(initial[:, 0], dtype=torch.float32, device=device),
        actions=torch.as_tensor(actions, dtype=torch.float32, device=device),
        true_terminal_object=torch.as_tensor(
            data["segment_states"][grouped, -1, 10:12], dtype=torch.float32, device=device
        ),
        goal=torch.as_tensor(goals[:, 0, :2], dtype=torch.float32, device=device),
        lock_mask=torch.as_tensor(mask, dtype=torch.float32, device=device),
        lock_angle=torch.as_tensor(angle, dtype=torch.float32, device=device),
        true_contact=(
            torch.as_tensor(
                np.repeat(data["contact_by_segment"][grouped].astype(bool),
                          segment_repeat, axis=2),
                dtype=torch.bool,
                device=device,
            )
            if "contact_by_segment" in data.files else None
        ),
        true_success=(
            torch.as_tensor(data["success"][grouped].astype(bool),
                            dtype=torch.bool, device=device)
            if "success" in data.files else None
        ),
        true_min_contact_distance=(
            torch.as_tensor(
                np.repeat(
                    data["minimum_contact_distance_by_segment"][grouped].astype(np.float32),
                    segment_repeat,
                    axis=2,
                ),
                dtype=torch.float32,
                device=device,
            )
            if "minimum_contact_distance_by_segment" in data.files else None
        ),
    )
    batch.validate()
    if set(group_lock.tolist()) - set(allowed_locked_joints):
        raise AssertionError("held-out lock leaked into the paired candidate batch")
    return batch


def terminal_costs(states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
    """Euclidean object-to-goal cost for `[batch, candidates, object_dim]` states."""
    if states.ndim != 3:
        raise ValueError("states must have shape [batch, candidates, object_dim]")
    if goals.ndim != 2 or goals.shape[0] != states.shape[0]:
        raise ValueError("goals must have shape [batch, object_dim]")
    if states.shape[-1] != goals.shape[-1]:
        raise ValueError("state and goal object dimensions must match")
    return torch.linalg.vector_norm(states - goals[:, None, :], dim=-1)


def paired_soft_regret_loss(
    predicted_terminal_object: torch.Tensor,
    true_terminal_object: torch.Tensor,
    goals: torch.Tensor,
    *,
    temperature: float = 0.02,
) -> torch.Tensor:
    """Expected true regret under the distribution induced by predicted costs.

    Every row must contain candidates rolled out from the same initial state and
    diagnosis. Ground-truth outcomes determine regret only during training; the
    deployed world model still selects candidates from its predicted state.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if predicted_terminal_object.shape != true_terminal_object.shape:
        raise ValueError("predicted and true terminal objects must have identical shape")
    predicted_cost = terminal_costs(predicted_terminal_object, goals)
    true_cost = terminal_costs(true_terminal_object, goals)
    probability = torch.softmax(-predicted_cost / temperature, dim=1)
    regret = true_cost - true_cost.min(dim=1, keepdim=True).values
    return (probability * regret).sum(dim=1).mean()


def rollout_candidate_terminal_object(
    model,
    batch: PairedCandidateBatch,
    *,
    object_cost_slice: slice = slice(10, 12),
) -> torch.Tensor:
    """Roll a state-predicting model over every paired candidate sequence."""
    batch.validate()
    groups, candidates, horizon, _ = batch.actions.shape
    state = batch.initial_state[:, None, :].expand(-1, candidates, -1).reshape(
        groups * candidates, -1
    )
    actions = batch.actions.reshape(groups * candidates, horizon, -1)
    mask = batch.lock_mask[:, None, :].expand(-1, candidates, -1).reshape(
        groups * candidates, -1
    )
    angle = batch.lock_angle[:, None, :].expand(-1, candidates, -1).reshape(
        groups * candidates, -1
    )
    hidden = None
    for step in range(horizon):
        state, hidden = model.step(state, actions[:, step], mask, angle, hidden)
    return state[:, object_cost_slice].reshape(groups, candidates, -1)


def rollout_candidate_diagnostics(
    model,
    batch: PairedCandidateBatch,
    *,
    contact_threshold: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Roll candidates and retain the minimum state needed by the six-stage audit.

    ``predicted_contact_reachable`` is a kinematic contact-manifold proxy computed
    from predicted states, not a simulator contact label. Physical contact is
    reported separately from ``true_contact`` when the archive provides it.
    """
    batch.validate()
    groups, candidates, horizon, _ = batch.actions.shape
    state = batch.initial_state[:, None, :].expand(-1, candidates, -1).reshape(
        groups * candidates, -1
    )
    actions = batch.actions.reshape(groups * candidates, horizon, -1)
    mask = batch.lock_mask[:, None, :].expand(-1, candidates, -1).reshape(
        groups * candidates, -1
    )
    angle = batch.lock_angle[:, None, :].expand(-1, candidates, -1).reshape(
        groups * candidates, -1
    )
    hidden = None
    maximum_position_violation = state.new_zeros(groups * candidates)
    maximum_velocity_violation = state.new_zeros(groups * candidates)
    maximum_contact_gate = state.new_zeros(groups * candidates)
    for step in range(horizon):
        state, hidden = model.step(state, actions[:, step], mask, angle, hidden)
        position_violation = ((state[:, :5] - angle).abs() * mask).amax(dim=1)
        velocity_violation = (state[:, 5:10].abs() * mask).amax(dim=1)
        maximum_position_violation = torch.maximum(
            maximum_position_violation, position_violation
        )
        maximum_velocity_violation = torch.maximum(
            maximum_velocity_violation, velocity_violation
        )
        contact_gate = pusher_box_contact_gate(state[:, :5], state[:, 10:12])
        maximum_contact_gate = torch.maximum(maximum_contact_gate, contact_gate)
    shape = (groups, candidates)
    return state[:, 10:12].reshape(groups, candidates, 2), {
        "maximum_locked_position_violation": maximum_position_violation.reshape(shape),
        "maximum_locked_velocity_violation": maximum_velocity_violation.reshape(shape),
        "maximum_predicted_contact_gate": maximum_contact_gate.reshape(shape),
        "predicted_contact_reachable": (
            maximum_contact_gate.reshape(shape) >= contact_threshold
        ),
    }


def world_model_paired_regret_loss(
    model,
    batch: PairedCandidateBatch,
    *,
    temperature: float = 0.02,
    object_cost_slice: slice = slice(10, 12),
) -> torch.Tensor:
    """Decision loss computed directly from the model's predicted terminal state."""
    predicted = rollout_candidate_terminal_object(
        model, batch, object_cost_slice=object_cost_slice
    )
    return paired_soft_regret_loss(
        predicted, batch.true_terminal_object, batch.goal, temperature=temperature
    )


def world_model_paired_regret_loss_batched(
    model,
    batch: PairedCandidateBatch,
    *,
    temperature: float = 0.02,
    object_cost_slice: slice = slice(10, 12),
    group_batch_size: int = 8,
) -> torch.Tensor:
    """Memory-bounded exact mean loss for validation or no-grad evaluation."""
    if group_batch_size < 1:
        raise ValueError("group_batch_size must be positive")
    total = batch.actions.new_zeros(())
    groups = int(batch.actions.shape[0])
    for start in range(0, groups, group_batch_size):
        chunk = subset_candidate_batch(batch, slice(start, min(start + group_batch_size, groups)))
        total = total + len(chunk.actions) * world_model_paired_regret_loss(
            model, chunk, temperature=temperature, object_cost_slice=object_cost_slice
        )
    return total / groups


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for group, count in enumerate(counts):
        if count > 1:
            indices = np.where(inverse == group)[0]
            ranks[indices] = ranks[indices].mean()
    return ranks


def spearman_correlation(predicted_cost: np.ndarray, true_cost: np.ndarray) -> float:
    """Spearman correlation with average ranks for ties."""
    predicted = np.asarray(predicted_cost, dtype=np.float64)
    truth = np.asarray(true_cost, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 1:
        raise ValueError("cost arrays must be one-dimensional and shape matched")
    if len(predicted) < 2:
        return float("nan")
    a, b = _rank(predicted), _rank(truth)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def kendall_correlation(predicted_cost: np.ndarray, true_cost: np.ndarray) -> float:
    """Kendall tau-a for candidate ordering; ties contribute zero."""
    predicted = np.asarray(predicted_cost, dtype=np.float64)
    truth = np.asarray(true_cost, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 1:
        raise ValueError("cost arrays must be one-dimensional and shape matched")
    concordant = discordant = 0
    for i in range(len(predicted)):
        for j in range(i + 1, len(predicted)):
            sign = np.sign(predicted[i] - predicted[j]) * np.sign(truth[i] - truth[j])
            concordant += int(sign > 0)
            discordant += int(sign < 0)
    pairs = len(predicted) * (len(predicted) - 1) // 2
    return float((concordant - discordant) / pairs) if pairs else float("nan")


def candidate_metrics(predicted_cost: np.ndarray, true_cost: np.ndarray) -> dict[str, float]:
    """Return ranking correlation, top-1 regret and selected true cost."""
    predicted = np.asarray(predicted_cost, dtype=np.float64)
    truth = np.asarray(true_cost, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 1 or not len(predicted):
        raise ValueError("cost arrays must be non-empty, one-dimensional and shape matched")
    choice = int(np.argmin(predicted))
    return {
        "spearman": spearman_correlation(predicted, truth),
        "kendall": kendall_correlation(predicted, truth),
        "top1_regret": float(truth[choice] - truth.min()),
        "selected_true_cost": float(truth[choice]),
    }


def world_model_candidate_metrics(
    model,
    batch: PairedCandidateBatch,
    *,
    object_cost_slice: slice = slice(10, 12),
    group_batch_size: int = 8,
) -> dict[str, float | int]:
    """Aggregate decision metrics over strictly matched candidate groups."""
    if group_batch_size < 1:
        raise ValueError("group_batch_size must be positive")
    started = time.perf_counter()
    predicted_chunks, true_chunks = [], []
    with torch.no_grad():
        for start in range(0, batch.actions.shape[0], group_batch_size):
            stop = min(start + group_batch_size, batch.actions.shape[0])
            chunk = subset_candidate_batch(batch, slice(start, stop))
            prediction = rollout_candidate_terminal_object(
                model, chunk, object_cost_slice=object_cost_slice
            )
            predicted_chunks.append(terminal_costs(prediction, chunk.goal).cpu().numpy())
            true_chunks.append(
                terminal_costs(chunk.true_terminal_object, chunk.goal).cpu().numpy()
            )
    predicted_cost = np.concatenate(predicted_chunks, axis=0)
    true_cost = np.concatenate(true_chunks, axis=0)
    rows = [candidate_metrics(predicted_cost[i], true_cost[i])
            for i in range(predicted_cost.shape[0])]
    return {
        "groups": len(rows),
        "candidates_per_group": int(predicted_cost.shape[1]),
        "spearman": float(np.mean([row["spearman"] for row in rows])),
        "kendall": float(np.mean([row["kendall"] for row in rows])),
        "top1_regret": float(np.mean([row["top1_regret"] for row in rows])),
        "selected_true_cost": float(np.mean([row["selected_true_cost"] for row in rows])),
        "oracle_true_cost": float(np.mean(true_cost.min(axis=1))),
        "evaluation_wall_time_seconds": float(time.perf_counter() - started),
    }


def world_model_six_stage_metrics(
    model,
    batch: PairedCandidateBatch,
    *,
    group_batch_size: int = 8,
    contact_threshold: float = 0.5,
    near_contact_distance: float = 0.01,
) -> dict[str, object]:
    """Return one same-protocol diagnostic chain from feasibility to outcome.

    The selected candidate is always chosen from predicted terminal cost. Hence
    every downstream stage evaluates the same choice rather than splicing metrics
    from unrelated runs. Missing simulator contact/success labels fail closed to
    ``available: false`` rather than being inferred from prediction.
    """
    if group_batch_size < 1:
        raise ValueError("group_batch_size must be positive")
    started = time.perf_counter()
    rows: list[dict[str, float | bool]] = []
    contact_response_errors: list[float] = []
    for start in range(0, batch.actions.shape[0], group_batch_size):
        stop = min(start + group_batch_size, batch.actions.shape[0])
        chunk = subset_candidate_batch(batch, slice(start, stop))
        with torch.no_grad():
            predicted_object, diagnostic = rollout_candidate_diagnostics(
                model, chunk, contact_threshold=contact_threshold
            )
            predicted_cost = terminal_costs(predicted_object, chunk.goal)
            true_cost = terminal_costs(chunk.true_terminal_object, chunk.goal)
        selected = predicted_cost.argmin(dim=1)
        for local in range(len(selected)):
            choice = int(selected[local])
            ranking = candidate_metrics(
                predicted_cost[local].cpu().numpy(), true_cost[local].cpu().numpy()
            )
            actual_contact = None
            if chunk.true_contact is not None:
                contact_candidates = chunk.true_contact[local].any(dim=1)
                actual_contact = bool(contact_candidates[choice])
                if bool(contact_candidates.any()):
                    errors = torch.linalg.vector_norm(
                        predicted_object[local] - chunk.true_terminal_object[local], dim=1
                    )
                    contact_response_errors.extend(
                        errors[contact_candidates].cpu().tolist()
                    )
            actual_success = (
                None if chunk.true_success is None
                else bool(chunk.true_success[local, choice])
            )
            realized_min_distance = (
                None if chunk.true_min_contact_distance is None
                else float(chunk.true_min_contact_distance[local, choice].amin())
            )
            rows.append({
                "locked_position_violation": float(
                    diagnostic["maximum_locked_position_violation"][local, choice]
                ),
                "locked_velocity_violation": float(
                    diagnostic["maximum_locked_velocity_violation"][local, choice]
                ),
                "predicted_contact_reachable": bool(
                    diagnostic["predicted_contact_reachable"][local, choice]
                ),
                "actual_contact": actual_contact,
                "actual_success": actual_success,
                "realized_min_contact_distance": realized_min_distance,
                "endpoint_error": float(true_cost[local, choice]),
                "spearman": ranking["spearman"],
                "kendall": ranking["kendall"],
                "top1_regret": ranking["top1_regret"],
            })
    contact_available = any(row["actual_contact"] is not None for row in rows)
    success_available = any(row["actual_success"] is not None for row in rows)
    distance_available = any(
        row["realized_min_contact_distance"] is not None for row in rows
    )
    return {
        "groups": len(rows),
        "selection_rule": "minimum predicted terminal object-to-goal distance",
        "constraint": {
            "locked_position_violation_max": max(
                float(row["locked_position_violation"]) for row in rows
            ),
            "locked_velocity_violation_max": max(
                float(row["locked_velocity_violation"]) for row in rows
            ),
        },
        "reachability": {
            "definition": (
                "predicted analytic contact gate plus realized continuous-time minimum "
                "MuJoCo geom distance when available"
            ),
            "contact_gate_threshold": contact_threshold,
            "predicted_proxy_selected_candidate_rate": float(np.mean([
                row["predicted_contact_reachable"] for row in rows
            ])),
            "realized_distance_available": distance_available,
            "near_contact_distance_threshold": near_contact_distance,
            "realized_min_contact_distance_mean": (
                float(np.mean([row["realized_min_contact_distance"] for row in rows]))
                if distance_available else None
            ),
            "realized_near_contact_rate": (
                float(np.mean([
                    row["realized_min_contact_distance"] <= near_contact_distance
                    for row in rows
                ])) if distance_available else None
            ),
        },
        "contact": {
            "available": contact_available,
            "selected_candidate_rate": (
                float(np.mean([row["actual_contact"] for row in rows]))
                if contact_available else None
            ),
        },
        "response": {
            "available": bool(contact_response_errors),
            "contact_candidate_terminal_object_rmse": (
                float(np.sqrt(np.mean(np.square(contact_response_errors))))
                if contact_response_errors else None
            ),
            "contact_candidate_count": len(contact_response_errors),
        },
        "action_ranking": {
            "spearman": float(np.mean([row["spearman"] for row in rows])),
            "kendall": float(np.mean([row["kendall"] for row in rows])),
            "top1_regret": float(np.mean([row["top1_regret"] for row in rows])),
        },
        "closed_loop_outcome": {
            "label": "realized open-loop candidate outcome; not receding-horizon MPC",
            "endpoint_error": float(np.mean([row["endpoint_error"] for row in rows])),
            "success_available": success_available,
            "success_rate": (
                float(np.mean([row["actual_success"] for row in rows]))
                if success_available else None
            ),
        },
        "evaluation_wall_time_seconds": float(time.perf_counter() - started),
    }
