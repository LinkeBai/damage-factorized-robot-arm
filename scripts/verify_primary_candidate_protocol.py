"""Fail-closed audit for the frozen primary-arm candidate protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


JOINT_INDEX = {f"D{index + 1}": index for index in range(5)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expanded_actions(actions: np.ndarray, steps_per_segment: int) -> np.ndarray:
    if actions.ndim == 3:
        return actions
    if actions.ndim != 4:
        raise ValueError("action_sequence must have shape [row,time,dof] or [row,segment,dof]")
    return np.repeat(actions, steps_per_segment, axis=2) if actions.shape[2] == 1 else actions


def audit_candidate_file(
    path: Path,
    *,
    allowed_locks: set[int],
    expected_candidates: int | None,
    horizon_steps: int,
    steps_per_segment: int,
) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {"group", "locked_joint", "action_sequence"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"missing candidate fields: {missing}")
        groups = np.asarray(data["group"])
        locks = np.asarray(data["locked_joint"])
        actions = np.asarray(data["action_sequence"])
    if groups.ndim != 1 or locks.shape != groups.shape or actions.shape[0] != groups.shape[0]:
        raise ValueError("candidate row dimensions do not agree")
    selected = np.isin(locks, sorted(allowed_locks))
    if not np.any(selected):
        raise ValueError("file contains no rows for the permitted intervention split")
    selected_groups, selected_locks, selected_actions = groups[selected], locks[selected], actions[selected]
    group_ids = np.unique(selected_groups)
    counts = np.asarray([(selected_groups == group).sum() for group in group_ids])
    if np.any(counts != counts[0]):
        raise ValueError("candidate groups have unequal sizes")
    candidates = int(counts[0])
    if expected_candidates is not None and candidates != expected_candidates:
        raise ValueError(
            f"expected {expected_candidates} candidates per group, found {candidates}; "
            "candidate replication is forbidden"
        )
    if selected_actions.ndim != 3:
        raise ValueError("action_sequence must be [row,segment,dof]")
    expanded_horizon = int(selected_actions.shape[1] * steps_per_segment)
    if expanded_horizon != horizon_steps:
        raise ValueError(f"expected {horizon_steps} action steps, found {expanded_horizon}")
    duplicate_groups = 0
    for group in group_ids:
        flat = np.ascontiguousarray(selected_actions[selected_groups == group]).reshape(candidates, -1)
        if np.unique(flat, axis=0).shape[0] != candidates:
            duplicate_groups += 1
    if duplicate_groups:
        raise ValueError(f"{duplicate_groups} groups contain duplicated candidate sequences")
    unexpected_selected = sorted(set(int(value) for value in np.unique(selected_locks)) - allowed_locks)
    if unexpected_selected:
        raise ValueError(f"unexpected selected locks: {unexpected_selected}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "rows_total": int(groups.shape[0]),
        "rows_selected": int(selected.sum()),
        "rows_excluded_by_lock_filter": int((~selected).sum()),
        "selected_locks": sorted(int(value) for value in np.unique(selected_locks)),
        "groups": int(group_ids.size),
        "candidates_per_group": candidates,
        "action_segments": int(selected_actions.shape[1]),
        "expanded_horizon_steps": expanded_horizon,
        "duplicate_candidate_groups": duplicate_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=Path(
        "config/experiment/icra_2027_primary_5dof_recovery_v1.yaml"))
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--training-data",
        action="store_true",
        help="Audit paired training groups; candidate count need not equal the planning budget.",
    )
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    interventions = contract["interventions"]
    planning = contract["planning"]
    if args.training_data:
        if args.phase != "development":
            raise ValueError("confirmation data must never be opened as training data")
        allowed = {JOINT_INDEX[name] for name in interventions["train_locks"]}
        expected_candidates = None
        role = "development_training"
    elif args.phase == "development":
        allowed = {JOINT_INDEX[name] for name in interventions["train_locks"]}
        expected_candidates = int(planning["candidates"])
        role = "development_evaluation"
    else:
        allowed = {JOINT_INDEX[interventions["heldout_lock"]]}
        expected_candidates = int(planning["candidates"])
        role = "confirmation_evaluation"
    try:
        candidate = audit_candidate_file(
            args.candidate_file,
            allowed_locks=allowed,
            expected_candidates=expected_candidates,
            horizon_steps=int(planning["horizon_steps"]),
            steps_per_segment=int(planning["steps_per_segment"]),
        )
        status, error = "PASS", None
    except Exception as exc:
        status, error = "FAIL", str(exc)
        candidate = {
            "path": str(args.candidate_file.resolve()),
            "sha256": sha256(args.candidate_file) if args.candidate_file.is_file() else None,
        }
    payload = {
        "status": status,
        "phase": args.phase,
        "role": role,
        "contract": str(args.contract),
        "heldout_lock": interventions["heldout_lock"],
        "planning_candidates": int(planning["candidates"]),
        "candidate_duplication_allowed": bool(planning["candidate_duplication_allowed"]),
        "privileged_future_state_allowed": bool(planning["privileged_future_state_allowed"]),
        "candidate_file": candidate,
        "error": error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
