"""Gate N0: oracle audit for a set-valued two-candidate impulse operator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.models.hybrid_contact_impulse import HybridContactConfig, HybridContactImpulseModel
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from scripts.run_push_benchmark import collect_push_domains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    gate_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_config = yaml.safe_load(Path(gate_config["data_config"]).read_text(encoding="utf-8"))
    seed = int(gate_config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    protocol = load_g1_protocol(Path(data_config["protocol"]))
    targets = load_target_split(Path(data_config["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    steps = int(args.steps or data_config["steps"])
    trajectories = collect_push_domains(
        protocol.train,
        trajectories_per_domain=int(data_config["trajectories_per_train_domain"]),
        seed=seed * 10_000,
        targets=calibration,
        steps=steps,
        excitation="goal",
        block_initial_xy=np.asarray(data_config["block_initial_xy"], dtype=float),
        goal_exploration_std=float(data_config["goal_exploration_std"]),
    )
    states, next_q, aggregate, source_activity = [], [], [], []
    exact_normal_components = []
    record_counts = []
    for trajectory in trajectories:
        if trajectory.contact_records is None or trajectory.contact_impulses is None:
            raise ValueError("Gate N0 requires contact records and impulses")
        for step, records in enumerate(trajectory.contact_records):
            states.append(trajectory.states[step])
            next_q.append(trajectory.states[step + 1, :5])
            aggregate.append(trajectory.contact_impulses[step])
            active = [False, False]
            for record in records:
                source = str(record["source_geom"])
                active[0 if source == "tool_geom" else 1] = True
                normal = np.asarray(record["normal_xy"], dtype=float)
                impulse = np.asarray(record["impulse_xy"], dtype=float)
                norm = np.linalg.norm(normal)
                if norm > 1e-8 and np.linalg.norm(impulse) > 1e-10:
                    exact_normal_components.append(float(np.dot(impulse, normal / norm)))
            source_activity.append(active)
            record_counts.append(len(records))
    states_t = torch.stack(states)
    next_q_t = torch.stack(next_q)
    aggregate_t = torch.stack(aggregate)
    activity_t = torch.tensor(source_activity, dtype=torch.float32)
    contact = activity_t.any(dim=1)
    model = HybridContactImpulseModel(HybridContactConfig(
        friction_coefficient=float(gate_config["friction_coefficient"])
    ))
    with torch.no_grad():
        geometry = gate_config.get("candidate_geometry", "bounding_circle")
        if geometry == "bounding_circle":
            _, normal, tangent = model.candidate_contact_frames(states_t, next_q_t)
        elif geometry == "axis_aligned_box":
            _, normal, tangent = model.candidate_box_contact_frames(states_t, next_q_t)
        else:
            raise ValueError(f"unknown candidate_geometry: {geometry}")
    normal = normal[contact]
    tangent = tangent[contact]
    activity = activity_t[contact]
    target = aggregate_t[contact]
    # Contact impulses are O(1e-3) N*s. softplus(0)=0.693 would initialize the
    # oracle projection hundreds of times above the target and make the finite
    # optimization budget diagnose solver scale instead of representability.
    raw_normal = torch.full((target.shape[0], 2), -8.0, requires_grad=True)
    raw_tangent = torch.zeros(target.shape[0], 2, requires_grad=True)
    optimizer = torch.optim.Adam(
        (raw_normal, raw_tangent),
        lr=float(gate_config["projection_learning_rate"]),
    )
    mu = float(gate_config["friction_coefficient"])
    for _ in range(int(gate_config["projection_steps"])):
        normal_impulse = torch.nn.functional.softplus(raw_normal) * activity
        tangent_impulse = mu * normal_impulse * torch.tanh(raw_tangent)
        prediction = (
            normal_impulse[..., None] * normal
            + tangent_impulse[..., None] * tangent
        ).sum(dim=1)
        loss = (prediction - target).pow(2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    projection_rmse = float((prediction.detach() - target).pow(2).mean().sqrt())
    target_rms = float(target.pow(2).mean().sqrt())
    relative_rmse = projection_rmse / max(target_rms, 1e-12)
    exact = np.asarray(exact_normal_components, dtype=float)
    negative_fraction = float(np.mean(exact < -1e-10)) if exact.size else 1.0
    contact_counts = np.asarray(record_counts)[np.asarray(record_counts) > 0]
    multicontact_fraction = float(np.mean(contact_counts > 1)) if contact_counts.size else 0.0
    gate = gate_config["gate"]
    passed = (
        negative_fraction <= float(gate["maximum_exact_record_negative_normal_fraction"])
        and relative_rmse <= float(gate["maximum_candidate_projection_relative_rmse"])
        and steps == int(data_config["steps"])
    )
    summary = {
        "seed": seed,
        "candidate_geometry": geometry,
        "steps": steps,
        "contact_transitions": int(contact.sum()),
        "exact_contact_records": int(exact.size),
        "multicontact_transition_fraction": multicontact_fraction,
        "exact_record_negative_normal_fraction": negative_fraction,
        "candidate_projection_rmse": projection_rmse,
        "measured_impulse_rms": target_rms,
        "candidate_projection_relative_rmse": relative_rmse,
        "gate_passed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[Gate N0] {'PASS' if passed else ('NO-GO' if steps == int(data_config['steps']) else 'SMOKE')}")


if __name__ == "__main__":
    main()
