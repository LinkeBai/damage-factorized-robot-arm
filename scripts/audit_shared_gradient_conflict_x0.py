"""X0: audit joint/object gradient conflict in the frozen shared graph model."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from scripts.run_push_benchmark import collect_push_domains


def _task_losses(model, states, actions, start, horizon):
    prediction, hidden = states[:, start], None
    joint_losses, object_losses = [], []
    zeros = torch.zeros(states.shape[0], 5, device=states.device)
    for offset in range(horizon):
        prediction, hidden = model.step(
            prediction, actions[:, start + offset], zeros, zeros, hidden
        )
        target = states[:, start + offset + 1]
        joint_losses.append((prediction[:, :10] - target[:, :10]).pow(2).mean())
        object_losses.append((prediction[:, 10:] - target[:, 10:]).pow(2).mean())
    return torch.stack(joint_losses).mean(), torch.stack(object_losses).mean()


def _flatten_gradients(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return torch.cat([
        torch.zeros_like(parameter).flatten() if gradient is None else gradient.flatten()
        for parameter, gradient in zip(parameters, gradients)
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--v0-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed not in frozen X0 list")
    v0 = yaml.safe_load(Path(config["v0_config"]).read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(v0["q0a_config"]).read_text(encoding="utf-8"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    trajectories = collect_push_domains(
        protocol.train, trajectories_per_domain=int(q0a["trajectories_per_train_domain"]),
        steps=int(q0a["steps"]), seed=args.seed * 10_000, targets=calibration,
        excitation="goal", block_initial_xy=np.asarray(q0a["block_initial_xy"], dtype=float),
        goal_exploration_std=float(q0a["goal_exploration_std"]),
    )
    model = TopologyGraphWorldModel(TopologyGraphConfig(
        hidden_dim=int(v0["shared_compute_matched_hidden_dim"])
    )).to(device)
    payload = torch.load(args.v0_run_dir / "models.pt", map_location=device, weights_only=True)
    model.load_state_dict(payload["shared_compute_matched"])
    shared_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("joint_head") and not name.startswith("object_head")
    ]
    horizon = int(config["rollout_horizon"])
    candidates = []
    for trajectory_index, trajectory in enumerate(trajectories):
        for start in range(0, len(trajectory.actions) - horizon + 1, horizon):
            candidates.append((trajectory_index, start))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(candidates)
    candidates = candidates[: int(config["maximum_windows"])]
    records = []
    for trajectory_index, start in candidates:
        trajectory = trajectories[trajectory_index]
        states = trajectory.states.unsqueeze(0).to(device)
        actions = trajectory.actions.unsqueeze(0).to(device)
        joint_loss, object_loss = _task_losses(model, states, actions, start, horizon)
        joint_gradient = _flatten_gradients(joint_loss, shared_parameters, True)
        object_gradient = _flatten_gradients(object_loss, shared_parameters, False)
        joint_norm = torch.linalg.vector_norm(joint_gradient)
        object_norm = torch.linalg.vector_norm(object_gradient)
        cosine = torch.dot(joint_gradient, object_gradient) / (
            joint_norm * object_norm
        ).clamp_min(1e-12)
        records.append({
            "domain": trajectory.domain_id,
            "start": start,
            "cosine": float(cosine),
            "joint_gradient_norm": float(joint_norm),
            "object_gradient_norm": float(object_norm),
            "joint_loss": float(joint_loss),
            "object_loss": float(object_loss),
        })
    cosines = np.asarray([item["cosine"] for item in records])
    grouped = defaultdict(list)
    for item in records:
        grouped[item["domain"]].append(item["cosine"])
    gate = config["conflict_gate"]
    negative_fraction = float(np.mean(cosines < 0))
    mean_cosine = float(np.mean(cosines))
    pcgrad_indicated = (
        negative_fraction >= float(gate["minimum_negative_cosine_fraction_for_pcgrad"])
        and mean_cosine <= float(gate["maximum_mean_cosine_for_pcgrad"])
    )
    summary = {
        "config_version": config["version"], "seed": args.seed,
        "n_windows": len(records), "mean_cosine": mean_cosine,
        "median_cosine": float(np.median(cosines)),
        "negative_cosine_fraction": negative_fraction,
        "pcgrad_indicated": pcgrad_indicated,
        "mean_joint_gradient_norm": float(np.mean([item["joint_gradient_norm"] for item in records])),
        "mean_object_gradient_norm": float(np.mean([item["object_gradient_norm"] for item in records])),
        "by_domain": {
            domain: {
                "n": len(values), "mean_cosine": float(np.mean(values)),
                "negative_fraction": float(np.mean(np.asarray(values) < 0)),
            }
            for domain, values in sorted(grouped.items())
        },
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[X0] mean_cos={mean_cosine:+.3f} median={summary['median_cosine']:+.3f} "
        f"negative={negative_fraction:.1%} pcgrad={'YES' if pcgrad_indicated else 'NO'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
