"""Gate K1: geometry-preserving fixed-transform graph on free-arm dynamics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.models.fixed_transform_graph import (
    FixedTransformGraphConfig,
    FixedTransformContactWorldModel,
    FixedTransformGraphObjectWorldModel,
    FixedTransformGraphWorldModel,
)
from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.training.sim_protocol import damage_from_name, load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors, evaluate_graph_surgery_model
from scripts.run_push_benchmark import collect_push_domains


def _batch(trajectories, device):
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    damages = [damage_from_name(item.domain_id.split("__", 1)[0]) for item in trajectories]
    mask, angle = _damage_tensors(damages, device)
    return states, actions, mask, angle


def _train(
    model, batch, *, epochs: int, learning_rate: float,
    use_topology: bool, include_object_loss: bool, object_loss_weight: float,
):
    states, actions, true_mask, true_angle = batch
    mask = true_mask if use_topology else torch.zeros_like(true_mask)
    angle = true_angle if use_topology else torch.zeros_like(true_angle)
    joint_mask = (
        torch.cat((1.0 - true_mask, 1.0 - true_mask), dim=-1)
        if use_topology else torch.ones_like(states[:, 0, :10])
    )
    joint_count = joint_mask.sum(dim=-1).clamp_min(1.0)

    def transition_loss(prediction, target):
        joint_error = (prediction[:, :10] - target[:, :10]).pow(2)
        result = ((joint_error * joint_mask).sum(dim=-1) / joint_count).mean()
        if include_object_loss:
            result = result + object_loss_weight * (
                prediction[:, 10:] - target[:, 10:]
            ).pow(2).mean()
        return result
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(epochs):
        hidden, losses = None, []
        for step in range(actions.shape[1]):
            prediction, hidden = model.step(states[:, step], actions[:, step], mask, angle, hidden)
            losses.append(transition_loss(prediction, states[:, step + 1]))
        rollout = []
        horizon = min(5, actions.shape[1])
        for start in range(0, actions.shape[1] - horizon + 1, horizon):
            prediction, hidden = states[:, start], None
            for offset in range(horizon):
                prediction, hidden = model.step(
                    prediction, actions[:, start + offset], mask, angle, hidden
                )
                rollout.append(
                    transition_loss(prediction, states[:, start + offset + 1])
                )
        loss = torch.stack(losses).mean() + 0.5 * torch.stack(rollout).mean()
        optimizer.zero_grad(); loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": float(loss.detach()), "gradient_norm": grad})
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"  epoch={epoch + 1:02d} loss={float(loss.detach()):.6f} grad={grad:.3f}",
                flush=True,
            )
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="Smoke-test override; formal runs use config")
    parser.add_argument("--steps", type=int, help="Smoke-test override; formal runs use config")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError("seed not in frozen list")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    block_xy = np.asarray(config["block_initial_xy"], dtype=float)
    epochs = args.epochs if args.epochs is not None else int(config["epochs"])
    steps = args.steps if args.steps is not None else int(config["steps"])
    if epochs < 1 or steps < 2:
        raise ValueError("epochs must be >= 1 and steps must be >= 2")
    common = dict(
        steps=steps, excitation="goal", block_initial_xy=block_xy,
        goal_exploration_std=float(config["goal_exploration_std"]),
    )
    print(f"device={device} seed={args.seed}", flush=True)
    train = collect_push_domains(
        protocol.train, trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        seed=args.seed * 10_000, targets=calibration, **common,
    )
    batch = _batch(train, device)
    hidden_dim = int(config["hidden_dim"])
    include_object_loss = bool(config.get("include_object_loss", False))
    model_type = config.get("ft_model_type", "isolated_object" if include_object_loss else "joint_only")
    ft_model_classes = {
        "joint_only": FixedTransformGraphWorldModel,
        "isolated_object": FixedTransformGraphObjectWorldModel,
        "fixed_transform_contact": FixedTransformContactWorldModel,
    }
    try:
        ft_model_class = ft_model_classes[model_type]
    except KeyError:
        raise ValueError(f"unknown ft_model_type: {model_type}") from None
    models = {
        "matched_graph": TopologyGraphWorldModel(TopologyGraphConfig(hidden_dim=hidden_dim)).to(device),
        "ft_gwm": ft_model_class(
            FixedTransformGraphConfig(hidden_dim=hidden_dim)
        ).to(device),
    }
    histories = {}
    for name, model in models.items():
        print(f"[train] {name}", flush=True)
        torch.manual_seed(args.seed)
        histories[name] = _train(
            model, batch, epochs=epochs,
            learning_rate=float(config["learning_rate"]), use_topology=name == "ft_gwm",
            include_object_loss=include_object_loss,
            object_loss_weight=float(config.get("object_loss_weight", 1.0)),
        )

    rows = []
    for index, domain in enumerate(protocol.test):
        trajectories = collect_push_domains(
            (domain,), trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            seed=args.seed * 100_000 + index * 1000 + 500, targets=evaluation, **common,
        )
        variants = (
            ("matched_graph", models["matched_graph"], False),
            ("matched_graph_projected", models["matched_graph"], True),
            ("ft_gwm", models["ft_gwm"], True),
        )
        for name, model, use_topology in variants:
            metrics = evaluate_graph_surgery_model(
                model, domain, trajectories, device=device,
                horizon=int(config["rollout_horizon"]), use_topology=use_topology,
            )
            rows.append({"domain": domain.domain_id, "method": name, **metrics})
            print(f"{domain.domain_id} {name}: free={metrics['free_arm_rmse']:.4f} violation={metrics['constraint_violation_rms']:.6f}", flush=True)

    primary = config["primary_domain"]
    selected = {row["method"]: row for row in rows if row["domain"] == primary}
    base, ft = selected["matched_graph"], selected["ft_gwm"]
    regression = 100.0 * (ft["free_arm_rmse"] - base["free_arm_rmse"]) / base["free_arm_rmse"]
    object_regression = 100.0 * (ft["object_rmse"] - base["object_rmse"]) / base["object_rmse"]
    passed = (
        ft["constraint_violation_rms"] <= float(config["gate"]["maximum_constraint_violation_rms"])
        and regression <= float(config["gate"]["maximum_free_arm_rmse_regression_pct"])
        and object_regression <= float(config["gate"].get("maximum_object_rmse_regression_pct", float("inf")))
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {
        "seed": args.seed, "epochs": epochs, "steps": steps,
        "ft_model_type": model_type,
        "free_arm_regression_pct": regression,
        "object_regression_pct": object_regression, "gate_passed": passed,
        "parameters": {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
        "histories": histories, "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[gate] free_regression={regression:+.2f}% "
        f"object_regression={object_regression:+.2f}% "
        f"decision={'PASS' if passed else 'NO-GO'}", flush=True,
    )


if __name__ == "__main__":
    main()
