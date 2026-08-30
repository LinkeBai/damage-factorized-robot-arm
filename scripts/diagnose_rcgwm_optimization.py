"""Diagnose RC-GWM checkpoint instability and multi-task gradient conflict."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.models.reduced_coordinate_graph import ReducedCoordinateGraphWorldModel
from robotarm.models.topology_graph_world_model import TopologyGraphConfig
from robotarm.training.sim_protocol import damage_from_name, load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import (
    _damage_tensors,
    evaluate_graph_surgery_model,
)
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


def _batch(trajectories: list, device: torch.device):
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    damages = [damage_from_name(item.domain_id.split("__", 1)[0]) for item in trajectories]
    mask, angle = _damage_tensors(damages, device)
    return states, actions, mask, angle


def _loss_parts(model, batch, *, with_rollout: bool) -> tuple[torch.Tensor, torch.Tensor]:
    states, actions, mask, angle = batch
    free_mask = torch.cat((1.0 - mask, 1.0 - mask), dim=-1)
    free_count = free_mask.sum(dim=-1).clamp_min(1.0)
    free_losses, object_losses = [], []
    hidden = None
    for step in range(actions.shape[1]):
        prediction, hidden = model.step(states[:, step], actions[:, step], mask, angle, hidden)
        error = (prediction - states[:, step + 1]).pow(2)
        free_losses.append(((error[:, :10] * free_mask).sum(dim=-1) / free_count).mean())
        object_losses.append(error[:, 10:].mean())
    free_loss = torch.stack(free_losses).mean()
    object_loss = torch.stack(object_losses).mean()
    if not with_rollout:
        return free_loss, object_loss

    rollout_free, rollout_object = [], []
    horizon = min(5, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        prediction, hidden = states[:, start], None
        for offset in range(horizon):
            prediction, hidden = model.step(
                prediction, actions[:, start + offset], mask, angle, hidden
            )
            error = (prediction - states[:, start + offset + 1]).pow(2)
            rollout_free.append(((error[:, :10] * free_mask).sum(dim=-1) / free_count).mean())
            rollout_object.append(error[:, 10:].mean())
    return (
        free_loss + 0.5 * torch.stack(rollout_free).mean(),
        object_loss + 0.5 * torch.stack(rollout_object).mean(),
    )


def _gradient_stats(
    free_loss: torch.Tensor, object_loss: torch.Tensor, parameters: list[torch.Tensor]
) -> tuple[float, float, float]:
    free_grad = torch.autograd.grad(free_loss, parameters, retain_graph=True, allow_unused=True)
    object_grad = torch.autograd.grad(object_loss, parameters, retain_graph=True, allow_unused=True)
    free_flat = torch.cat([g.reshape(-1) for g in free_grad if g is not None])
    object_flat = torch.cat([g.reshape(-1) for g in object_grad if g is not None])
    cosine = torch.nn.functional.cosine_similarity(free_flat, object_flat, dim=0)
    return float(cosine), float(free_flat.norm()), float(object_flat.norm())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError(f"seed {args.seed} not in frozen seed list")
    epochs = args.epochs or int(config["epochs"])
    steps = args.steps or int(config["steps"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    block_xy = np.asarray(config["block_initial_xy"], dtype=float)
    # Initialize MuJoCo once here so missing assets fail before expensive collection.
    MujocoArmEnv(xml_path=PUSH_XML)

    print(f"device={device} seed={args.seed} epochs={epochs}", flush=True)
    train = collect_push_domains(
        protocol.train, trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps, seed=args.seed * 10_000, targets=calibration,
        excitation="goal", block_initial_xy=block_xy,
        goal_exploration_std=float(config.get("goal_exploration_std", 0.0)),
    )
    validation = collect_push_domains(
        protocol.validation, trajectories_per_domain=int(config["trajectories_per_validation_domain"]),
        steps=steps, seed=args.seed * 10_000 + 50_000, targets=calibration,
        excitation="goal", block_initial_xy=block_xy,
        goal_exploration_std=float(config.get("goal_exploration_std", 0.0)),
    )
    train_batch = _batch(train, device)
    validation_batch = _batch(validation, device)
    model = ReducedCoordinateGraphWorldModel(
        TopologyGraphConfig(hidden_dim=int(config["matched_hidden_dim"])),
        detach_object_features=bool(config.get("detach_object_features", False)),
        bridge_edge_features=bool(config.get("bridge_edge_features", False)),
        packed_active_nodes=bool(config.get("packed_active_nodes", False)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    shared = list(model.node_encoder.parameters()) + list(model.message.parameters())
    shared += list(model.update.parameters()) + list(model.temporal.parameters())
    history, best_value, best_epoch, best_state = [], float("inf"), -1, None

    for epoch in range(epochs):
        model.train()
        free_loss, object_loss = _loss_parts(model, train_batch, with_rollout=True)
        cosine, free_norm, object_norm = _gradient_stats(free_loss, object_loss, shared)
        total = free_loss + object_loss
        optimizer.zero_grad()
        total.backward()
        total_grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_free, val_object = _loss_parts(model, validation_batch, with_rollout=True)
        val_total = float(val_free + val_object)
        if val_total < best_value:
            best_value, best_epoch = val_total, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch + 1,
            "train_free_loss": float(free_loss.detach()),
            "train_object_loss": float(object_loss.detach()),
            "validation_free_loss": float(val_free), "validation_object_loss": float(val_object),
            "gradient_cosine": cosine, "free_gradient_norm": free_norm,
            "object_gradient_norm": object_norm, "total_gradient_norm_before_clip": total_grad,
        }
        history.append(row)
        print(
            f"epoch={epoch + 1:02d} val={val_total:.6f} "
            f"free={float(val_free):.6f} obj={float(val_object):.6f} "
            f"grad_cos={cosine:+.3f} grad={total_grad:.3f}", flush=True,
        )

    final_model = copy.deepcopy(model)
    best_model = copy.deepcopy(model)
    assert best_state is not None
    best_model.load_state_dict(best_state)
    rows = []
    for index, domain in enumerate(protocol.test):
        trajectories = collect_push_domains(
            (domain,), trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            steps=steps, seed=args.seed * 100_000 + index * 1000 + 500,
            targets=evaluation, excitation="goal", block_initial_xy=block_xy,
            goal_exploration_std=float(config.get("goal_exploration_std", 0.0)),
        )
        for checkpoint, selected in (("final", final_model), ("best_validation", best_model)):
            metrics = evaluate_graph_surgery_model(
                selected, domain, trajectories, device=device,
                horizon=int(config["rollout_horizon"]), use_topology=True,
            )
            rows.append({"domain": domain.domain_id, "checkpoint": checkpoint, **metrics})
            print(
                f"{domain.domain_id} {checkpoint}: free={metrics['free_arm_rmse']:.4f} "
                f"object={metrics['object_rmse']:.4f}", flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    summary = {
        "seed": args.seed, "epochs": epochs, "steps": steps,
        "best_epoch": best_epoch, "best_validation_loss": best_value,
        "negative_gradient_fraction": sum(row["gradient_cosine"] < 0 for row in history) / len(history),
        "history": history, "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
