"""Z6: validation-select a bounded reaction checkpoint, including zero correction."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.topology_graph_world_model import TopologyGraphConfig
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from scripts.run_bt_dpwm_gate_y0 import cached_collect
from scripts.run_dual_expert_gate_q0a import _batch
from scripts.run_object_preserving_projection_x1 import _losses, evaluate
from scripts.run_push_benchmark import collect_push_domains


def validation_score(model, protocol, trajectories_by_domain, device, horizon):
    scores = []
    for domain in protocol.validation:
        row = evaluate({"candidate": model}, domain, trajectories_by_domain[domain.domain_id],
                       device, horizon)[0]
        scores.append(row["free_rmse"])
    return float(np.mean(scores))


def one_step_joint_loss(model, batch):
    states, actions, mask, _ = batch
    zeros = torch.zeros_like(mask)
    hidden, losses = None, []
    for step in range(actions.shape[1]):
        prediction, hidden = model.step(
            states[:, step], actions[:, step], zeros, zeros, hidden
        )
        losses.append((prediction[:, :10] - states[:, step + 1, :10]).pow(2).mean())
    return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/trajectory_cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in cfg["seeds"]: raise ValueError("seed outside frozen Z6 list")
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"])); targets = load_target_split(Path(q0a["targets"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = dict(steps=int(q0a["steps"]), excitation="goal",
                  block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
                  goal_exploration_std=float(q0a["goal_exploration_std"]))
    train_key = json.dumps({"kind": "push_train", "seed": args.seed,
        "domains": [x.domain_id for x in protocol.train], "q0a": q0a}, sort_keys=True)
    train = cached_collect(args.cache_dir, train_key, lambda: collect_push_domains(
        protocol.train, trajectories_per_domain=int(q0a["trajectories_per_train_domain"]),
        seed=args.seed * 10_000, targets=tuple(x.as_array() for x in targets.calibration), **common))
    validation = {}
    validation_seed = args.seed * 100_000 + 90_000
    for index, domain in enumerate(protocol.validation):
        seed = validation_seed + index * 1000
        key = json.dumps({"kind": "push_validation", "seed": seed,
                          "domain": domain.domain_id, "q0a": q0a}, sort_keys=True)
        validation[domain.domain_id] = cached_collect(args.cache_dir, key, lambda d=domain, s=seed:
            collect_push_domains((d,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
                                 seed=s, targets=tuple(x.as_array() for x in targets.evaluation), **common))
    torch.manual_seed(args.seed)
    model = BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=int(cfg["robot_hidden_dim"])),
        contact_conditioned_robot=True, independent_object_encoder=True,
        object_hidden_dim=int(cfg["object_hidden_dim"]), reaction_rank=int(cfg["reaction_rank"]),
        reaction_geometry_gate=bool(cfg.get("reaction_geometry_gate", False)),
        reaction_gate_threshold=float(cfg.get("reaction_gate_threshold", -0.005)),
        reaction_gate_temperature=float(cfg.get("reaction_gate_temperature", 0.002)),
        reaction_scale=float(cfg.get("reaction_scale", 1.0)),
        reaction_physical_features=bool(cfg.get("reaction_physical_features", False)),
        reaction_event_decay=cfg.get("reaction_event_decay"),
    ).to(device)
    source = torch.load(str(cfg["source_model_template"]).format(seed=args.seed), map_location=device)
    fresh = model.state_dict()
    # Keep trained scaffold/object weights, but restore the zero-initialized adapter.
    for name, value in source.items():
        if not name.startswith("reaction_adapter."):
            fresh[name] = value
    model.load_state_dict(fresh)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("reaction_adapter."))
    optimizer = torch.optim.Adam(model.reaction_adapter.parameters(),
                                 lr=float(cfg["reaction_learning_rate"]))
    batch = _batch(train, device); horizon = int(cfg["rollout_horizon"])
    domain_batches = None
    if bool(cfg.get("group_robust_reaction_training", False)):
        domain_ids = sorted({trajectory.domain_id for trajectory in train})
        domain_batches = [
            _batch([trajectory for trajectory in train if trajectory.domain_id == domain_id], device)
            for domain_id in domain_ids
        ]
    best_epoch = 0; best_score = validation_score(model, protocol, validation, device, horizon)
    best_state = copy.deepcopy(model.state_dict()); records = [{"epoch": 0, "validation_free_rmse": best_score}]
    print(f"[select] epoch=000 validation_free={best_score:.6f}", flush=True)
    for epoch in range(1, int(cfg["reaction_epochs"]) + 1):
        if domain_batches is not None:
            domain_losses = torch.stack([_losses(model, item, horizon)[0]
                                         for item in domain_batches])
            temperature = float(cfg.get("group_robust_temperature", 0.002))
            joint = temperature * torch.logsumexp(domain_losses / temperature, dim=0)
        elif bool(cfg.get("one_step_reaction_training", False)):
            joint = one_step_joint_loss(model, batch)
        else:
            joint, _ = _losses(model, batch, horizon)
        optimizer.zero_grad(); joint.backward()
        torch.nn.utils.clip_grad_norm_(model.reaction_adapter.parameters(), 5.0); optimizer.step()
        if epoch % int(cfg["selection_interval"]) == 0:
            score = validation_score(model, protocol, validation, device, horizon)
            records.append({"epoch": epoch, "train_joint_loss": float(joint.detach()),
                            "validation_free_rmse": score})
            print(f"[select] epoch={epoch:03d} train={joint.item():.6f} validation_free={score:.6f}", flush=True)
            if score < best_score:
                best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
    if not bool(cfg.get("select_best_validation", True)):
        best_epoch = int(cfg["reaction_epochs"])
        best_score = records[-1]["validation_free_rmse"]
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    summary = {"config_version": cfg["version"], "seed": args.seed,
               "selected_epoch": best_epoch, "best_validation_free_rmse": best_score,
               "zero_validation_free_rmse": records[0]["validation_free_rmse"], "records": records}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Z6 seed={args.seed}] selected_epoch={best_epoch} validation_free={best_score:.6f}", flush=True)


if __name__ == "__main__": main()
