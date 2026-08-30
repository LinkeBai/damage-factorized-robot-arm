"""Z27: closed-form joint-shared ridge reaction on a frozen BT-DPWM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from scripts.run_bt_dpwm_gate_y0 import cached_collect
from scripts.run_dual_expert_gate_q0a import _batch
from scripts.run_object_preserving_projection_x1 import _losses, evaluate
from scripts.run_push_benchmark import collect_push_domains


@torch.no_grad()
def residual_design(model, batch):
    states, actions, mask, _ = batch
    zeros = torch.zeros_like(mask)
    depth = torch.linspace(0.0, 1.0, model.cfg.dof, device=states.device)
    depth = depth.view(1, 1, -1, 1).expand(states.shape[0], actions.shape[1], -1, -1)
    obj = states[:, :-1, 10:].unsqueeze(2).expand(-1, -1, model.cfg.dof, -1)
    features = torch.cat((
        states[:, :-1, :5].unsqueeze(-1),
        states[:, :-1, 5:10].unsqueeze(-1),
        actions.unsqueeze(-1),
        zeros[:, None, :, None].expand(-1, actions.shape[1], -1, -1),
        zeros[:, None, :, None].expand(-1, actions.shape[1], -1, -1),
        depth, obj,
    ), dim=-1)
    predictions, hidden = [], None
    for step in range(actions.shape[1]):
        prediction, hidden = model.step(
            states[:, step], actions[:, step], zeros, zeros, hidden
        )
        predictions.append(prediction[:, :10])
    prediction = torch.stack(predictions, dim=1)
    residual = states[:, 1:, :10] - prediction
    target = torch.stack((residual[..., :5], residual[..., 5:10]), dim=-1)
    return features.reshape(-1, features.shape[-1]), target.reshape(-1, 2)


def ridge_solutions(features, target, lambdas):
    mean = features.mean(0)
    std = features.std(0).clamp_min(1e-4)
    normalized = (features - mean) / std
    design = torch.cat((normalized, torch.ones_like(normalized[:, :1])), dim=-1)
    gram, rhs = design.T @ design, design.T @ target
    penalty = torch.eye(design.shape[1], device=design.device, dtype=design.dtype)
    penalty[-1, -1] = 0.0
    for value in lambdas:
        if float(value) == 0.0:
            theta = torch.linalg.lstsq(design, target).solution
        else:
            theta = torch.linalg.solve(gram + float(value) * penalty, rhs)
        normalized_weight, normalized_bias = theta[:-1], theta[-1]
        weight = normalized_weight / std[:, None]
        bias = normalized_bias - (mean[:, None] * weight).sum(0)
        yield float(value), weight.T.contiguous(), bias.contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/trajectory_cache"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = dict(steps=int(q0a["steps"]), excitation="goal",
                  block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
                  goal_exploration_std=float(q0a["goal_exploration_std"]))

    # Match the historical cache key/collector seed for the frozen train split.
    train_key = json.dumps({"kind": "push_train", "seed": args.seed,
        "domains": [x.domain_id for x in protocol.train], "q0a": q0a}, sort_keys=True)
    train = cached_collect(args.cache_dir, train_key, lambda: collect_push_domains(
        protocol.train, trajectories_per_domain=int(q0a["trajectories_per_train_domain"]),
        seed=args.seed * 10_000, targets=tuple(x.as_array() for x in targets.calibration), **common))
    validation_seed = args.seed * 10_000 + 750
    validation_key = json.dumps({"kind": "push_validation", "seed": args.seed,
        "domains": [x.domain_id for x in protocol.validation], "q0a": q0a}, sort_keys=True)
    validation = cached_collect(args.cache_dir, validation_key, lambda: collect_push_domains(
        protocol.validation, trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        seed=validation_seed, targets=tuple(x.as_array() for x in targets.validation), **common))

    model = BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=int(cfg["robot_hidden_dim"])),
        contact_conditioned_robot=True, independent_object_encoder=True,
        object_hidden_dim=int(cfg["object_hidden_dim"]), linear_physical_reaction=True,
    ).to(device)
    source = torch.load(str(cfg["source_model_template"]).format(seed=args.seed),
                        map_location=device)
    compatible = {name: value for name, value in source.items()
                  if name in model.state_dict() and model.state_dict()[name].shape == value.shape}
    model.load_state_dict({**model.state_dict(), **compatible})
    train_batch, validation_batch = _batch(train, device), _batch(validation, device)

    def assign(weight, bias):
        model.linear_reaction_adapter.weight.data.copy_(weight)
        model.linear_reaction_adapter.bias.data.copy_(bias)

    def zero_reaction():
        model.linear_reaction_adapter.weight.data.zero_()
        model.linear_reaction_adapter.bias.data.zero_()

    topology_cv = []
    if cfg.get("selection") == "leave_one_topology_out":
        scores = {float(value): [] for value in cfg["ridge_lambdas"]}
        for heldout in ("intact", "D2", "D4"):
            fit = [item for item in train if not item.domain_id.startswith(heldout + "__")]
            audit = [item for item in train if item.domain_id.startswith(heldout + "__")]
            zero_reaction()
            fold_features, fold_target = residual_design(model, _batch(fit, device))
            for regularization, weight, bias in ridge_solutions(
                    fold_features, fold_target, cfg["ridge_lambdas"]):
                assign(weight, bias)
                loss = float(_losses(
                    model, _batch(audit, device), int(cfg["validation_rollout_horizon"])
                )[0].detach())
                if np.isfinite(loss):
                    scores[regularization].append(loss)
                    topology_cv.append({"heldout": heldout, "lambda": regularization,
                                        "joint_loss": loss})
        means = {value: float(np.mean(losses)) for value, losses in scores.items()
                 if len(losses) == 3}
        selected_regularization = min(means, key=means.get)
        print("[selection] leave-one-topology-out", flush=True)
        for value in cfg["ridge_lambdas"]:
            value = float(value)
            if value in means:
                print(f"  lambda={value:g} topology_cv_joint={means[value]:.6f}", flush=True)
        zero_reaction()
        features, target = residual_design(model, train_batch)
        solution = next(row for row in ridge_solutions(
            features, target, [selected_regularization]
        ))
        _, selected_weight, selected_bias = solution
        assign(selected_weight, selected_bias)
        selected = {"lambda": selected_regularization,
                    "validation_joint_loss": means[selected_regularization],
                    "weight": selected_weight, "bias": selected_bias}
        candidates = []
    else:
        zero_reaction()
        features, target = residual_design(model, train_batch)
        candidates = []
        for regularization, weight, bias in ridge_solutions(
                features, target, cfg["ridge_lambdas"]):
            assign(weight, bias)
            validation_loss = float(_losses(
                model, validation_batch, int(cfg["validation_rollout_horizon"])
            )[0].detach())
            if not np.isfinite(validation_loss):
                print(f"  lambda={regularization:g} validation_joint=non-finite (rejected)",
                      flush=True)
                continue
            candidates.append({"lambda": regularization, "validation_joint_loss": validation_loss,
                               "weight": weight.detach().clone(), "bias": bias.detach().clone()})
            print(f"  lambda={regularization:g} validation_joint={validation_loss:.6f}", flush=True)
        selected = min(candidates, key=lambda row: row["validation_joint_loss"])
        assign(selected["weight"], selected["bias"])

    baseline = TopologyGraphWorldModel(TopologyGraphConfig(
        hidden_dim=int(cfg["robot_hidden_dim"]))).to(device)
    baseline.load_state_dict(torch.load(
        str(cfg["baseline_model_template"]).format(seed=args.seed), map_location=device))
    domain = next(x for x in protocol.test if x.domain_id == cfg["primary_domain"])
    index = list(protocol.test).index(domain); test_seed = args.seed * 100_000 + index * 1000 + 500
    test_key = json.dumps({"kind": "push_test", "seed": test_seed,
                           "domain": domain.domain_id, "q0a": q0a}, sort_keys=True)
    test = cached_collect(args.cache_dir, test_key, lambda: collect_push_domains(
        (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
        seed=test_seed, targets=tuple(x.as_array() for x in targets.evaluation), **common))
    rows = evaluate({"shared_baseline": baseline, "bt_dpwm": model}, domain, test,
                    device, int(q0a["rollout_horizon"]))
    values = {row["method"]: row for row in rows}; base, cand = values["shared_baseline"], values["bt_dpwm"]
    improve = lambda key: 100.0 * (base[key] - cand[key]) / base[key]
    summary = {
        "config_version": cfg["version"], "seed": args.seed, "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "selected_lambda": selected["lambda"],
        "selected_validation_joint_loss": selected["validation_joint_loss"],
        "validation_candidates": [{k: v for k, v in row.items() if k not in ("weight", "bias")}
                                  for row in candidates],
        "topology_cv": topology_cv,
        "free_arm_improvement_pct": improve("free_rmse"),
        "object_improvement_pct": improve("object_rmse"),
        "overall_improvement_pct": improve("overall_rmse"), "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Z27] lambda={selected['lambda']:g} free={summary['free_arm_improvement_pct']:+.2f}% "
          f"object={summary['object_improvement_pct']:+.2f}% "
          f"overall={summary['overall_improvement_pct']:+.2f}%", flush=True)


if __name__ == "__main__":
    main()
