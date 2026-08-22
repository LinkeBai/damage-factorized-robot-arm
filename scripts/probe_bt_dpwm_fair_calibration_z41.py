"""Z41: fair closed-form short-trajectory calibration probe for BT-DPWM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from torch import nn

from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_bt_dpwm_gate_y0 import cached_collect
from scripts.run_object_preserving_projection_x1 import evaluate
from scripts.run_push_benchmark import collect_push_domains


class BiasCalibratedModel(nn.Module):
    """A model plus a deployment-estimated, non-trainable output correction."""

    def __init__(self, model: nn.Module, bias: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("bias", bias)

    def step(self, state, action, mask, angle, hidden):
        prediction, hidden = self.model.step(state, action, mask, angle, hidden)
        return prediction + self.bias, hidden


class AffineCalibratedModel(nn.Module):
    """Per-state delta calibration with the same coefficient count for both models."""

    def __init__(self, model: nn.Module, gain: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("gain", gain)
        self.register_buffer("bias", bias)

    def step(self, state, action, mask, angle, hidden):
        prediction, hidden = self.model.step(state, action, mask, angle, hidden)
        return state + self.gain * (prediction - state) + self.bias, hidden


@torch.no_grad()
def transition_residuals(model, trajectories, domain, device, topology_aware):
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    if not topology_aware:
        mask, angle = torch.zeros_like(mask), torch.zeros_like(angle)
    residuals, hidden = [], None
    for step in range(actions.shape[1]):
        prediction, hidden = model.step(
            states[:, step], actions[:, step], mask, angle, hidden
        )
        residuals.append(states[:, step + 1] - prediction)
    return torch.stack(residuals, dim=1), mask


@torch.no_grad()
def fit_bias(model, trajectories, domain, device, topology_aware, shrinkages):
    residuals, mask = transition_residuals(
        model, trajectories, domain, device, topology_aware
    )
    split = max(1, residuals.shape[1] // 2)
    raw = residuals[:, :split].mean(dim=(0, 1))
    locked = torch.cat((mask[0], mask[0], torch.zeros(4, device=device)))
    raw = raw * (1.0 - locked)
    validation = residuals[:, split:]
    if validation.numel() == 0:
        validation = residuals[:, :split]
    candidates = []
    for shrinkage in shrinkages:
        bias = raw * shrinkage
        joint = (validation[..., :10] - bias[:10]).pow(2).mean()
        obj = (validation[..., 10:] - bias[10:]).pow(2).mean()
        candidates.append((float(joint + obj), shrinkage, bias))
    value, shrinkage, bias = min(candidates, key=lambda item: item[0])
    return bias, {"shrinkage": shrinkage, "validation_loss": value,
                  "bias_norm": float(torch.linalg.vector_norm(bias))}


@torch.no_grad()
def fit_affine(model, trajectories, domain, device, topology_aware, shrinkages,
               blockwise=False):
    states = torch.stack([item.states for item in trajectories]).to(device)
    residuals, mask = transition_residuals(
        model, trajectories, domain, device, topology_aware
    )
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    model_mask, angle = _damage_tensors([domain.damage] * len(trajectories), device)
    if not topology_aware:
        model_mask, angle = torch.zeros_like(model_mask), torch.zeros_like(angle)
    predictions, hidden = [], None
    for step in range(actions.shape[1]):
        prediction, hidden = model.step(
            states[:, step], actions[:, step], model_mask, angle, hidden
        )
        predictions.append(prediction)
    predictions = torch.stack(predictions, dim=1)
    x = predictions - states[:, :-1]
    y = states[:, 1:] - states[:, :-1]
    split = max(1, x.shape[1] // 2)
    x_fit, y_fit = x[:, :split].reshape(-1, x.shape[-1]), y[:, :split].reshape(-1, y.shape[-1])
    x_mean, y_mean = x_fit.mean(0), y_fit.mean(0)
    covariance = ((x_fit - x_mean) * (y_fit - y_mean)).mean(0)
    variance = (x_fit - x_mean).pow(2).mean(0)
    raw_gain = covariance / (variance + 1e-5)
    raw_gain = raw_gain.clamp(0.25, 2.0)
    raw_bias = y_mean - raw_gain * x_mean
    locked = torch.cat((mask[0], mask[0], torch.zeros(4, device=device)))
    raw_gain = raw_gain * (1.0 - locked) + locked
    raw_bias = raw_bias * (1.0 - locked)
    x_validation, y_validation = x[:, split:], y[:, split:]
    if x_validation.numel() == 0:
        x_validation, y_validation = x[:, :split], y[:, :split]
    candidates = []
    for shrinkage in shrinkages:
        gain = 1.0 + shrinkage * (raw_gain - 1.0)
        bias = shrinkage * raw_bias
        error = y_validation - (gain * x_validation + bias)
        joint = error[..., :10].pow(2).mean()
        obj = error[..., 10:].pow(2).mean()
        candidates.append({"loss": float(joint + obj), "joint": float(joint),
                           "object": float(obj), "shrinkage": shrinkage,
                           "gain": gain, "bias": bias})
    if blockwise:
        robot_choice = min(candidates, key=lambda item: item["joint"])
        object_choice = min(candidates, key=lambda item: item["object"])
        gain = torch.cat((robot_choice["gain"][:10], object_choice["gain"][10:]))
        bias = torch.cat((robot_choice["bias"][:10], object_choice["bias"][10:]))
        value = robot_choice["joint"] + object_choice["object"]
        shrinkage = {"robot": robot_choice["shrinkage"],
                     "object": object_choice["shrinkage"]}
    else:
        choice = min(candidates, key=lambda item: item["loss"])
        value, shrinkage, gain, bias = (choice["loss"], choice["shrinkage"],
                                        choice["gain"], choice["bias"])
    return gain, bias, {
        "shrinkage": shrinkage, "validation_loss": value,
        "gain_deviation_norm": float(torch.linalg.vector_norm(gain - 1.0)),
        "bias_norm": float(torch.linalg.vector_norm(bias)),
    }


def rollout_affine_loss(model, trajectories, domain, device, topology_aware,
                        gain, bias, horizon):
    states = torch.stack([item.states for item in trajectories]).to(device)
    actions = torch.stack([item.actions for item in trajectories]).to(device)
    true_mask, true_angle = _damage_tensors(
        [domain.damage] * len(trajectories), device
    )
    zeros = torch.zeros_like(true_mask)
    model_mask, model_angle = ((true_mask, true_angle) if topology_aware
                               else (zeros, zeros))
    surgery = TopologySurgery()
    free_mask = torch.cat((1.0 - true_mask, 1.0 - true_mask), -1)
    free_count = free_mask.sum(-1).clamp_min(1.0)
    endpoint_losses = []
    horizon = min(horizon, actions.shape[1])
    for start in range(0, actions.shape[1] - horizon + 1, horizon):
        prediction, hidden = states[:, start], None
        for offset in range(horizon):
            raw, hidden = model.step(
                prediction, actions[:, start + offset],
                model_mask, model_angle, hidden,
            )
            prediction = prediction + gain * (raw - prediction) + bias
            prediction = surgery.project_state(prediction, true_mask, true_angle)
        target = states[:, start + horizon]
        error = (prediction - target).pow(2)
        joint = (error[:, :10] * free_mask).sum(-1) / free_count
        obj = error[:, 10:].mean(-1)
        endpoint_losses.append(joint + obj)
    return torch.stack(endpoint_losses).mean()


def fit_rollout_affine(model, trajectories, domain, device, topology_aware, *,
                       horizon, steps, learning_rate, l2, validation_every):
    if len(trajectories) < 2:
        raise ValueError("rollout affine calibration needs at least two trajectories")
    train, validation = trajectories[:-1], trajectories[-1:]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gain = nn.Parameter(torch.ones(14, device=device))
    bias = nn.Parameter(torch.zeros(14, device=device))
    optimizer = torch.optim.Adam((gain, bias), lr=learning_rate)
    with torch.no_grad():
        best_value = float(rollout_affine_loss(
            model, validation, domain, device, topology_aware,
            gain, bias, horizon,
        ))
    best_step, best_gain, best_bias = 0, gain.detach().clone(), bias.detach().clone()
    history = [{"step": 0, "validation_loss": best_value}]
    for step in range(1, steps + 1):
        loss = rollout_affine_loss(
            model, train, domain, device, topology_aware, gain, bias, horizon
        ) + l2 * ((gain - 1.0).pow(2).mean() + bias.pow(2).mean())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            gain.clamp_(0.25, 2.0); bias.clamp_(-0.1, 0.1)
        if step % validation_every == 0 or step == steps:
            with torch.no_grad():
                value = float(rollout_affine_loss(
                    model, validation, domain, device, topology_aware,
                    gain, bias, horizon,
                ))
            history.append({"step": step, "validation_loss": value})
            if value < best_value:
                best_value, best_step = value, step
                best_gain, best_bias = gain.detach().clone(), bias.detach().clone()
    return best_gain, best_bias, {
        "selected_step": best_step, "validation_loss": best_value,
        "gain_deviation_norm": float(torch.linalg.vector_norm(best_gain - 1.0)),
        "bias_norm": float(torch.linalg.vector_norm(best_bias)),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/trajectory_cache"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(cfg["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    targets = load_target_split(Path(q0a["targets"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = dict(excitation="goal", block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
                  goal_exploration_std=float(q0a["goal_exploration_std"]))
    baseline = TopologyGraphWorldModel(
        TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"]))
    ).to(device)
    baseline.load_state_dict(torch.load(
        str(cfg["baseline_model_template"]).format(seed=args.seed), map_location=device
    ))
    candidate = BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=int(cfg["hidden_dim"])),
        contact_conditioned_robot=True, independent_object_encoder=False,
        compact_bridge_object_head=True,
    ).to(device)
    candidate.load_state_dict(torch.load(
        str(cfg["candidate_model_template"]).format(seed=args.seed), map_location=device
    ))
    rows, calibration = [], []
    for index, domain in enumerate(protocol.test):
        calibration_seed = args.seed * 100_000 + index * 1000 + 100
        calibration_key = json.dumps({
            "kind": "push_domain_calibration_z41", "seed": calibration_seed,
            "domain": domain.domain_id, "steps": int(cfg["calibration_steps"]),
            "trajectories": int(cfg["calibration_trajectories"]), "q0a": q0a,
        }, sort_keys=True)
        calibration_data = cached_collect(args.cache_dir, calibration_key, lambda d=domain: collect_push_domains(
            (d,), trajectories_per_domain=int(cfg["calibration_trajectories"]),
            steps=int(cfg["calibration_steps"]), seed=calibration_seed,
            targets=tuple(item.as_array() for item in targets.calibration), **common,
        ))
        if cfg.get("calibration_type", "bias") == "rollout_affine":
            fit_kwargs = dict(
                horizon=int(cfg["calibration_rollout_horizon"]),
                steps=int(cfg["calibration_optimization_steps"]),
                learning_rate=float(cfg["calibration_learning_rate"]),
                l2=float(cfg["calibration_l2"]),
                validation_every=int(cfg["calibration_validation_every"]),
            )
            base_gain, base_bias, base_diag = fit_rollout_affine(
                baseline, calibration_data, domain, device, False, **fit_kwargs
            )
            candidate_gain, candidate_bias, candidate_diag = fit_rollout_affine(
                candidate, calibration_data, domain, device, True, **fit_kwargs
            )
            calibrated_baseline = AffineCalibratedModel(baseline, base_gain, base_bias)
            calibrated_candidate = AffineCalibratedModel(candidate, candidate_gain, candidate_bias)
        elif cfg.get("calibration_type", "bias") in ("affine", "block_affine"):
            blockwise = cfg["calibration_type"] == "block_affine"
            base_gain, base_bias, base_diag = fit_affine(
                baseline, calibration_data, domain, device, False, cfg["shrinkages"],
                blockwise=blockwise,
            )
            candidate_gain, candidate_bias, candidate_diag = fit_affine(
                candidate, calibration_data, domain, device, True, cfg["shrinkages"],
                blockwise=blockwise,
            )
            calibrated_baseline = AffineCalibratedModel(baseline, base_gain, base_bias)
            calibrated_candidate = AffineCalibratedModel(candidate, candidate_gain, candidate_bias)
        else:
            base_bias, base_diag = fit_bias(
                baseline, calibration_data, domain, device, False, cfg["shrinkages"]
            )
            candidate_bias, candidate_diag = fit_bias(
                candidate, calibration_data, domain, device, True, cfg["shrinkages"]
            )
            calibrated_baseline = BiasCalibratedModel(baseline, base_bias)
            calibrated_candidate = BiasCalibratedModel(candidate, candidate_bias)
        test_seed = args.seed * 100_000 + index * 1000 + 500
        test_key = json.dumps({"kind": "push_test", "seed": test_seed,
            "domain": domain.domain_id, "q0a": q0a}, sort_keys=True)
        test_data = cached_collect(args.cache_dir, test_key, lambda d=domain: collect_push_domains(
            (d,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
            steps=int(q0a["steps"]), seed=test_seed,
            targets=tuple(item.as_array() for item in targets.evaluation), **common,
        ))
        domain_rows = evaluate({
            "shared_calibrated": calibrated_baseline,
            "bt_dpwm_calibrated": calibrated_candidate,
        }, domain, test_data, device, int(q0a["rollout_horizon"]),
            topology_aware_methods=("bt_dpwm_calibrated",))
        result = {item["method"]: item for item in domain_rows}
        base, cand = result["shared_calibrated"], result["bt_dpwm_calibrated"]
        improve = lambda key: 100.0 * (base[key] - cand[key]) / base[key]
        rows.append({
            "seed": args.seed, "domain": domain.domain_id,
            "free_improvement_pct": improve("free_rmse"),
            "object_improvement_pct": improve("object_rmse"),
            "overall_improvement_pct": improve("overall_rmse"),
            "baseline": base, "candidate": cand,
        })
        calibration.append({"domain": domain.domain_id, "baseline": base_diag,
                            "candidate": candidate_diag})
    summary = {
        "config_version": cfg["version"], "seed": args.seed,
        "mean_free_improvement_pct": float(np.mean([x["free_improvement_pct"] for x in rows])),
        "mean_object_improvement_pct": float(np.mean([x["object_improvement_pct"] for x in rows])),
        "mean_overall_improvement_pct": float(np.mean([x["overall_improvement_pct"] for x in rows])),
        "overall_regression_count": sum(x["overall_improvement_pct"] < 0 for x in rows),
        "rows": rows, "calibration": calibration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Z41] free={summary['mean_free_improvement_pct']:+.2f}% "
          f"object={summary['mean_object_improvement_pct']:+.2f}% "
          f"overall={summary['mean_overall_improvement_pct']:+.2f}% "
          f"regressions={summary['overall_regression_count']}/4", flush=True)


if __name__ == "__main__":
    main()
