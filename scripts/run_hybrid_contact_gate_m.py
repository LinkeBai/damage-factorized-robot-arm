"""Gate M0--M2 for event-driven planar contact impulse dynamics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn
import yaml

from robotarm.models.hybrid_contact_impulse import (
    HybridContactConfig,
    HybridContactImpulseModel,
    oracle_velocity_impulse,
)
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from scripts.run_push_benchmark import collect_push_domains


class ContinuousObjectResidual(nn.Module):
    """Unconstrained continuous impulse with matched features and integration."""

    def __init__(self, feature_model: HybridContactImpulseModel, hidden_dim: int = 64) -> None:
        super().__init__()
        self.feature_model = feature_model
        for parameter in self.feature_model.parameters():
            parameter.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(19, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.raw_drag = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self, state: torch.Tensor, next_q: torch.Tensor, _contact: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features, _, _ = self.feature_model.contact_features(state, next_q)
        obj = state[:, 10:14]
        drag = torch.nn.functional.softplus(self.raw_drag)
        free_velocity = obj[:, 2:] * torch.exp(
            -drag * self.feature_model.cfg.time_step
        )
        # Bounded but otherwise unconstrained: it can act in free motion, pull
        # through a negative normal direction, or violate the friction cone.
        delta_velocity = torch.tanh(self.head(features))
        next_velocity = free_velocity + delta_velocity
        next_position = (
            obj[:, :2] + self.feature_model.cfg.time_step * next_velocity
        )
        return torch.cat((next_position, next_velocity), dim=-1), {
            "features": features,
            "delta_velocity": delta_velocity,
        }


def _flatten(trajectories, device: torch.device):
    states, targets, next_q, contacts, impulses, table_impulses = [], [], [], [], [], []
    for trajectory in trajectories:
        if trajectory.contact_mask is None:
            raise ValueError("Gate M requires per-transition contact_mask")
        if trajectory.contact_impulses is None:
            raise ValueError("Gate M requires per-transition contact_impulses")
        if trajectory.table_impulses is None:
            raise ValueError("Gate M requires per-transition table_impulses")
        states.append(trajectory.states[:-1])
        targets.append(trajectory.states[1:, 10:14])
        next_q.append(trajectory.states[1:, :5])
        contacts.append(trajectory.contact_mask)
        impulses.append(trajectory.contact_impulses)
        table_impulses.append(trajectory.table_impulses)
    return tuple(
        torch.cat(items, dim=0).to(device)
        for items in (states, targets, next_q, contacts, impulses, table_impulses)
    )


def _train(model, batch, *, epochs: int, learning_rate: float):
    states, targets, next_q, contacts, _impulses, _table_impulses = batch
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    history = []
    for epoch in range(epochs):
        prediction, _ = model(states, next_q, contacts)
        # Equal position/velocity blocks prevent dimension-count reweighting.
        loss = (
            (prediction[:, :2] - targets[:, :2]).pow(2).mean()
            + (prediction[:, 2:] - targets[:, 2:]).pow(2).mean()
        )
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        history.append({
            "epoch": epoch + 1,
            "loss": float(loss.detach()),
            "gradient_norm": gradient_norm,
        })
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"  epoch={epoch + 1:02d} loss={float(loss.detach()):.6f}",
                flush=True,
            )
    return history


@torch.no_grad()
def _rollout_rmse(model, trajectories, device: torch.device) -> float:
    errors = []
    for trajectory in trajectories:
        if trajectory.contact_mask is None:
            raise ValueError("Gate M requires per-transition contact_mask")
        prediction_object = trajectory.states[0, 10:14].to(device).unsqueeze(0)
        for step in range(trajectory.actions.shape[0]):
            state = trajectory.states[step].to(device).unsqueeze(0).clone()
            # Teacher-force the observed joint path to isolate the contact operator.
            state[:, 10:14] = prediction_object
            next_q = trajectory.states[step + 1, :5].to(device).unsqueeze(0)
            contact = trajectory.contact_mask[step].to(device).view(1)
            prediction_object, _ = model(state, next_q, contact)
            target = trajectory.states[step + 1, 10:14].to(device).unsqueeze(0)
            errors.append((prediction_object - target).pow(2).mean())
    return float(torch.stack(errors).mean().sqrt())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seeds"][0])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    steps = int(args.steps or config["steps"])
    epochs = int(args.epochs or config["epochs"])
    common = dict(
        steps=steps,
        excitation="goal",
        block_initial_xy=np.asarray(config["block_initial_xy"], dtype=float),
        goal_exploration_std=float(config["goal_exploration_std"]),
    )
    train = collect_push_domains(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        seed=seed * 10_000,
        targets=calibration,
        **common,
    )
    test = collect_push_domains(
        tuple(domain for domain in protocol.test if domain.domain_id == config["primary_domain"]),
        trajectories_per_domain=int(config["trajectories_per_test_domain"]),
        seed=seed * 100_000 + 500,
        targets=evaluation,
        **common,
    )
    train_batch = _flatten(train, device)
    contact_count = int(train_batch[3].sum())
    total_count = int(train_batch[3].numel())
    print(f"[M0] contacts={contact_count}/{total_count}", flush=True)

    states, target_obj, _, _, measured_impulse, table_impulse = train_batch
    target_states = states.clone()
    target_states[:, 10:14] = target_obj
    oracle = oracle_velocity_impulse(
        states,
        target_states,
        time_step=float(config["time_step"]),
    )
    reconstructed_velocity = states[:, 12:14] + oracle
    oracle_rmse = float((reconstructed_velocity - target_obj[:, 2:]).pow(2).mean().sqrt())
    print(f"[M1] oracle_velocity_rmse={oracle_rmse:.9f}", flush=True)

    hybrid_cfg = HybridContactConfig(
        time_step=float(config["time_step"]),
        friction_coefficient=float(config["friction_coefficient"]),
    )
    hybrid = HybridContactImpulseModel(hybrid_cfg).to(device)
    with torch.no_grad():
        _, contact_normal, contact_tangent = hybrid.contact_features(
            states, train_batch[2]
        )
        normal_component = (oracle * contact_normal).sum(-1)
        tangent_component = (oracle * contact_tangent).sum(-1)
        active = train_batch[3].bool()
        active_normal = normal_component[active]
        active_tangent = tangent_component[active]
        projected_normal = active_normal.clamp_min(0.0)
        limit = float(config["friction_coefficient"]) * projected_normal
        projected_tangent = torch.maximum(
            torch.minimum(active_tangent, limit), -limit
        )
        projected_impulse = (
            projected_normal[:, None] * contact_normal[active]
            + projected_tangent[:, None] * contact_tangent[active]
        )
        cone_projection_rmse = float(
            (projected_impulse - oracle[active]).pow(2).mean().sqrt()
        )
        negative_normal_fraction = float((active_normal < 0.0).float().mean())
        friction_violation_fraction = float(
            (active_tangent.abs() > float(config["friction_coefficient"])
             * active_normal.clamp_min(0.0)).float().mean()
        )
        measured_delta_velocity = measured_impulse / float(config["block_mass"])
        measured_normal = (measured_delta_velocity * contact_normal).sum(-1)[active]
        measured_tangent = (measured_delta_velocity * contact_tangent).sum(-1)[active]
        measured_limit = (
            float(config["friction_coefficient"]) * measured_normal.clamp_min(0.0)
        )
        measured_negative_normal_fraction = float(
            (measured_normal < -1e-8).float().mean()
        )
        measured_friction_violation_fraction = float(
            (measured_tangent.abs() > measured_limit + 1e-8).float().mean()
        )
        unexplained_velocity_rmse = float(
            (oracle[active] - measured_delta_velocity[active]).pow(2).mean().sqrt()
        )
        total_measured_delta_velocity = (
            measured_impulse + table_impulse
        ) / float(config["block_mass"])
        total_unexplained_velocity_rmse = float(
            (oracle - total_measured_delta_velocity).pow(2).mean().sqrt()
        )
        mass = float(config["block_mass"])
        implicit_denominator = 1.0 + (
            float(config["time_step"])
            * float(config["block_joint_damping"])
            / mass
        )
        reconstructed_next_velocity = (
            states[:, 12:14] + (measured_impulse + table_impulse) / mass
        ) / implicit_denominator
        implicit_momentum_rmse = float(
            (reconstructed_next_velocity - target_obj[:, 2:]).pow(2).mean().sqrt()
        )
    print(
        f"[M1-cone] projection_rmse={cone_projection_rmse:.6f} "
        f"negative_normal={negative_normal_fraction:.2%} "
        f"friction_violation={friction_violation_fraction:.2%}",
        flush=True,
    )
    print(
        f"[M1-force] negative_normal={measured_negative_normal_fraction:.2%} "
        f"friction_violation={measured_friction_violation_fraction:.2%} "
        f"unexplained_dv_rmse={unexplained_velocity_rmse:.6f}",
        flush=True,
    )
    print(
        f"[M1b-force-balance] total_unexplained_dv_rmse="
        f"{total_unexplained_velocity_rmse:.6f} "
        f"implicit_momentum_rmse={implicit_momentum_rmse:.6f}",
        flush=True,
    )
    continuous = ContinuousObjectResidual(
        HybridContactImpulseModel(hybrid_cfg).to(device)
    ).to(device)
    histories = {}
    for name, model in (("continuous", continuous), ("hybrid_impulse", hybrid)):
        torch.manual_seed(seed)
        print(f"[train] {name}", flush=True)
        histories[name] = _train(
            model,
            train_batch,
            epochs=epochs,
            learning_rate=float(config["learning_rate"]),
        )

    continuous_rmse = _rollout_rmse(continuous, test, device)
    hybrid_rmse = _rollout_rmse(hybrid, test, device)
    improvement = 100.0 * (continuous_rmse - hybrid_rmse) / continuous_rmse
    gate = config["gate"]
    m0_pass = contact_count >= int(gate["minimum_contact_transitions"])
    m1_pass = oracle_rmse <= float(gate["maximum_oracle_velocity_reconstruction_rmse"])
    m2_pass = improvement >= float(gate["minimum_object_rmse_improvement_vs_continuous_pct"])
    formal_budget = steps == int(config["steps"]) and epochs == int(config["epochs"])
    summary = {
        "seed": seed,
        "device": str(device),
        "steps": steps,
        "epochs": epochs,
        "contact_transitions": contact_count,
        "total_transitions": total_count,
        "contact_fraction": contact_count / max(total_count, 1),
        "oracle_velocity_reconstruction_rmse": oracle_rmse,
        "oracle_cone_projection_rmse": cone_projection_rmse,
        "negative_normal_impulse_fraction": negative_normal_fraction,
        "friction_cone_violation_fraction": friction_violation_fraction,
        "measured_impulse_negative_normal_fraction": measured_negative_normal_fraction,
        "measured_impulse_friction_violation_fraction": measured_friction_violation_fraction,
        "velocity_change_unexplained_by_tool_impulse_rmse": unexplained_velocity_rmse,
        "velocity_change_unexplained_by_tool_and_table_impulses_rmse": (
            total_unexplained_velocity_rmse
        ),
        "implicit_momentum_reconstruction_rmse": implicit_momentum_rmse,
        "continuous_rollout_rmse": continuous_rmse,
        "hybrid_impulse_rollout_rmse": hybrid_rmse,
        "hybrid_improvement_pct": improvement,
        "m0_passed": m0_pass,
        "m1_passed": m1_pass,
        "m2_passed": m2_pass,
        "formal_budget": formal_budget,
        "gate_passed": formal_budget and m0_pass and m1_pass and m2_pass,
        "parameters": {
            "continuous": sum(p.numel() for p in continuous.parameters() if p.requires_grad),
            "hybrid_impulse": sum(p.numel() for p in hybrid.parameters() if p.requires_grad),
        },
        "histories": histories,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"[M2] continuous={continuous_rmse:.6f} hybrid={hybrid_rmse:.6f} "
        f"improvement={improvement:+.2f}% decision="
        f"{'PASS' if summary['gate_passed'] else ('NO-GO' if formal_budget else 'SMOKE')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
