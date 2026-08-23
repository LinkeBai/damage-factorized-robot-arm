"""Z49: safe transition-budget adaptation of frozen Z48 adapters."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.projected_residual_innovation import (
    FewShotProjectedModel, ProjectedResidualInnovation,
)
from robotarm.models.physical_context_encoder import (
    PhysicalContextEncoder, UncertainPhysicalContextEncoder)
from robotarm.models.topology_graph_world_model import (
    TopologyGraphConfig, TopologyGraphWorldModel,
)
from robotarm.training.safe_residual_adaptation import (
    SafeAdaptConfig, safe_adapt_residual,
)
from robotarm.training.g1_mechanism import residual_descriptor
from robotarm.envs.residual_physics import ResidualPhysicsConfig, RESIDUAL_PROFILES
from robotarm.training.sim_protocol import DomainSpec, load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.run_bt_dpwm_fewshot_z48 import (
    add_compositional_training_domains, physical_context_batch, rollout_loss,
)
from scripts.run_object_preserving_projection_x1 import evaluate
from scripts.run_push_benchmark import collect_push_domains


class IdentityTopologySurgery:
    def project_state(self, state, damage_mask, lock_angle):
        return state

    def project_action(self, action, damage_mask):
        return action


def register_robustness_domains(specifications):
    """Build evaluation-only domains from frozen inline residual profiles."""
    domains = []
    for item in specifications:
        name = str(item["name"])
        values = dict(item["physics"])
        if "actuator_scale" in values:
            values["actuator_scale"] = tuple(float(x) for x in values["actuator_scale"])
        if "backlash" in values:
            values["backlash"] = tuple(float(x) for x in values["backlash"])
        profile = ResidualPhysicsConfig(name=name, **values)
        incumbent = RESIDUAL_PROFILES.get(name)
        if incumbent is not None and incumbent != profile:
            raise ValueError(f"residual profile {name!r} already has different values")
        RESIDUAL_PROFILES[name] = profile
        domains.extend(DomainSpec(str(topology), name, "test")
                       for topology in item["topologies"])
    ids = [domain.domain_id for domain in domains]
    if len(ids) != len(set(ids)):
        raise ValueError("robustness domains must have unique topology/profile pairs")
    return tuple(domains)


def trajectory_segment(trajectory, start: int, transitions: int):
    stop = start + transitions
    kwargs = {
        "states": trajectory.states[start:stop + 1],
        "actions": trajectory.actions[start:stop],
        "applied_actions": trajectory.applied_actions[start:stop],
    }
    if trajectory.contact_mask is not None:
        kwargs["contact_mask"] = trajectory.contact_mask[start:stop]
    if trajectory.contact_impulses is not None:
        kwargs["contact_impulses"] = trajectory.contact_impulses[start:stop]
    if trajectory.table_impulses is not None:
        kwargs["table_impulses"] = trajectory.table_impulses[start:stop]
    return replace(trajectory, **kwargs)


def adapt(model, adapter, trajectory, domain, budget, device, cfg,
          topology_aware):
    if budget == 0:
        return torch.zeros(adapter.latent_dim, device=device), {
            "rolled_back": True, "accepted_steps": 0, "reason": "zero_budget"
        }
    fit_count = max(2, int(np.floor(budget * float(cfg["fit_fraction"]))))
    validation_count = budget - fit_count
    if validation_count < 2:
        validation_count = 2
        fit_count = budget - validation_count
    fit = trajectory_segment(trajectory, 0, fit_count)
    validation = trajectory_segment(trajectory, fit_count, validation_count)
    horizon = min(5, fit_count, validation_count)
    fit_fn = lambda z: rollout_loss(
        model, adapter, [fit], domain, device, z, horizon, topology_aware)
    validation_fn = lambda z: rollout_loss(
        model, adapter, [validation], domain, device, z, horizon, topology_aware)
    acfg = cfg["adaptation"]
    result = safe_adapt_residual(
        fit_fn, validation_fn, device=device,
        config=SafeAdaptConfig(
            latent_dim=adapter.latent_dim, steps=int(acfg["steps"]),
            initial_step_size=float(acfg["initial_step_size"]),
            backtracking_factor=float(acfg["backtracking_factor"]),
            backtracking_steps=int(acfg["backtracking_steps"]),
            trust_radius=float(acfg["trust_radius"]), l2=float(acfg["l2"]),
            validation_tolerance=float(acfg["validation_tolerance"]),
            minimum_validation_improvement=float(
                acfg["minimum_validation_improvement"]),
        ))
    diagnostics = {
        "rolled_back": result.rolled_back,
        "accepted_steps": result.accepted_steps,
        "attempted_steps": result.attempted_steps,
        "initial_fit_loss": result.initial_fit_loss,
        "initial_validation_loss": result.initial_validation_loss,
        "best_validation_loss": result.best_validation_loss,
        "z_norm": float(result.z.norm()), "history": result.history,
    }
    return result.z, diagnostics


def encode_context(model, adapter, encoder, trajectory, domain, budget, device,
                   cfg, topology_aware, incumbent=None,
                   incumbent_log_variance=None):
    """Infer physical context on a fit split and safely select its magnitude."""
    zero = torch.zeros(adapter.latent_dim, device=device)
    if budget == 0:
        return zero, {"rolled_back": True, "accepted_steps": 0,
                      "reason": "zero_budget", "z_norm": 0.0}
    fit_count = max(2, int(np.floor(budget * float(cfg["fit_fraction"]))))
    validation_count = budget - fit_count
    if validation_count < 2:
        validation_count = 2
        fit_count = budget - validation_count
    topology_key = ",".join(str(x) for x in sorted(domain.damage.locked))
    minimum_fit = int(cfg.get("minimum_fit_transitions_by_topology", {}).get(
        topology_key, 0))
    if fit_count < minimum_fit:
        retained = zero if incumbent is None else incumbent.detach().clone()
        return retained, {"rolled_back": True, "accepted_steps": 0,
            "reason": "topology_observability_wait", "fit_count": fit_count,
            "minimum_fit_count": minimum_fit, "z_norm": float(retained.norm()),
            "posterior_log_variance": incumbent_log_variance}
    validation = trajectory_segment(trajectory, fit_count, validation_count)
    mask, _ = _damage_tensors([domain.damage], device)
    prefix_counts = sorted(set(
        x for x in cfg.get("encoder_prefix_counts", [2, 3, 5, 10, 15, 30])
        if 2 <= int(x) <= fit_count))
    if fit_count not in prefix_counts:
        prefix_counts.append(fit_count)
    with torch.no_grad():
        estimates = []
        for prefix_count in prefix_counts:
            prefix = trajectory_segment(trajectory, 0, int(prefix_count))
            encoder_args = (prefix.states.unsqueeze(0).to(device),
                            prefix.actions.unsqueeze(0).to(device), mask)
            if isinstance(encoder, UncertainPhysicalContextEncoder):
                mean, log_variance = encoder(
                    *encoder_args, return_uncertainty=True)
                estimates.append((int(prefix_count), mean[0], log_variance[0]))
            else:
                mean = encoder(*encoder_args)
                estimates.append((int(prefix_count), mean[0], None))
    horizon = min(5, fit_count, validation_count)
    validations = [(budget, validation, horizon)]
    if cfg.get("nested_support_validation", False):
        validations = []
        prefixes = sorted(set(int(x) for x in
            cfg.get("support_validation_prefix_budgets", [5, 10, 25, 50])
            if int(x) <= budget))
        if budget not in prefixes:
            prefixes.append(budget)
        for prefix_budget in prefixes:
            prefix_fit = max(2, int(np.floor(
                prefix_budget*float(cfg["fit_fraction"]))))
            prefix_validation = prefix_budget-prefix_fit
            if prefix_validation < 2:
                prefix_validation = 2; prefix_fit = prefix_budget-2
            if prefix_fit < max(2, minimum_fit):
                continue
            segment = trajectory_segment(
                trajectory, prefix_fit, prefix_validation)
            validations.append((prefix_budget, segment,
                min(5, prefix_fit, prefix_validation)))
        if not validations:
            validations = [(budget, validation, horizon)]
    validation_losses = lambda z: [float(rollout_loss(
        model, adapter, [segment], domain, device, z, window_horizon,
        topology_aware)) for _, segment, window_horizon in validations]
    incumbent = zero if incumbent is None else incumbent.detach().clone()
    with torch.no_grad():
        incumbent_losses = validation_losses(incumbent)
        incumbent_loss = float(np.mean(incumbent_losses))
        denominators = np.maximum(np.asarray(incumbent_losses), 1e-12)
        maximum_window_regression = float(cfg.get(
            "maximum_support_window_regression", float("inf")))
        def candidate_score(z):
            losses = validation_losses(z)
            ratios = np.asarray(losses)/denominators
            eligible = bool(np.max(ratios-1.0) <= maximum_window_regression)
            return float(np.mean(ratios)), losses, eligible
        candidates = []
        for prefix_count, estimate, log_variance in estimates:
            for shrink in cfg.get("encoder_shrink_factors", [1.0, 0.5, 0.25, 0.1]):
                scaled = estimate * float(shrink)
                fused_log_variance = log_variance
                z = scaled
                if (log_variance is not None and incumbent_log_variance is not None):
                    old_lv = torch.as_tensor(
                        incumbent_log_variance, device=device, dtype=scaled.dtype)
                    old_precision, new_precision = torch.exp(-old_lv), torch.exp(-log_variance)
                    z = ((old_precision*incumbent + new_precision*scaled) /
                         (old_precision+new_precision))
                    fused_log_variance = -torch.log(old_precision+new_precision)
                score, losses, eligible = candidate_score(z)
                candidates.append((score, int(prefix_count), float(shrink),
                                   z, fused_log_variance, losses, eligible))
        # A previously accepted context is never irreversible: z=0 remains a
        # deployment-safe candidate at every larger budget.
        if float(incumbent.norm()) > 0.0:
            score, losses, eligible = candidate_score(zero)
            candidates.append((score, 0, 0.0, zero, None, losses, eligible))
    eligible_candidates = [item for item in candidates if item[6]]
    if eligible_candidates:
        selected_candidate = min(eligible_candidates, key=lambda x: x[0])
    else:
        selected_candidate = (float("inf"), 0, 0.0, incumbent,
                              None, incumbent_losses, False)
    (best_score, best_prefix, best_shrink, best_z, best_log_variance,
     best_window_losses, candidate_eligible) = selected_candidate
    best_loss = float(np.mean(best_window_losses))
    if isinstance(encoder, UncertainPhysicalContextEncoder):
        minimum = float(cfg.get("replacement_minimum_validation_improvement", 0.0))
        minimum = float(cfg.get(
            "replacement_minimum_by_topology", {}).get(topology_key, minimum))
    else:
        minimum = float(cfg["adaptation"]["minimum_validation_improvement"])
    relative_improvement = 1.0-best_score
    mean_std = (float(torch.exp(0.5*best_log_variance).mean())
                if best_log_variance is not None else 0.0)
    uncertainty_ok = mean_std <= float(cfg.get(
        "maximum_context_mean_std", float("inf")))
    accepted = candidate_eligible and relative_improvement >= minimum and uncertainty_ok
    selected = best_z if accepted else incumbent
    zero_recovery = accepted and best_prefix == 0 and best_shrink == 0.0
    return selected, {
        "rolled_back": not accepted, "accepted_steps": int(accepted),
        "reason": ("zero_context_recovery" if zero_recovery else
                   "context_encoder" if accepted else "incumbent_retained"),
        "fit_count": fit_count, "validation_count": validation_count,
        "initial_validation_loss": incumbent_loss,
        "best_validation_loss": best_loss,
        "support_validation_budgets": [x[0] for x in validations],
        "initial_support_window_losses": incumbent_losses,
        "best_support_window_losses": best_window_losses,
        "maximum_support_window_regression": maximum_window_regression,
        "validation_improvement": relative_improvement,
        "required_validation_improvement": minimum,
        "prefix_count": best_prefix if accepted else 0,
        "shrink_factor": best_shrink if accepted else None,
        "raw_z_norm": (0.0 if zero_recovery else
                       float(best_z.norm()/max(best_shrink, 1e-12))),
        "z_norm": float(selected.norm()),
        "candidate_context": [float(x) for x in best_z.detach().cpu()],
        "selected_context": [float(x) for x in selected.detach().cpu()],
        "context_mean_std": mean_std,
        "context_log_variance": ([float(x) for x in best_log_variance.detach().cpu()]
                                 if best_log_variance is not None else None),
        "posterior_log_variance": (None if zero_recovery else
            [float(x) for x in best_log_variance.detach().cpu()] if accepted
            and best_log_variance is not None else incumbent_log_variance),
        "candidate_validation_losses": {
            f"p{prefix}/s{shrink}": {"relative_score": score,
                "window_losses": losses, "eligible": eligible}
            for score, prefix, shrink, _, _, losses, eligible in candidates},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(
        "config/experiment/g2_bt_dpwm_safe_adapt_z49_v1.yaml"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--oracle-context", action="store_true",
                        help="Use the simulator residual descriptor as z; diagnostic only.")
    parser.add_argument("--oracle-scale", type=float, default=1.0,
                        help="Diagnostic multiplier for the oracle descriptor.")
    parser.add_argument("--query-oracle", action="store_true",
                        help="Infer z on one evaluation trajectory; diagnostic only.")
    parser.add_argument("--domains", nargs="*",
                        help="Optional domain ids for focused diagnostics.")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if cfg.get("base_evaluation_config"):
        parent_cfg = yaml.safe_load(Path(
            cfg["base_evaluation_config"]).read_text(encoding="utf-8"))
        cfg = {**parent_cfg, **{key: value for key, value in cfg.items()
                               if key != "base_evaluation_config"}}
    z48_cfg = yaml.safe_load(Path(cfg["z48_config"]).read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path(z48_cfg["base_config"]).read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(base_cfg["q0a_config"]).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = load_g1_protocol(Path(q0a["protocol"]))
    context_center = None
    if z48_cfg.get("center_physical_context", False):
        train_domains = add_compositional_training_domains(protocol, z48_cfg)
        context_center = physical_context_batch(
            train_domains, device, torch.float32, centered=False).mean(0)
    targets = load_target_split(Path(q0a["targets"]))
    context_encoder = None
    if cfg.get("context_encoder_run_template"):
        encoder_cls = (UncertainPhysicalContextEncoder
                       if cfg.get("context_encoder_uncertainty", False)
                       else PhysicalContextEncoder)
        context_encoder = encoder_cls(
            hidden_dim=int(cfg.get("context_encoder_hidden_dim", 96))).to(device)
        encoder_run = Path(str(cfg["context_encoder_run_template"]).format(
            seed=args.seed))
        context_encoder.load_state_dict(torch.load(
            encoder_run/"context_encoder.pt", map_location=device))
        context_encoder.eval()
        for parameter in context_encoder.parameters():
            parameter.requires_grad_(False)
    base_run = Path(str(cfg.get("base_run_template",
        "runs/g2_bt_dpwm_meta_train_z32/seed{seed}_v1")).format(seed=args.seed))
    adapter_run = Path(str(cfg["z48_run_template"]).format(seed=args.seed))
    shared = TopologyGraphWorldModel(TopologyGraphConfig(
        hidden_dim=int(base_cfg["baseline_hidden_dim"]))).to(device)
    shared.load_state_dict(torch.load(base_run/"baseline_model.pt", map_location=device))
    bt_model_cfg = {}
    if cfg.get("bt_model_config"):
        bt_model_cfg = yaml.safe_load(
            Path(cfg["bt_model_config"]).read_text(encoding="utf-8"))
    bt = BlockTriangularDPWM(
        TopologyGraphConfig(hidden_dim=int(base_cfg["hidden_dim"])),
        contact_conditioned_robot=bool(
            bt_model_cfg.get("contact_conditioned_robot", True)),
        independent_object_encoder=bool(
            bt_model_cfg.get("independent_object_encoder", True)),
        object_hidden_dim=int(bt_model_cfg.get(
            "object_hidden_dim", base_cfg["object_hidden_dim"])),
        compact_bridge_object_head=bool(
            bt_model_cfg.get("compact_bridge_object_head", False)),
        geometric_object_rank=int(bt_model_cfg.get("geometric_object_rank", 0)),
        object_integration_dt=bt_model_cfg.get("object_integration_dt"),
        object_position_blend=float(
            bt_model_cfg.get("object_position_blend", 0.0)),
        geometric_object_contact_gate=bool(
            bt_model_cfg.get("geometric_object_contact_gate", False)),
        intervention_residual_support_joints=tuple(int(x) for x in
            bt_model_cfg.get("intervention_residual_support_joints", [])),
        intervention_residual_meta_train=bool(
            bt_model_cfg.get("intervention_residual_meta_train", False)),
        intervention_object_rank=int(
            bt_model_cfg.get("intervention_object_rank", 0)),
        object_bridge_alignment_rank=int(
            bt_model_cfg.get("object_bridge_alignment_rank", 0))).to(device)
    bt_run = Path(str(cfg.get("bt_run_template",
        str(base_run))).format(seed=args.seed))
    bt.load_state_dict(torch.load(bt_run/"model.pt", map_location=device))
    adapter_args = dict(latent_dim=8, rank=int(z48_cfg["adapter_rank"]),
                        hidden_dim=int(z48_cfg["adapter_hidden_dim"]),
                        position_limit=float(cfg["correction_position_limit"]),
                        velocity_limit=float(cfg["correction_velocity_limit"]),
                        factorized_context=bool(z48_cfg.get("factorized_context", False)),
                        joint_factorized_basis=bool(
                            z48_cfg.get("joint_factorized_basis", False)),
                        memory_dim=int(z48_cfg.get("adapter_memory_dim", 0)),
                        analytic_history=bool(z48_cfg.get("analytic_history", False)),
                        history_deadband=float(z48_cfg.get("history_deadband", 0.04)),
                        shared_joint_basis=bool(
                            z48_cfg.get("shared_joint_basis", False)))
    shared_adapter = ProjectedResidualInnovation(**adapter_args).to(device)
    bt_adapter = ProjectedResidualInnovation(**adapter_args).to(device)
    shared_adapter.load_state_dict(torch.load(adapter_run/"shared_adapter.pt", map_location=device))
    bt_adapter.load_state_dict(torch.load(adapter_run/"bt_adapter.pt", map_location=device))
    if cfg.get("bt_disable_locked_residual_projection", False):
        bt_adapter.project_free_coordinates = False
    if cfg.get("bt_disable_analytic_projection", False):
        bt.surgery = IdentityTopologySurgery()
    for module in (shared, bt, shared_adapter, bt_adapter):
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)

    rows = []
    common = dict(excitation=cfg.get("evaluation_excitation", "goal"),
        block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
        goal_exploration_std=float(q0a["goal_exploration_std"]))
    evaluation_domains = (register_robustness_domains(cfg["robustness_domains"])
                          if cfg.get("robustness_domains") else protocol.test)
    selected_domains = [(i, d) for i, d in enumerate(evaluation_domains)
                        if not args.domains or d.domain_id in args.domains]
    for index, domain in selected_domains:
        calibration_common = dict(common)
        calibration_common["excitation"] = cfg.get(
            "calibration_excitation", "goal")
        calibration = collect_push_domains(
            (domain,), trajectories_per_domain=1, steps=max(cfg["transition_budgets"]),
            seed=args.seed*100000+index*1000+100,
            targets=tuple(x.as_array() for x in targets.calibration),
            **calibration_common)[0]
        test = collect_push_domains(
            (domain,), trajectories_per_domain=int(q0a["trajectories_per_test_domain"]),
            steps=int(q0a["steps"]), seed=args.seed*100000+index*1000+500,
            targets=tuple(x.as_array() for x in targets.evaluation), **common)
        adaptation_trajectory = test[0] if args.query_oracle else calibration
        evaluation_trajectories = test[1:] if args.query_oracle else test
        shared_incumbent = torch.zeros(8, device=device)
        bt_incumbent = torch.zeros(8, device=device)
        shared_incumbent_log_variance = None
        bt_incumbent_log_variance = None
        for budget in cfg["transition_budgets"]:
            if args.oracle_context and budget > 0:
                oracle_z = residual_descriptor(
                    domain.residual_name, device=device,
                    dtype=next(bt_adapter.parameters()).dtype) * args.oracle_scale
                if context_center is not None:
                    oracle_z = oracle_z - context_center.to(oracle_z)
                shared_z = bt_z = oracle_z
                shared_diag = bt_diag = {
                    "rolled_back": False, "accepted_steps": 0,
                    "reason": "oracle_context", "z_norm": float(oracle_z.norm()),
                }
            elif args.oracle_context:
                shared_z = torch.zeros(8, device=device)
                bt_z = torch.zeros(8, device=device)
                shared_diag = bt_diag = {
                    "rolled_back": True, "accepted_steps": 0,
                    "reason": "zero_budget", "z_norm": 0.0,
                }
            elif context_encoder is not None:
                shared_z, shared_diag = encode_context(
                    shared, shared_adapter, context_encoder,
                    adaptation_trajectory, domain, budget, device, cfg, False,
                    shared_incumbent, shared_incumbent_log_variance)
                bt_z, bt_diag = encode_context(
                    bt, bt_adapter, context_encoder,
                    adaptation_trajectory, domain, budget, device, cfg, True,
                    bt_incumbent, bt_incumbent_log_variance)
                shared_incumbent = shared_z.detach().clone()
                bt_incumbent = bt_z.detach().clone()
                shared_incumbent_log_variance = shared_diag.get(
                    "posterior_log_variance")
                bt_incumbent_log_variance = bt_diag.get("posterior_log_variance")
            else:
                shared_z, shared_diag = adapt(shared, shared_adapter, adaptation_trajectory,
                    domain, budget, device, cfg, False)
                bt_z, bt_diag = adapt(bt, bt_adapter, adaptation_trajectory,
                    domain, budget, device, cfg, True)
            if budget == 0 and cfg.get("bt_k0_ablation_context") is not None:
                bt_z = torch.as_tensor(cfg["bt_k0_ablation_context"], device=device,
                                       dtype=next(bt_adapter.parameters()).dtype)
                bt_diag = {"rolled_back": False, "accepted_steps": 0,
                           "reason": "nonzero_k0_ablation",
                           "z_norm": float(bt_z.norm())}
            shared_model = FewShotProjectedModel(
                shared, shared_adapter, base_uses_topology=False).to(device)
            bt_model = FewShotProjectedModel(bt, bt_adapter,
                adapter_before_object=not cfg.get(
                    "bt_post_object_residual_ablation", False)).to(device)
            if cfg.get("bt_disable_analytic_projection", False):
                bt_model.surgery = IdentityTopologySurgery()
            shared_model.set_residual_context(shared_z)
            bt_model.set_residual_context(bt_z)
            result = evaluate({"shared": shared_model, "bt_dpwm": bt_model},
                domain, evaluation_trajectories, device, int(q0a["rollout_horizon"]),
                topology_aware_methods=("shared", "bt_dpwm"),
                unprojected_methods=(("bt_dpwm",) if cfg.get(
                    "bt_unprojected_evaluation", False) else ()))
            values = {x["method"]: x for x in result}
            base, candidate = values["shared"], values["bt_dpwm"]
            bt_candidate_counterfactual = None
            if (cfg.get("record_candidate_counterfactuals", False) and
                    bt_diag.get("candidate_context") is not None):
                counterfactual_z = torch.as_tensor(
                    bt_diag["candidate_context"], device=device,
                    dtype=next(bt_adapter.parameters()).dtype)
                bt_model.set_residual_context(counterfactual_z)
                bt_candidate_counterfactual = evaluate(
                    {"bt_candidate": bt_model}, domain, evaluation_trajectories,
                    device, int(q0a["rollout_horizon"]),
                    topology_aware_methods=("bt_candidate",))[0]
                bt_model.set_residual_context(bt_z)
            improvement = 100*(base["overall_rmse"]-candidate["overall_rmse"])/base["overall_rmse"]
            rows.append({"domain": domain.domain_id, "budget": budget,
                         "residual_physics": domain.residual.as_dict(),
                         "oracle_context": [float(x) for x in residual_descriptor(
                             domain.residual_name, device=torch.device("cpu"),
                             dtype=torch.float32)],
                         "improvement_pct": improvement, "shared": base,
                         "bt_dpwm": candidate, "shared_adaptation": shared_diag,
                         "bt_adaptation": bt_diag,
                         "bt_candidate_counterfactual": bt_candidate_counterfactual})
            print(f"[Z49] {domain.domain_id} n={budget} imp={improvement:+.2f}% "
                  f"rollback(shared/bt)={shared_diag['rolled_back']}/{bt_diag['rolled_back']}",
                  flush=True)
    output = args.output or Path("runs/g2_bt_dpwm_safe_adapt_z49")/f"seed{args.seed}_v1/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"config_version": cfg["version"],
        "seed": args.seed, "oracle_context": args.oracle_context,
        "oracle_scale": args.oracle_scale, "query_oracle": args.query_oracle,
        "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
