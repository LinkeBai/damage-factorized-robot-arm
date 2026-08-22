"""Infer a safe Z70 residual context from a recorded real-arm calibration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.envs.damage import DamageConfig
from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.physical_context_encoder import UncertainPhysicalContextEncoder
from robotarm.models.projected_residual_innovation import ProjectedResidualInnovation
from robotarm.models.topology_graph_world_model import TopologyGraphConfig
from robotarm.training.sim_data import SimTrajectory
from scripts.evaluate_bt_dpwm_safe_adapt_z49 import encode_context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration", type=Path, help="transitions.npz from real collector")
    ap.add_argument("--config", type=Path, default=Path(
        "config/experiment/g2_bt_dpwm_z69_context_eval_z70_v1.yaml"))
    ap.add_argument("--model-seed", type=int, default=7, choices=(7, 17, 27))
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    adapter_cfg = yaml.safe_load(Path(cfg["z48_config"]).read_text(encoding="utf-8"))
    base_cfg = yaml.safe_load(Path(adapter_cfg["base_config"]).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.calibration)
    states = torch.as_tensor(data["states"], dtype=torch.float32)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32)
    locked = int(data["locked_index"]); lock_angle = float(data["lock_angle"])
    if states.shape[0] != actions.shape[0]+1 or states.shape[1:] != (14,) or actions.shape[1:] != (5,):
        raise ValueError("invalid real calibration tensor shapes")
    damage = DamageConfig.lock_single(locked, lock_angle)
    domain = SimpleNamespace(damage=damage)
    trajectory = SimTrajectory("real", states, actions, actions.clone())
    bt = BlockTriangularDPWM(TopologyGraphConfig(hidden_dim=int(base_cfg["hidden_dim"])),
        contact_conditioned_robot=True, independent_object_encoder=True,
        object_hidden_dim=int(base_cfg["object_hidden_dim"])).to(device)
    bt_run = Path(str(cfg["bt_run_template"]).format(seed=args.model_seed))
    bt.load_state_dict(torch.load(bt_run/"model.pt", map_location=device))
    adapter = ProjectedResidualInnovation(latent_dim=8,
        rank=int(adapter_cfg["adapter_rank"]),
        hidden_dim=int(adapter_cfg["adapter_hidden_dim"]),
        position_limit=float(cfg["correction_position_limit"]),
        velocity_limit=float(cfg["correction_velocity_limit"]),
        factorized_context=bool(adapter_cfg.get("factorized_context", False)),
        analytic_history=bool(adapter_cfg.get("analytic_history", False)),
        history_deadband=float(adapter_cfg.get("history_deadband", .04))).to(device)
    adapter_run = Path(str(cfg["z48_run_template"]).format(seed=args.model_seed))
    adapter.load_state_dict(torch.load(adapter_run/"bt_adapter.pt", map_location=device))
    encoder = UncertainPhysicalContextEncoder(
        hidden_dim=int(cfg["context_encoder_hidden_dim"])).to(device)
    encoder_run = Path(str(cfg["context_encoder_run_template"]).format(seed=args.model_seed))
    encoder.load_state_dict(torch.load(encoder_run/"context_encoder.pt", map_location=device))
    for module in (bt, adapter, encoder):
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    incumbent, incumbent_lv, rows = torch.zeros(8, device=device), None, []
    available = actions.shape[0]
    for budget in cfg["transition_budgets"]:
        if budget > available: continue
        context, diagnostics = encode_context(bt, adapter, encoder, trajectory,
            domain, int(budget), device, cfg, True, incumbent, incumbent_lv)
        incumbent = context.detach().clone()
        incumbent_lv = diagnostics.get("posterior_log_variance")
        rows.append({"budget": int(budget), "context": context.cpu().tolist(),
                     "diagnostics": diagnostics})
    result = {"version": "bt_dpwm_real_context_v1", "model_seed": args.model_seed,
        "source": str(args.calibration), "locked_index": locked,
        "lock_angle": lock_angle, "available_transitions": int(available),
        "contexts": rows, "simulator_privileged_labels_used": False}
    output = args.output or args.calibration.with_name("inferred_context.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
