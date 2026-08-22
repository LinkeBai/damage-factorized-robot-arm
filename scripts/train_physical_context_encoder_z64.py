"""Z64: supervised variable-budget physical context inference for frozen Z63."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.models.physical_context_encoder import (
    PhysicalContextEncoder, UncertainPhysicalContextEncoder)
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.collect_warp import collect_push_domains_warp
from scripts.run_bt_dpwm_fewshot_z48 import (
    add_compositional_training_domains, physical_context_batch)


CONTEXT_SCALE = torch.tensor([0.3, 0.7, 0.7, 1.0, 0.6, 0.6, 0.6, 0.6])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(
        "config/experiment/g2_bt_dpwm_context_encoder_z64_v1.yaml"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    parent = yaml.safe_load(Path(cfg["parent_config"]).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(parent["base_config"]).read_text(encoding="utf-8"))
    q0a = yaml.safe_load(Path(base["q0a_config"]).read_text(encoding="utf-8"))
    protocol = load_g1_protocol(q0a["protocol"])
    domains = add_compositional_training_domains(protocol, parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    count = int(cfg["trajectories_per_domain"])
    trajectories = collect_push_domains_warp(
        domains, trajectories_per_domain=count, steps=int(q0a["steps"]),
        seed=args.seed * 10000 + 640,
        block_initial_xy=np.asarray(q0a["block_initial_xy"], float),
        excitation="active")
    grouped = {d.domain_id: trajectories[i*count:(i+1)*count]
               for i, d in enumerate(domains)}
    uncertain = bool(cfg.get("uncertainty", False))
    encoder_cls = UncertainPhysicalContextEncoder if uncertain else PhysicalContextEncoder
    encoder = encoder_cls(hidden_dim=int(cfg["hidden_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-4)
    target = physical_context_batch(domains, device, torch.float32)
    scale = CONTEXT_SCALE.to(device)
    masks, _ = _damage_tensors([d.damage for d in domains], device)
    rng = np.random.default_rng(args.seed + 64000)
    history = []
    best_state, best_val = None, float("inf")
    budgets = tuple(int(x) for x in cfg["transition_budgets"])
    for epoch in range(int(cfg["epochs"])):
        budget = int(rng.choice(budgets))
        selected = [grouped[d.domain_id][int(rng.integers(0, count-1))]
                    for d in domains]
        states = torch.stack([x.states[:budget+1] for x in selected]).to(device)
        actions = torch.stack([x.actions[:budget] for x in selected]).to(device)
        if uncertain:
            nested = tuple(int(x) for x in cfg.get(
                "nested_training_budgets", budgets))
            predictions, log_variances = [], []
            for nested_budget in nested:
                nested_states = torch.stack(
                    [x.states[:nested_budget+1] for x in selected]).to(device)
                nested_actions = torch.stack(
                    [x.actions[:nested_budget] for x in selected]).to(device)
                mean, log_variance = encoder(
                    nested_states, nested_actions, masks, return_uncertainty=True)
                predictions.append(mean); log_variances.append(log_variance)
            errors = [(p-target)/scale for p in predictions]
            mse = torch.stack([e.pow(2).mean() for e in errors]).mean()
            nll = torch.stack([0.5*(e.pow(2)*torch.exp(-lv)+lv).mean()
                for e, lv in zip(errors, log_variances)]).mean()
            consistency = torch.stack([((predictions[i]-predictions[i+1])/
                scale).pow(2).mean() for i in range(len(predictions)-1)]).mean()
            monotonic = torch.stack([torch.relu(
                log_variances[i+1]-log_variances[i]).mean()
                for i in range(len(log_variances)-1)]).mean()
            loss = (mse + float(cfg.get("nll_weight", 0.1))*nll +
                    float(cfg.get("consistency_weight", 0.2))*consistency +
                    float(cfg.get("uncertainty_monotonic_weight", 0.1))*monotonic)
        else:
            prediction = encoder(states, actions, masks)
            loss = ((prediction-target)/scale).pow(2).mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        entry = {"epoch": epoch+1, "budget": budget, "loss": float(loss.detach())}
        if epoch == 0 or (epoch+1) % int(cfg["validation_interval"]) == 0:
            val_losses = []
            with torch.no_grad():
                for val_budget in (3, 6, 15, 30):
                    heldout = [grouped[d.domain_id][-1] for d in domains]
                    val_states = torch.stack(
                        [x.states[:val_budget+1] for x in heldout]).to(device)
                    val_actions = torch.stack(
                        [x.actions[:val_budget] for x in heldout]).to(device)
                    val_prediction = encoder(val_states, val_actions, masks)
                    val_losses.append(float(
                        (((val_prediction-target)/scale).pow(2).mean()).detach()))
            val = float(np.mean(val_losses)); entry["validation_loss"] = val
            if val < best_val:
                best_val = val
                best_state = {k: v.detach().cpu().clone()
                              for k, v in encoder.state_dict().items()}
        history.append(entry)
        if epoch == 0 or (epoch+1) % 25 == 0:
            print(f"[Z64] epoch {epoch+1}/{cfg['epochs']} budget={budget} "
                f"loss={float(loss.detach()):.6f}", flush=True)
    encoder.load_state_dict(best_state)
    output = args.output_dir or Path("runs/g2_bt_dpwm_context_encoder_z64")/f"seed{args.seed}_v1"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), output/"context_encoder.pt")
    (output/"summary.json").write_text(json.dumps({
        "version": cfg["version"], "seed": args.seed,
        "best_validation_loss": best_val, "history": history,
        "context_scale": CONTEXT_SCALE.tolist()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
