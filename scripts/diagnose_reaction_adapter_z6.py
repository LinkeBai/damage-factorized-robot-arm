"""Measure reaction-adapter magnitude on contact and non-contact transitions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.hybrid_contact_impulse import HybridContactImpulseModel
from robotarm.models.topology_graph_world_model import TopologyGraphConfig
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.topology_surgery_gate import _damage_tensors


@torch.no_grad()
def main():
    q0a = yaml.safe_load(Path("config/experiment/g2_dual_expert_gate_q0a_v1.yaml").read_text())
    protocol = load_g1_protocol(Path(q0a["protocol"])); device = torch.device("cuda")
    cache_map = {
        7: ["1c2c590b85d15d980d76", "090da952144d9eddb533", "474740b351c4abffab2d", "6a2d65dab2b2497e00b6"],
        17: ["c4eed9152a37598d41f3", "fb91c0dd9f2271fb9a84", "a41d5c86c6dccfd63a19", "4c8c31ea4dfb2d01eac6"],
        27: ["9df93be109b69a0653ab", "054e0d02fd758ca4d56b", "7c9bdd8d92963ddc5704", "bd87791f915e42de5df4"],
    }
    geometry = HybridContactImpulseModel().to(device).eval()
    rows = []
    for seed, keys in cache_map.items():
        model = BlockTriangularDPWM(TopologyGraphConfig(hidden_dim=136),
            contact_conditioned_robot=True, independent_object_encoder=True,
            object_hidden_dim=32, reaction_rank=8).to(device)
        model.load_state_dict(torch.load(
            f"runs/g2_bt_dpwm_reaction_z5/seed{seed}_v1/model.pt", map_location=device))
        captured = []
        hook = model.reaction_adapter.register_forward_hook(
            lambda module, inputs, output: captured.append(output.detach().cpu()))
        contact_norm, free_norm = [], []
        contact_speed, free_speed = [], []
        contact_gap, free_gap = [], []
        for domain, key in zip(protocol.test, keys):
            trajectories = torch.load(f"runs/trajectory_cache/{key}.pt", weights_only=False)
            mask, angle = _damage_tensors([domain.damage], device)
            for trajectory in trajectories:
                hidden = None
                for step in range(len(trajectory.actions)):
                    before = len(captured)
                    _, hidden = model.step(trajectory.states[step:step+1].to(device),
                                           trajectory.actions[step:step+1].to(device),
                                           mask, angle, hidden)
                    norm = captured[before].norm(dim=-1).mean().item()
                    is_contact = bool(trajectory.contact_mask[step])
                    (contact_norm if is_contact else free_norm).append(norm)
                    speed = trajectory.states[step, -2:].norm().item()
                    (contact_speed if is_contact else free_speed).append(speed)
                    state = trajectory.states[step:step+1].to(device)
                    gap = geometry.candidate_box_contact_frames(
                        state, state[:, :5]
                    )[0].min().item()
                    (contact_gap if is_contact else free_gap).append(gap)
        hook.remove()
        rows.append({"seed": seed, "contact_count": len(contact_norm), "free_count": len(free_norm),
                     "contact_mean_norm": sum(contact_norm) / len(contact_norm),
                     "free_mean_norm": sum(free_norm) / len(free_norm),
                     "contact_mean_speed": sum(contact_speed) / len(contact_speed),
                     "free_mean_speed": sum(free_speed) / len(free_speed),
                     "moving_contact_rate": sum(x > 1e-3 for x in contact_speed) / len(contact_speed),
                     "moving_free_rate": sum(x > 1e-3 for x in free_speed) / len(free_speed),
                     "contact_gap_q50": torch.tensor(contact_gap).quantile(0.5).item(),
                     "contact_gap_q90": torch.tensor(contact_gap).quantile(0.9).item(),
                     "free_gap_q10": torch.tensor(free_gap).quantile(0.1).item(),
                     "free_gap_q50": torch.tensor(free_gap).quantile(0.5).item(),
                     "contact_to_free_ratio": (sum(contact_norm) / len(contact_norm)) /
                                              (sum(free_norm) / len(free_norm))})
    output = Path("runs/g2_bt_dpwm_reaction_diagnosis_z6/summary.json")
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps({"rows": rows}, indent=2))
    for row in rows:
        print(f"seed={row['seed']} contact={row['contact_mean_norm']:.6f} "
              f"free={row['free_mean_norm']:.6f} ratio={row['contact_to_free_ratio']:.2f} "
              f"speed={row['contact_mean_speed']:.4f}/{row['free_mean_speed']:.4f} "
              f"moving={row['moving_contact_rate']:.2%}/{row['moving_free_rate']:.2%} "
              f"gap c50/c90={row['contact_gap_q50']:.4f}/{row['contact_gap_q90']:.4f} "
              f"f10/f50={row['free_gap_q10']:.4f}/{row['free_gap_q50']:.4f}")


if __name__ == "__main__": main()
