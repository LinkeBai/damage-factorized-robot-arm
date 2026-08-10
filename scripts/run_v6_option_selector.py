"""Residual-aware selection among feasible IK references."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from robotarm.envs.fk import forward_kinematics
from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.controllers import joint_reference_action, solve_reach_reference
from robotarm.training.control_eval import infer_dfwm_context
from robotarm.training.g1_mechanism import train_mechanism_models
from robotarm.training.sim_data import collect_controller_domains
from robotarm.training.sim_protocol import build_g1_protocol
from robotarm.training.target_split import load_target_split


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=7); args = ap.parse_args()
    seed = args.seed; torch.manual_seed(seed); np.random.seed(seed)
    protocol = build_g1_protocol(); split = load_target_split()
    cal = tuple(t.as_array() for t in split.calibration); targets = tuple(t.as_array() for t in split.evaluation)
    ranges = MujocoArmEnv().joint_ranges; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = collect_controller_domains(protocol.train, trajectories_per_domain=2, steps=100, seed=seed*10000, targets=cal)
    models = train_mechanism_models(protocol.train, train, ranges, epochs=60, device=device)
    rows=[]
    for di, domain in enumerate(protocol.test):
        calibration=collect_controller_domains((domain,), trajectories_per_domain=5, steps=100, seed=seed*100000+di*1000, targets=cal)
        context=infer_dfwm_context(models,domain,calibration,ranges,shots=5,latent_steps=30,device=device)
        wm=models.dfwm_world_model
        for ti,target in enumerate(targets):
            env=MujocoArmEnv(residual_physics=domain.residual); locked={i:domain.damage.lock_angle_of(i) for i in domain.damage.locked}
            base_ref,_=solve_reach_reference(target,env.joint_ranges,locked_joints=locked)
            obs=env.reset(target=target,damage_config=domain.damage); reached=False
            rng=np.random.default_rng(seed+ti+di*100)
            for step in range(300):
                candidates=[base_ref]
                for _ in range(15):
                    q=base_ref+rng.normal(0.0,0.12,5); q=np.clip(q,env.joint_ranges[:,0],env.joint_ranges[:,1])
                    for j,a in locked.items(): q[j]=a
                    candidates.append(q)
                state=torch.as_tensor(obs['state'],dtype=torch.float32,device=device).reshape(1,-1)
                candidate_actions=np.stack([joint_reference_action(obs['state'],q,locked_joints=tuple(domain.damage.locked)) for q in candidates])
                with torch.no_grad():
                    states=state.expand(len(candidates),-1); actions=torch.as_tensor(candidate_actions,dtype=torch.float32,device=device)
                    pred,_=wm.step(states,actions,context.reshape(1,-1).expand(len(candidates),-1),None)
                    qpred=pred['mean'][:,:5].cpu().numpy()
                pred_pos=np.stack([forward_kinematics(q) for q in qpred])
                score=np.linalg.norm(pred_pos-target,axis=1)+0.02*np.mean(candidate_actions**2,axis=1)
                action=candidate_actions[int(np.argmin(score))]
                result=env.step(action); obs=result['observation']; distance=float(np.linalg.norm(env.ee_pos()-target))
                if distance<=0.05: reached=True; break
            rows.append({'seed':seed,'domain':domain.domain_id,'target':f'eval_{ti:02d}','success':int(reached),'steps':step+1,'final_distance_m':distance,'candidates':16})
            print(rows[-1],flush=True)
    out=Path(f'results/final/v6-option-selector-seed{seed}.csv'); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'wrote {out.resolve()}')


if __name__=='__main__': main()
