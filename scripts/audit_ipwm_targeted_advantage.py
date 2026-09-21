"""Frozen-checkpoint, fresh-query physics spectrum; no fitting or result filtering."""
from __future__ import annotations
import argparse, copy, hashlib, inspect, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import torch
import yaml
from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.topology_graph_world_model import TopologyGraphConfig, TopologyGraphWorldModel
from robotarm.models.selective_intervention_rollout import SelectiveInterventionRollout
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.models.physical_context_encoder import UncertainPhysicalContextEncoder
from robotarm.models.projected_residual_innovation import FewShotProjectedModel
from robotarm.training.sim_protocol import DomainSpec
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from scripts.evaluate_ipwm_support_validation_gate import build_strict, make_adapter
from scripts.run_push_benchmark import collect_push_domains

PHYSICS = ['nominal', 'weak_motor', 'high_damping', 'delay_1', 'noisy_deadband', 'mixed_composition', 'mixed_unseen']
OUT = ROOT / 'runs/ipwm_targeted_advantage_20260911'
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return yaml.safe_load(Path(path).read_text(encoding='utf-8'))
def load(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return model.to(device).eval()
def final_models(seed, device):
    cfg = read(ROOT / 'config/experiment/icra_primary_d2d4_decision_development_3seed_v1.yaml')
    kwargs = {k:v for k,v in cfg.items() if k in inspect.signature(BlockTriangularDPWM).parameters and k != 'cfg'}
    full = load(BlockTriangularDPWM(TopologyGraphConfig(hidden_dim=136), **kwargs),
                ROOT / f'runs/icra_primary_decision_full_w10_128eval_strict_v2/seed{seed}/model.pt', device)
    carrier = copy.deepcopy(full)
    with torch.no_grad():
        for p in carrier.geometric_object_head.parameters(): p.zero_()
    globalkw = dict(kwargs, geometric_object_rank=0, global_residual_rank=10)
    global_model = load(BlockTriangularDPWM(TopologyGraphConfig(hidden_dim=136), **globalkw),
                        ROOT / f'runs/icra_primary_global_matched_w10_128eval_strict_v2/seed{seed}/model.pt', device)
    nominal = load(TopologyGraphWorldModel(TopologyGraphConfig(hidden_dim=136)),
                   ROOT / f'runs/g2_bt_dpwm_meta_train_z32/seed{seed}_v1/baseline_model.pt', device)
    return {'nominal':nominal, 'carrier':carrier, 'full':full,
            'selective':SelectiveInterventionRollout(full, carrier).eval(), 'global':global_model}, None
def historical_models(seed, device):
    cfg = read(ROOT / 'config/experiment/g2_ipwm_d3_physics_spectrum_seed27_audit_v1.yaml')
    path = ('runs/g2_r0_physical_context_residual/seed27_confirmation_v1/model.pt' if seed == 27 else
            f'runs/g2_r0_physical_context_residual_extension/seed{seed}_v1/model.pt')
    full = load(build_strict(cfg, device), ROOT / path, device)
    carrier = copy.deepcopy(full)
    with torch.no_grad():
        for name in ['geometric_object_head','intervention_object_head']:
            for p in getattr(carrier, name).parameters(): p.zero_()
    acfg = read(ROOT / cfg['matched_adapter_config'])
    apath = ROOT / cfg['matched_adapter_run_template'].format(seed=seed) / 'bt_adapter.pt'
    a1, a2 = load(make_adapter(acfg,device), apath,device), load(make_adapter(acfg,device),apath,device)
    fw = FewShotProjectedModel(full,a1,base_uses_topology=True).to(device).eval()
    cw = FewShotProjectedModel(carrier,a2,base_uses_topology=True).to(device).eval()
    sel = SelectiveInterventionRollout(fw,cw).eval()
    encoder = load(UncertainPhysicalContextEncoder(hidden_dim=96),
                   ROOT / cfg['context_encoder_run_template'].format(seed=seed) / 'context_encoder.pt',device)
    return {'carrier':cw, 'full':fw, 'selective':sel}, (full,carrier,sel,encoder,cfg)

def collect(domain, query, calibration=False):
    path = OUT / 'data' / f'q{query}_{domain.domain_id}_{"cal" if calibration else "test"}.pt'
    if path.exists(): return torch.load(path,weights_only=False)
    targets = load_target_split(ROOT / 'config/splits/push_targets_5dof_v1.yaml')
    idx = ['D2','D3','D4'].index(domain.topology)*7 + PHYSICS.index(domain.residual_name)
    data = collect_push_domains((domain,), trajectories_per_domain=1 if calibration else 12,
        steps=25 if calibration else 150, seed=query*10000+idx*100+(0 if calibration else 50),
        targets=tuple(t.as_array() for t in (targets.calibration if calibration else targets.evaluation)),
        excitation='active' if calibration else 'goal', goal_exploration_std=.08,
        block_initial_xy=np.asarray([.24,.10]), xml_path=ROOT/'sim/assets/arm_push.xml')
    path.parent.mkdir(parents=True,exist_ok=True); torch.save(data,path)
    return data

@torch.no_grad()
def evaluate(models, trajectories, domain, device):
    rows = []; surgery=TopologySurgery()
    for horizon in [10,25,50]:
        pairs=[(ti,start) for ti in range(len(trajectories)) for start in range(0,151-horizon,horizon)]
        initial=torch.stack([trajectories[i].states[s] for i,s in pairs]).to(device)
        actions=torch.stack([trajectories[i].actions[s:s+horizon] for i,s in pairs]).to(device)
        truth=torch.stack([trajectories[i].states[s+horizon] for i,s in pairs]).to(device)
        mask,angle=_damage_tensors([domain.damage]*len(pairs),device)
        free=torch.cat([1-mask,1-mask],-1); zero=torch.zeros_like(mask)
        outputs={}
        for name,model in models.items():
            pred=initial.clone(); hidden=None
            for t in range(horizon):
                pred,hidden=model.step(pred,actions[:,t],zero if name=='nominal' else mask,
                                       zero if name=='nominal' else angle,hidden)
                pred=surgery.project_state(pred,mask,angle)
            if not torch.isfinite(pred).all(): raise ValueError(f'Nonfinite predictions {name} {domain.domain_id}')
            outputs[name]=pred
        carrier=outputs['carrier']
        for name,pred in outputs.items():
            e=(pred-truth).square()
            vals={'object_mse':e[:,10:].mean(-1),'object_xy_mse':e[:,10:12].mean(-1),
                'object_velocity_mse':e[:,12:14].mean(-1),'free_mse':(e[:,:10]*free).sum(-1)/free.sum(-1),
                'robot_carrier_max':(pred[:,:10]-carrier[:,:10]).abs().amax(-1),
                'lock_violation':surgery.constraint_violation(pred,mask,angle)}
            vals={k:v.cpu().numpy() for k,v in vals.items()}
            for n,(ti,start) in enumerate(pairs):
                rows.append(dict(domain=domain.domain_id,horizon=horizon,method=name,trajectory=ti,start=start,
                    contact=bool(trajectories[ti].contact_mask[start:start+horizon].any()),
                    **{k:float(v[n]) for k,v in vals.items()}))
    return rows

def freeze():
    paths=[Path(__file__), ROOT/'sim/assets/arm_push.xml',
        ROOT/'config/experiment/icra_primary_d2d4_decision_development_3seed_v1.yaml',
        ROOT/'config/experiment/g2_ipwm_d3_physics_spectrum_seed27_audit_v1.yaml']
    for seed in [7,17,27]:
        for run in ['icra_primary_decision_full_w10_128eval_strict_v2','icra_primary_global_matched_w10_128eval_strict_v2']:
            paths.append(ROOT/f'runs/{run}/seed{seed}/model.pt')
        paths.append(ROOT/f'runs/g2_bt_dpwm_meta_train_z32/seed{seed}_v1/baseline_model.pt')
    for seed in [27,37,47]:
        paths.extend([ROOT/('runs/g2_r0_physical_context_residual/seed27_confirmation_v1/model.pt' if seed==27 else
            f'runs/g2_r0_physical_context_residual_extension/seed{seed}_v1/model.pt'),
            ROOT/f'runs/g2_bt_dpwm_z69_adapter_z70/seed{seed}_v1/bt_adapter.pt',
            ROOT/f'runs/g2_bt_dpwm_context_encoder_z65/seed{seed}_v1/context_encoder.pt'])
    spec=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        purpose='Prospective query replication and model-version bridge; no training or hyperparameter changes.',
        final_seeds=[7,17,27], historical_seeds=[27,37,47],queries=[91101,91102],
        topologies_final=['D2','D3','D4'],topologies_historical=['D3'],physics=PHYSICS,
        trajectories_per_domain=12,steps=150,horizons=[10,25,50],primary_horizon=50,
        targeted_stratum=['high_damping','mixed_composition','mixed_unseen'],
        primary_metrics=['object_xy_rmse','object_velocity_rmse','robot_carrier_deviation'],
        secondary_metric='Historical mixed-unit object-state RMSE; not an SE2 position error.',
        inference='Report complete spectrum. Training-seed replication, not independent windows. Two fresh query archives on previously inspected domains; no unseen-domain claim.',
        hashes={str(p.relative_to(ROOT)):digest(p) for p in paths})
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'protocol.json'
    if p.exists(): raise FileExistsError('Frozen protocol already exists')
    p.write_text(json.dumps(spec,indent=2),encoding='utf-8'); print(p,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true')
    p.add_argument('--family',choices=['final','historical']);p.add_argument('--query',type=int,choices=[91101,91102]);a=p.parse_args()
    if a.freeze: freeze(); return
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,h in protocol['hashes'].items():
        if digest(ROOT/path)!=h: raise ValueError(f'Frozen source changed: {path}')
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for seed in ([7,17,27] if a.family=='final' else [27,37,47]):
        models,context_parts=(final_models if a.family=='final' else historical_models)(seed,device)
        for topology in (['D2','D3','D4'] if a.family=='final' else ['D3']):
            for physics in PHYSICS:
                domain=DomainSpec(topology,physics,'test'); dest=OUT/f'{a.family}_q{a.query}_s{seed}_{domain.domain_id}.json'
                if dest.exists(): continue
                traj=collect(domain,a.query); context_norm=None
                if context_parts:
                    full,carrier,sel,encoder,cfg=context_parts;cal=collect(domain,a.query,True)[0]
                    mask,_=_damage_tensors([domain.damage],device)
                    with torch.no_grad():
                        mean,_=encoder(cal.states[None].to(device),cal.actions[None].to(device),mask,return_uncertainty=True)
                    context=mean[0]*float(cfg['context_posterior_scale']);context_norm=float(context.norm())
                    full.set_intervention_context(context);carrier.set_intervention_context(context);sel.set_residual_context(context)
                    models['routed']=sel if context_norm>=1.2 else models['carrier']
                rows=evaluate(models,traj,domain,device)
                dest.write_text(json.dumps(dict(family=a.family,query=a.query,seed=seed,domain=domain.domain_id,
                    context_norm=context_norm,device=str(device),rows=rows,
                    trajectory_metadata=[t.metadata for t in traj]),indent=2),encoding='utf-8')
                print(f'{a.family} q{a.query} s{seed} {domain.domain_id}: {len(rows)} rows',flush=True)
    print('COMPLETE',a.family,a.query,flush=True)
if __name__=='__main__': main()
