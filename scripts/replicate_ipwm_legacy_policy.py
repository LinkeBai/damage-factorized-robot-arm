"""Larger fresh-query replication of the historical collection policy, labelled explicitly."""
import argparse,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np,torch
from scripts import audit_ipwm_targeted_advantage as audit
from scripts import run_push_benchmark as collector
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
from robotarm.models.topology_surgery import TopologySurgery
OUT=audit.OUT/'legacy_policy_100'
def old_waypoints(block,target):
    return np.asarray([block[0]-.03,block[1],.025]),np.asarray([target[0]+.03,target[1],.025])
original_solver=collector.solve_reach_reference;ik_cache={}
def memo(t,r,*,locked_joints=None,config=None):
    key=(np.asarray(t).tobytes(),np.asarray(r).tobytes(),tuple(sorted((locked_joints or {}).items())),repr(config))
    if key not in ik_cache:ik_cache[key]=original_solver(t,r,locked_joints=locked_joints,config=config)
    q,e=ik_cache[key];return q.copy(),e
collector.solve_reach_reference=memo
collector.directional_push_waypoints=old_waypoints

def collect(physics):
    path=OUT/f'{physics}.pt'
    if path.exists():return torch.load(path,weights_only=False)
    domain=audit.DomainSpec('D3',physics,'test')
    targets=load_target_split(ROOT/'config/splits/push_targets_5dof_v1.yaml')
    data=collector.collect_push_domains((domain,),trajectories_per_domain=100,steps=150,
        seed=911030000+audit.PHYSICS.index(physics)*1000,
        targets=tuple(t.as_array() for t in targets.evaluation),excitation='goal',goal_exploration_std=.08,
        block_initial_xy=np.asarray([.24,.10]),xml_path=ROOT/'sim/assets/arm_push.xml')
    torch.save(data,path);return data

@torch.no_grad()
def evaluate(models,traj,domain,device):
    pairs=[(ti,start) for ti in range(len(traj)) for start in [0,50,100]]
    initial=torch.stack([traj[i].states[s] for i,s in pairs]).to(device)
    actions=torch.stack([traj[i].actions[s:s+50] for i,s in pairs]).to(device)
    truth=torch.stack([traj[i].states[s+50] for i,s in pairs]).to(device)
    mask,angle=_damage_tensors([domain.damage]*len(pairs),device);zero=torch.zeros_like(mask)
    surgery=TopologySurgery();free=torch.cat([1-mask,1-mask],-1);outputs={}
    for name,model in models.items():
        state=initial.clone();hidden=None
        for t in range(50):
            state,hidden=model.step(state,actions[:,t],zero if name=='nominal' else mask,zero if name=='nominal' else angle,hidden)
            state=surgery.project_state(state,mask,angle)
        if not torch.isfinite(state).all():raise ValueError('Nonfinite predictions')
        outputs[name]=state
    rows=[]
    for name,state in outputs.items():
        e=(state-truth).square();delta=state[:,:10]-outputs['carrier'][:,:10]
        vals={'object_mse':e[:,10:].mean(-1),'object_xy_mse':e[:,10:12].mean(-1),
            'object_velocity_mse':e[:,12:].mean(-1),'free_mse':(e[:,:10]*free).sum(-1)/free.sum(-1),
            'robot_carrier_max':delta.abs().amax(-1),'lock_violation':surgery.constraint_violation(state,mask,angle)}
        vals={k:v.cpu().numpy() for k,v in vals.items()}
        for n,(ti,s) in enumerate(pairs):
            rows.append(dict(method=name,trajectory=ti,start=s,horizon=50,contact=bool(traj[ti].contact_mask[s:s+50].any()),
                **{k:float(v[n]) for k,v in vals.items()}))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--collect-only',action='store_true')
    p.add_argument('--family',choices=['historical','final']);args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);protocol=OUT/'protocol.json'
    if args.freeze:
        if protocol.exists():raise FileExistsError(protocol)
        protocol.write_text(json.dumps(dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            script_sha256=audit.digest(__file__),query_seed=91103,trajectories_per_condition=100,
            physics=audit.PHYSICS,topology='D3',horizon=50,steps=150,
            families={'historical':[27,37,47],'final':[7,17,27]},
            rationale='Historical cache results reproduced; git diff 9035efb..HEAD shows approach policy change. Test old policy with NEW stochastic action trajectories, retain all seven conditions. No new training, no tuning.',
            limitation='Historical fixed-side approach is not direction-aware goal recovery. This is prediction/mechanism evidence, never task-success evidence.',
            primary_stratum=['high_damping','mixed_composition','mixed_unseen'],
            primary_metrics=['object_xy_rmse','object_velocity_rmse','robot_carrier_deviation'],
            secondary_metric='mixed-unit object-state RMSE for historical comparability',
            provenance_protocol_sha256=audit.digest(audit.OUT/'protocol.json')),indent=2));return
    frozen=json.loads(protocol.read_text())
    assert frozen['script_sha256']==audit.digest(__file__)
    for key,h in json.loads((audit.OUT/'protocol.json').read_text())['hashes'].items():assert audit.digest(ROOT/key)==h
    torch.set_num_threads(2)
    if args.collect_only:
        for phy in audit.PHYSICS:collect(phy);print('collected legacy 100',phy,flush=True)
        return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for seed in frozen['families'][args.family]:
        models,parts=(audit.historical_models if args.family=='historical' else audit.final_models)(seed,device)
        for phy in audit.PHYSICS:
            dest=OUT/f'{args.family}_s{seed}_{phy}.json'
            if dest.exists():continue
            data=collect(phy);domain=audit.DomainSpec('D3',phy,'test');norm=None
            if parts:
                full,carrier,sel,enc,cfg=parts;cal=audit.collect(domain,91102,True)[0]
                mask,_=_damage_tensors([domain.damage],device)
                with torch.no_grad():mean,_=enc(cal.states[None].to(device),cal.actions[None].to(device),mask,return_uncertainty=True)
                context=mean[0]*1.38;norm=float(context.norm());full.set_intervention_context(context)
                carrier.set_intervention_context(context);sel.set_residual_context(context)
                models['routed']=sel if norm>=1.2 else models['carrier']
            rows=evaluate(models,data,domain,device)
            dest.write_text(json.dumps(dict(family=args.family,seed=seed,domain=domain.domain_id,query=91103,
                context_norm=norm,policy='legacy_fixed_side',rows=rows),indent=2))
            print('legacy',args.family,seed,phy,flush=True)
    print('COMPLETE',args.family,flush=True)
if __name__=='__main__':main()
