"""Paired collection-policy control on prespecified target physics plus nominal."""
import argparse,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import replicate_ipwm_legacy_policy as legacy
from robotarm.training.controllers import directional_push_waypoints
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import _damage_tensors
import torch,numpy as np
audit=legacy.audit;OUT=audit.OUT/'directional_policy_100'
PHYSICS=['nominal','high_damping','mixed_composition','mixed_unseen']
legacy.collector.directional_push_waypoints=directional_push_waypoints
def collect(phy):
    p=OUT/f'{phy}.pt'
    if p.exists():return torch.load(p,weights_only=False)
    targets=load_target_split(ROOT/'config/splits/push_targets_5dof_v1.yaml')
    traj=legacy.collector.collect_push_domains((audit.DomainSpec('D3',phy,'test'),),
        trajectories_per_domain=100,steps=150,seed=911030000+audit.PHYSICS.index(phy)*1000,
        targets=tuple(t.as_array() for t in targets.evaluation),excitation='goal',goal_exploration_std=.08,
        block_initial_xy=np.asarray([.24,.10]),xml_path=ROOT/'sim/assets/arm_push.xml')
    torch.save(traj,p);return traj
def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--collect-only',action='store_true');p.add_argument('--family',choices=['historical','final']);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);protocol=OUT/'protocol.json'
    if a.freeze:
        if protocol.exists():raise FileExistsError(protocol)
        protocol.write_text(json.dumps(dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            script_sha256=audit.digest(__file__),legacy_script_sha256=audit.digest(legacy.__file__),
            physics=PHYSICS,trajectories_per_condition=100,horizon=50,
            pairing='Same seeds, initial state, targets and stochastic action perturbations as legacy_policy_100; deterministic approach waypoint differs.',
            selection='Three targeted physics were selected before original experiment plus nominal control. All seven remain reported in broad spectrum.',
            limitation='Policy sensitivity of prediction; no closed-loop success or learning-efficiency claim.'),indent=2));return
    spec=json.loads(protocol.read_text());assert spec['script_sha256']==audit.digest(__file__)
    assert spec['legacy_script_sha256']==audit.digest(legacy.__file__)
    torch.set_num_threads(2)
    if a.collect_only:
        for phy in PHYSICS:collect(phy);print('directional 100 collected',phy,flush=True)
        return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for seed in ([27,37,47] if a.family=='historical' else [7,17,27]):
        models,parts=(audit.historical_models if a.family=='historical' else audit.final_models)(seed,device)
        for phy in PHYSICS:
            dest=OUT/f'{a.family}_s{seed}_{phy}.json'
            if dest.exists():continue
            domain=audit.DomainSpec('D3',phy,'test');norm=None
            if parts:
                full,carrier,sel,enc,cfg=parts;cal=audit.collect(domain,91102,True)[0]
                mask,_=_damage_tensors([domain.damage],device)
                with torch.no_grad():mean,_=enc(cal.states[None].to(device),cal.actions[None].to(device),mask,return_uncertainty=True)
                context=mean[0]*1.38;norm=float(context.norm());full.set_intervention_context(context)
                carrier.set_intervention_context(context);sel.set_residual_context(context)
                models['routed']=sel if norm>=1.2 else models['carrier']
            rows=legacy.evaluate(models,collect(phy),domain,device)
            dest.write_text(json.dumps(dict(family=a.family,seed=seed,domain=domain.domain_id,query=91103,
                context_norm=norm,policy='direction_aware',rows=rows),indent=2))
            print('directional',a.family,seed,phy,flush=True)
    print('COMPLETE',a.family,flush=True)
if __name__=='__main__':main()
