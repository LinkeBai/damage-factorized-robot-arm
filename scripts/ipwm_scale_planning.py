"""Matched candidate-budget receding-horizon simulation; no oracle selection."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import mujoco
from ipwm_scale_data import make_model,PROFILES
from ipwm_scale_train import build,OUT
from audit_ipwm_targeted_advantage import final_models,digest
from audit_ipwm_fresh_action_selection import NominalProjection
from generate_primary_sequence_candidates import JOINTS,CONTACT_QPOS

METHODS=('ipwm','carrier','global','nominal_projected')

def setup(lock,band,n=120):
    models=[];datas=[];goals=[];ids=[]
    for i in range(n):
        rng=np.random.default_rng(np.random.SeedSequence([9112026,404,lock,band,i]))
        m,eq=make_model(lock,PROFILES[i%4]);d=mujoco.MjData(m)
        q=CONTACT_QPOS+rng.normal(0,.015,5);v=rng.normal(0,.01,5);v[lock]=0
        for j,name in enumerate(JOINTS):
            d.qpos[int(m.joint(name).qposadr[0])]=q[j]
            d.qvel[int(m.joint(name).dofadr[0])]=v[j]
        for name in ('block_x','block_y'):
            d.qpos[int(m.joint(name).qposadr[0])]=rng.uniform(-.004,.004)
        m.eq_data[eq,0]=q[lock]
        mujoco.mj_forward(m,d)
        distance=rng.uniform(*((.04,.065) if band==0 else (.065,.09)))
        theta=rng.uniform(-np.pi/4,np.pi/4)
        goals.append(d.body('block').xpos[:2]+distance*np.array([np.cos(theta),np.sin(theta)]))
        ids.append(f'planning-D{lock+1}-B{band}-{i:04d}')
        models.append(m);datas.append(d)
    return models,datas,np.asarray(goals),ids

def state(m,d):
    return np.concatenate(([d.qpos[int(m.joint(j).qposadr[0])] for j in JOINTS],
        [d.qvel[int(m.joint(j).dofadr[0])] for j in JOINTS],
        d.body('block').xpos[:2],
        [d.qvel[int(m.joint(j).dofadr[0])] for j in ('block_x','block_y')])).astype(np.float32)

@torch.no_grad()
def select(model,states,actions,goals,lock,diagnosed_angles=None):
    n,c=actions.shape[:2];device=torch.device('cuda')
    initial=torch.tensor(np.repeat(states,c,axis=0),device=device)
    sequence=torch.tensor(actions.reshape(n*c,5,5),device=device)
    if diagnosed_angles is None:
        diagnosed_angles=states[:,lock]
    fixed_angles=torch.tensor(np.repeat(diagnosed_angles,c),device=device)
    outputs=[]
    for start in range(0,len(initial),2048):
        x=initial[start:start+2048];u=sequence[start:start+2048]
        mask=torch.zeros((len(x),5),device=device);mask[:,lock]=1
        angles=torch.zeros_like(mask)
        angles[:,lock]=fixed_angles[start:start+len(x)]
        hidden=None
        for t in range(50):
            x,hidden=model.step(x,u[:,t//10],mask,angles,hidden)
        if not torch.isfinite(x).all():raise ValueError('Nonfinite planning prediction')
        outputs.append(x[:,10:12].cpu().numpy())
    predicted=np.concatenate(outputs).reshape(n,c,2)
    return np.linalg.norm(predicted-goals[:,None,:],axis=-1).argmin(1)

def load_model(family,method,seed):
    device=torch.device('cuda')
    if family=='adapted' and method!='nominal_projected':
        folder=OUT/'training'/method/f'seed{seed}'
        complete=json.loads((folder/'complete.json').read_text())
        assert complete['model_sha256']==digest(folder/'model.pt')
        model=build(method,seed,device)
        model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True))
        return model.eval(),digest(folder/'model.pt')
    old,_=final_models(seed if family=='deployed' else 27,device)
    if method=='nominal_projected':return NominalProjection(old['nominal']).eval(),'common-nominal-27' if family=='adapted' else f'nominal-{seed}'
    return old['full' if method=='ipwm' else method],f'frozen-deployed-family-{seed}-{method}'

def freeze():
    path=OUT/'planning-protocol.json'
    if path.exists():raise FileExistsError(path)
    spec=dict(script_sha256=digest(__file__),methods=METHODS,adapted_seeds=[7,17,27,37,47,57],
        deployed_seeds=[7,17,27],deployed_checkpoint27_planner_repeats=[0,1,2,3,4,5],locks=[1,2,3,4,5],distance_bands_m=[[.04,.065],[.065,.09]],
        problems_per_cell=120,independent_problems_total=1200,
        repeat_interpretation='Six adapted-model fit seeds each paired with one candidate RNG repeat; separately the deployed checkpoint27 receives six planner repeats with fixed weights. Deployed seeds7/17 each receive repeat0 as a training-seed sensitivity check. All share 1200 reset/goal problems; repeated runs are not independent starts.',
        candidate_budget=128,prediction_horizon_steps=50,segments=5,steps_per_segment=10,
        replans=5,executed_steps_per_replan=10,total_executed_steps=50,
        primary_metrics=['terminal_distance_m','success_at_30mm'],secondary_metrics=['lock_position_deviation_rad','compute_seconds'],
        state_feedback='Exact simulator state each replan; no simulated visual noise claim.',
        scope='Contact-neighborhood box pushing across damage and four physics profiles; distance bands are strata, not ActivePusher task replicas.',
        fairness='Identical starts/goals/command candidates and simulation steps across methods within repeat; model-predicted costs select actions, never true candidate costs.',
        inference='Paired reset bootstrap within fit seed, all fit seed results separately, aggregate across six fits; repeats do not inflate independent problem count.',
        termination='Execute all five replans; success determined at final step, never stop on favorable intermediate state.',
        data_source_sha256=digest(ROOT/'scripts/ipwm_scale_data.py'))
    path.write_text(json.dumps(spec,indent=2))

def run(a):
    spec=json.loads((OUT/'planning-protocol.json').read_text());assert spec['script_sha256']==digest(__file__)
    torch.set_num_threads(2)
    lock=a.lock-1
    model,source=load_model(a.family,a.method,a.seed)
    ms,ds,goals,ids=setup(lock,a.band)
    initial=np.stack([state(m,d) for m,d in zip(ms,ds)])
    trajectories=[initial];commands=[];choices=[];began=time.time()
    for step in range(5):
        rng=np.random.default_rng(np.random.SeedSequence([9112026,505,a.repeat,lock,a.band,step]))
        candidates=rng.uniform(-.8,.8,(120,128,5,5)).astype(np.float32);candidates[:,:,:,lock]=0
        current=trajectories[-1]
        chosen=select(model,current,candidates,goals,lock,initial[:,lock])
        action=candidates[np.arange(120),chosen,0]
        # CPU executes only the selected first segment; every subsequent
        # decision sees that method's own resulting state.
        for m,d,u in zip(ms,ds,action):
            d.ctrl[:]=u;mujoco.mj_step(m,d,nstep=10);mujoco.mj_forward(m,d)
        trajectories.append(np.stack([state(m,d) for m,d in zip(ms,ds)]))
        commands.append(action);choices.append(chosen)
        print(a.family,a.method,a.seed,a.lock,a.band,'replan',step+1,round(time.time()-began,1),flush=True)
    trajectory=np.stack(trajectories,1)
    endpoint=np.linalg.norm(trajectory[:,-1,10:12]-goals,axis=1)
    dest=OUT/'planning'/a.family/f'seed{a.seed}'/f'repeat{a.repeat}'/a.method/f'D{a.lock}-B{a.band}'
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'rows.npz',reset_id=np.asarray(ids),goal=goals,states=trajectory,
        chosen_commands=np.stack(commands,1),chosen_indices=np.stack(choices,1),endpoint_m=endpoint,
        success=endpoint<.03,lock_deviation_rad=np.abs(trajectory[:,:,lock]-initial[:,None,lock]).max(1))
    (dest/'complete.json').write_text(json.dumps(dict(family=a.family,method=a.method,seed=a.seed,repeat=a.repeat,lock=a.lock,band=a.band,
        source=source,independent_problems=120,mean_endpoint_m=float(endpoint.mean()),success_rate=float((endpoint<.03).mean()),
        seconds=time.time()-began,rows_sha256=digest(dest/'rows.npz'),protocol_sha256=digest(OUT/'planning-protocol.json')),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--family',choices=['adapted','deployed']);p.add_argument('--method',choices=METHODS);p.add_argument('--seed',type=int);p.add_argument('--repeat',type=int,choices=range(6));p.add_argument('--lock',type=int,choices=range(1,6));p.add_argument('--band',type=int,choices=[0,1]);a=p.parse_args()
    if a.freeze:freeze()
    else:run(a)
