"""Six-seed matched adaptation study, separate from deployed model evidence."""
import argparse, copy, inspect, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from scripts.audit_ipwm_targeted_advantage import final_models, read, digest
from robotarm.models.block_triangular_dpwm import BlockTriangularDPWM
from robotarm.models.topology_graph_world_model import TopologyGraphConfig
OUT=ROOT/'runs/ipwm_scale_goal_20260911'
METHODS=('ipwm','carrier','global')
SEEDS=(7,17,27,37,47,57)

def dataset(split,device):
    shards=sorted((OUT/'data'/split).glob('D*/*.npz'))
    states=[]; actions=[]; locks=[]; ids=[]
    for path in shards:
        with np.load(path) as a:
            states.append(a['states']);actions.append(a['segment_actions']);locks.append(a['locked_joint']);ids.extend(a['reset_id'].tolist())
    return (torch.tensor(np.concatenate(states),device=device),
            torch.tensor(np.concatenate(actions),device=device),
            torch.tensor(np.concatenate(locks),device=device,dtype=torch.long),ids)

def build(method,seed,device):
    # Common pretrained carrier controls initialization; seeds vary the new
    # residual initialization and minibatch order, not pretraining provenance.
    torch.manual_seed(seed)
    models,_=final_models(27,device)
    carrier=models['carrier']
    if method=='global':
        cfg=read(ROOT/'config/experiment/icra_primary_d2d4_decision_development_3seed_v1.yaml')
        kw={k:v for k,v in cfg.items() if k in inspect.signature(BlockTriangularDPWM).parameters and k!='cfg'}
        kw.update(geometric_object_rank=0,global_residual_rank=10)
        model=BlockTriangularDPWM(TopologyGraphConfig(hidden_dim=136),**kw).to(device)
        common={k:v for k,v in carrier.state_dict().items() if k in model.state_dict()}
        model.load_state_dict(common,strict=False)
    else:
        model=copy.deepcopy(carrier)
        if method=='ipwm':
            model.geometric_object_head[0].reset_parameters()
            torch.nn.init.zeros_(model.geometric_object_head[-1].weight)
            torch.nn.init.zeros_(model.geometric_object_head[-1].bias)
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith('object_') or
                         (method=='ipwm' and name.startswith('geometric_object_head.')) or
                         (method=='global' and name.startswith('global_residual_head.')))
    return model

def batch_rollout(model,s,u,locks,start,horizon):
    rows=torch.arange(len(s),device=s.device)
    state=s[rows,start]; mask=torch.nn.functional.one_hot(locks,5).float()
    angles=s[:,0,:5]*mask
    hidden=None; losses=[]
    scales=state.new_tensor([.1]*5+[1.]*5+[.03]*2+[.1]*2)
    for t in range(horizon):
        action=u[rows,(start+t)//10]
        state,hidden=model.step(state,action,mask,angles,hidden)
        losses.append(((state-s[rows,start+t+1])/scales).square().mean(-1))
    return torch.stack(losses,1).mean(), state

@torch.no_grad()
def validation(model,data):
    s,u,l,_=data; total=0.
    for start in range(0,len(s),512):
        ss=s[start:start+512];uu=u[start:start+512];ll=l[start:start+512]
        _,pred=batch_rollout(model,ss,uu,ll,torch.zeros(len(ss),device=s.device,dtype=torch.long),25)
        total+=(pred[:,10:12]-ss[:,25,10:12]).square().sum().item()
    return total/(len(s)*2)

def freeze():
    path=OUT/'training-protocol.json'
    if path.exists():raise FileExistsError(path)
    p=dict(methods=METHODS,seeds=SEEDS,common_pretraining_seed=27,
        identity='New matched adaptation repetitions; not end-to-end independent pretraining and not the real-robot deployment checkpoints.',
        epochs=10,batch_size=512,learning_rate=.0003,gradient_clip=5.,
        horizon_schedule=[10]*5+[25]*5,validation_selection='minimum validation H25 object xy MSE after each epoch, including initialization',
        loss='Mean rollout full-state squared error normalized by [0.1rad x5, 1rad/s x5, 0.03m x2, 0.1m/s x2]; robot carrier weights frozen in all variants; global robot residual receives direct supervision.',
        training_examples='All 50,000 pool trajectories are consumed once per epoch; one seeded random contiguous window per trajectory per epoch. No replacement sampling.',
        data_audit_sha256=digest(OUT/'dataset-audit.json'),script_sha256=digest(__file__),
        source_checkpoint_sha256=digest(ROOT/'runs/icra_primary_decision_full_w10_128eval_strict_v2/seed27/model.pt'),
        test_access='No test dataset loaded by this script. Test evaluation after all model selection is fixed.')
    path.write_text(json.dumps(p,indent=2));print(path)

def train(method,seed):
    spec=json.loads((OUT/'training-protocol.json').read_text())
    assert spec['script_sha256']==digest(__file__)
    assert spec['data_audit_sha256']==digest(OUT/'dataset-audit.json')
    torch.set_num_threads(2);device=torch.device('cuda')
    s,u,l,ids=dataset('pool',device);val=dataset('validation',device)
    model=build(method,seed,device)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=spec['learning_rate'])
    dest=OUT/'training'/method/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    history=[];best=validation(model,val);best_epoch=0
    began=time.time();seen=set();first_gradient=None;resume_epoch=0
    if (dest/'resume.pt').exists():
        saved=torch.load(dest/'resume.pt',map_location=device,weights_only=False)
        assert saved['protocol_sha256']==digest(OUT/'training-protocol.json')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng'];history=saved['history']
        best=saved['best'];best_epoch=saved['best_epoch'];resume_epoch=saved['epoch']
        first_gradient=saved['first_gradient'];seen=set(saved['seen'])
    else:
        torch.save(model.state_dict(),dest/'model.pt')
    for epoch,horizon in enumerate(spec['horizon_schedule'],1):
        if epoch<=resume_epoch:continue
        order=rng.permutation(len(s));total=0.;n=0
        for b in range(0,len(order),spec['batch_size']):
            ix=order[b:b+spec['batch_size']];seen.update(ix.tolist())
            ix=torch.tensor(ix,device=device)
            starts=torch.tensor(rng.integers(0,51-horizon,len(ix)),device=device)
            optimizer.zero_grad(set_to_none=True)
            loss,_=batch_rollout(model,s[ix],u[ix],l[ix],starts,horizon)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward()
            if first_gradient is None and method=='global':
                first_gradient=float(model.global_residual_head[-1].weight.grad[:10].norm())
                assert first_gradient>0,'Global robot outputs are not supervised'
            torch.nn.utils.clip_grad_norm_(model.parameters(),spec['gradient_clip'])
            optimizer.step();total+=loss.item()*len(ix);n+=len(ix)
        value=validation(model,val)
        if value<best:
            best=value;best_epoch=epoch;torch.save(model.state_dict(),dest/'model.pt')
        history.append(dict(epoch=epoch,horizon=horizon,train_loss=total/n,validation_xy_mse=value))
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
            history=history,best=best,best_epoch=best_epoch,epoch=epoch,first_gradient=first_gradient,
            seen=sorted(seen),protocol_sha256=digest(OUT/'training-protocol.json')),dest/'resume.tmp')
        (dest/'resume.tmp').replace(dest/'resume.pt')
        (dest/'progress.json').write_text(json.dumps(dict(history=history,best_epoch=best_epoch,elapsed=time.time()-began),indent=2))
        print(method,seed,epoch,total/n,value,'seconds',round(time.time()-began,1),flush=True)
    result=dict(method=method,seed=seed,history=history,best_epoch=best_epoch,best_validation_xy_mse=best,
        actual_training_unique_trajectories=len(seen),examples_per_epoch=len(ids),
        model_sha256=digest(dest/'model.pt'),protocol_sha256=digest(OUT/'training-protocol.json'),
        global_robot_first_gradient_norm=first_gradient,seconds=time.time()-began,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    (dest/'complete.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--method',choices=METHODS);p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args()
    if a.freeze:freeze()
    else:train(a.method,a.seed)
