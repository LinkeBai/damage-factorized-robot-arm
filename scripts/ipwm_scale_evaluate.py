"""Prediction audit with separate SI metrics and immutable test rows."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from ipwm_scale_train import dataset,build,OUT
from audit_ipwm_targeted_advantage import final_models,digest
from audit_ipwm_fresh_action_selection import NominalProjection

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['adapted','deployed'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(2);device=torch.device('cuda')
    audit=json.loads((OUT/'dataset-audit.json').read_text());assert audit['passed']
    s,u,l,ids=dataset('test',device)
    sources={}
    if a.family=='adapted':
        models={}
        for name in ('ipwm','carrier','global'):
            folder=OUT/'training'/name/f'seed{a.seed}'
            complete=json.loads((folder/'complete.json').read_text())
            assert complete['model_sha256']==digest(folder/'model.pt')
            model=build(name,a.seed,device)
            model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True))
            models[name]=model.eval();sources[name]=digest(folder/'model.pt')
        old,_=final_models(27,device)
        models['nominal_projected']=NominalProjection(old['nominal'])
    else:
        old,_=final_models(a.seed,device)
        models={'ipwm':old['full'],'carrier':old['carrier'],'global':old['global'],
                'nominal_projected':NominalProjection(old['nominal'])}
        for name,run in [('ipwm','icra_primary_decision_full_w10_128eval_strict_v2'),('global','icra_primary_global_matched_w10_128eval_strict_v2')]:
            sources[name]=digest(ROOT/f'runs/{run}/seed{a.seed}/model.pt')
    dest=OUT/'prediction'/a.family/f'seed{a.seed}';dest.mkdir(parents=True,exist_ok=True)
    results={};began=time.time()
    mask=torch.nn.functional.one_hot(l,5).float();angles=s[:,0,:5]*mask
    for name,model in models.items():
        records={h:[] for h in [10,25,50]}
        for b in range(0,len(s),512):
            x=s[b:b+512,0];hidden=None;mm=mask[b:b+512];aa=angles[b:b+512]
            for t in range(50):
                x,hidden=model.step(x,u[b:b+512,t//10],mm,aa,hidden)
                if not torch.isfinite(x).all():raise ValueError(f'Nonfinite {name} at {t}')
                if t+1 in records:records[t+1].append(x.cpu().numpy())
        for h,values in records.items():
            pred=np.concatenate(values);truth=s[:,h].cpu().numpy();error=pred-truth
            results[f'{name}_h{h}_xy_sq_m']=np.mean(error[:,10:12]**2,axis=1)
            results[f'{name}_h{h}_velocity_sq_mps']=np.mean(error[:,12:14]**2,axis=1)
            free=1-mask.cpu().numpy()
            results[f'{name}_h{h}_free_q_sq_rad']=(error[:,:5]**2*free).sum(1)/4
            results[f'{name}_h{h}_lock_q_rad']=np.max(np.abs((pred[:,:5]-angles.cpu().numpy())*mask.cpu().numpy()),axis=1)
        print('evaluated',a.family,a.seed,name,'seconds',round(time.time()-began,1),flush=True)
    results.update(reset_id=np.asarray(ids),locked_joint=l.cpu().numpy())
    np.savez_compressed(dest/'rows.npz',**results)
    summary={key:float(np.sqrt(value.mean())) for key,value in results.items() if '_sq_' in key}
    (dest/'complete.json').write_text(json.dumps(dict(family=a.family,seed=a.seed,
        independent_test_trajectories=len(s),source_hashes=sources,rows_sha256=digest(dest/'rows.npz'),
        dataset_audit_sha256=digest(OUT/'dataset-audit.json'),script_sha256=digest(__file__),
        rmse=summary,seconds=time.time()-began),indent=2))

if __name__=='__main__':main()
