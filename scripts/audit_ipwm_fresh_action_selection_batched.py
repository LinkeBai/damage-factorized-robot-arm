"""Fresh independent D3 candidate groups; actual final checkpoints, no training."""
import argparse,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np,torch,mujoco
from scripts import generate_primary_sequence_candidates as generator
from scripts import audit_ipwm_targeted_advantage as audit
from robotarm.models.topology_surgery import TopologySurgery
from robotarm.training.decision_focused import load_sequence_candidate_npz,subset_candidate_batch,rollout_candidate_terminal_object,spearman_correlation
OUT=audit.OUT/'fresh_selection_120'
class NominalProjection(torch.nn.Module):
    def __init__(self,base):super().__init__();self.base=base;self.surgery=TopologySurgery()
    def step(self,state,action,mask,angle,hidden):
        state=self.surgery.project_state(state,mask,angle);action=self.surgery.project_action(action,mask)
        pred,h=self.base.step(state,action,torch.zeros_like(mask),torch.zeros_like(angle),hidden)
        return self.surgery.project_state(pred,mask,angle),h
def corrected_arm_state(model,data):
    s=original_state(model,data);s[12:14]=data.body('block').cvel[3:5];return s
original_state=generator.arm_state
def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--generate',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);protocol=OUT/'protocol.json'
    if a.freeze:
        if protocol.exists():raise FileExistsError(protocol)
        protocol.write_text(json.dumps(dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            script_sha256=audit.digest(__file__),generator_sha256=audit.digest(generator.__file__),
            query_seed=91104,episodes=120,groups_per_episode=1,candidates=128,horizon=50,lock='D3',
            models=['nominal_as_previous_evaluator','nominal_zero_topology_plus_projection','carrier','selective','global'],
            seeds=[7,17,27],primary_comparison='selective versus each nominal implementation; carrier and global attribution controls',
            primary_metrics=['selected_true_terminal_distance','top1_regret'],secondary=['success_at_30mm','rank_correlation'],
            normalization='Report relative and absolute gains, paired group bootstrap and each trained seed separately.',
            independence='120 independent reset problems; no oracle-driven later stages. All candidates preserved. All checkpoints frozen.',
            correction='MuJoCo cvel is angular then linear: object velocity uses [3:5]. First-stage object starts at rest.',
            limitation='Selected candidate executed in simulation for 50 steps, not a closed-loop recovery experiment. D3 previously inspected; fresh query only.'),indent=2));return
    spec=json.loads(protocol.read_text());assert spec['script_sha256']==audit.digest(ROOT/'scripts/audit_ipwm_fresh_action_selection.py')
    for key,h in json.loads((audit.OUT/'protocol.json').read_text())['hashes'].items():assert audit.digest(ROOT/key)==h
    archive=OUT/'candidates.npz'
    if a.generate:
        assert not archive.exists()
        generator.arm_state=corrected_arm_state
        sys.argv=[sys.argv[0],'--phase','confirmation','--seed','91104','--episodes-per-lock','120',
            '--replans','1','--output',str(archive)]
        generator.main();return
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch=load_sequence_candidate_npz(archive,device=device,allowed_locked_joints=(2,),split='all')
    true=np.linalg.norm(batch.true_terminal_object.cpu().numpy()-batch.goal.cpu().numpy()[:,None,:],axis=-1)
    initial=np.linalg.norm(batch.initial_state[:,10:12].cpu().numpy()-batch.goal.cpu().numpy(),axis=-1)
    assert len(true)==120 and np.all(initial>.03)
    for seed in [7,17,27]:
        dest=OUT/f'seed{seed}.json'
        if dest.exists():continue
        models,_=audit.final_models(seed,device)
        models={'nominal_original':models['nominal'],'nominal_projected':NominalProjection(models['nominal']),
            'carrier':models['carrier'],'selective':models['full'],'global':models['global']}
        # Final full and selective are numerically identical; checked in the spectrum.
        rows=[]
        with torch.no_grad():
            for name,model in models.items():
                pred=[]
                for start in range(0,120,32):
                    chunk=subset_candidate_batch(batch,slice(start,start+32))
                    pred.append(rollout_candidate_terminal_object(model,chunk).cpu().numpy())
                terminal=np.concatenate(pred);cost=np.linalg.norm(terminal-batch.goal.cpu().numpy()[:,None,:],axis=-1)
                for i in range(120):
                    choice=int(cost[i].argmin());selected=float(true[i,choice]);oracle=float(true[i].min())
                    rows.append(dict(seed=seed,method=name,episode=i,chosen=choice,endpoint=selected,regret=selected-oracle,
                        success=bool(selected<.03),oracle=oracle,initial_distance=float(initial[i]),
                        spearman=spearman_correlation(cost[i],true[i])))
                print('selection',seed,name,flush=True)
        dest.write_text(json.dumps(dict(seed=seed,archive_sha256=audit.digest(archive),rows=rows),indent=2))
    print('COMPLETE selection',flush=True)
if __name__=='__main__':main()
