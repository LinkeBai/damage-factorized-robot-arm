"""Check batched evaluator against established evaluator and audit global support."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np,torch
from scripts.audit_ipwm_targeted_advantage import final_models,DomainSpec,OUT
from scripts.evaluate_g2_r0_core_metrics import evaluate
from robotarm.training.topology_surgery_gate import _damage_tensors
torch.set_num_threads(2);device=torch.device('cpu')
trajectories=torch.load(OUT/'data/q91101_D2__nominal_test.pt',weights_only=False)
models,_=final_models(7,device)
old=evaluate({'carrier':models['carrier'],'selective':models['selective']},trajectories,
             DomainSpec('D2','nominal','test'),[10,25,50],device)
new=json.loads((OUT/'final_q91101_s7_D2__nominal.json').read_text())['rows']
checks=[]
for row in old:
    subset=[r for r in new if r['horizon']==row['horizon'] and r['method']==row['method']]
    for legacy,fresh in [('object_rmse','object_mse'),('free_rmse','free_mse')]:
        value=np.sqrt(np.mean([r[fresh] for r in subset]))
        checks.append(dict(method=row['method'],horizon=row['horizon'],metric=legacy,
            old_cpu=row[legacy],new_gpu=float(value),close=bool(np.isclose(row[legacy],value,rtol=2e-5,atol=1e-7))))
support=[]
for seed in [7,17,27]:
    ms,_=final_models(seed,device);m=ms['global'];state=trajectories[0].states[50:51].clone()
    mask,angle=_damage_tensors([DomainSpec('D2','nominal','test').damage],device)
    hidden=None
    for t in range(10): state,hidden=m.step(state,trajectories[0].actions[50+t:51+t],mask,angle,hidden)
    loss=(state[:,10:]-trajectories[0].states[60:61,10:]).square().mean();loss.backward()
    layer=m.global_residual_head[-1]
    support.append(dict(seed=seed,robot_weight_abs_max=float(layer.weight[:10].detach().abs().max()),
        robot_bias_abs_max=float(layer.bias[:10].detach().abs().max()),
        robot_weight_gradient_abs_max=float(layer.weight.grad[:10].abs().max()),
        robot_bias_gradient_abs_max=float(layer.bias.grad[:10].abs().max()),
        object_weight_gradient_abs_max=float(layer.weight.grad[10:].abs().max())))
result=dict(evaluator_checks=checks,all_checks_pass=all(r['close'] for r in checks),
    global_support=support,interpretation='Frozen global checkpoints have zero robot-output rows. This comparator does not exercise robot-state corruption. Object-only losses have zero gradient to those rows in this audited rollout because robot-to-object paths detach. No training performed.')
(OUT/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert result['all_checks_pass']
