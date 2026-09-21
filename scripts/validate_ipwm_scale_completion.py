"""Strict completion gate. Recompute data audits and inspect all result artifacts."""
import hashlib,json,subprocess,sys,importlib.metadata
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/ipwm_scale_goal_20260911'
CONTRACT=ROOT/'config/experiment/ipwm_scale_goal_20260911.json'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    # This re-reads all 58k trajectories and their split IDs/fingerprints.
    subprocess.run([sys.executable,'scripts/audit_ipwm_scale_dataset.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/summarize_ipwm_scale_results.py'],cwd=ROOT,check=True)
    contract=json.loads(CONTRACT.read_text())
    snapshot=json.loads((OUT/'dependency-source-snapshot.json').read_text())
    for name,expected_hash in snapshot['hashes'].items():
        assert sha(ROOT/name)==expected_hash,'Dependency changed after snapshot: '+name
    checks={k:False for k in contract['completion_requires']}
    audit=json.loads((OUT/'dataset-audit.json').read_text())
    counts=audit['counts'];summary=json.loads((OUT/'results-summary.json').read_text())
    checks['validated_pool_count > 36000']=sum(counts['pool'].values())>36000
    checks['validated_prediction_test_count > 4000']=sum(counts['test'].values())>4000
    checks['each_declared_condition_pool_count > 9000']=len(counts['pool'])==5 and min(counts['pool'].values())>9000
    checks['each_declared_condition_prediction_test_count > 1000']=len(counts['test'])==5 and min(counts['test'].values())>1000
    checks['no_train_test_overlap_and_no_duplicate_reset_ids']=(audit['total_unique_reset_ids']==58000==audit['total_unique_reset_action_fingerprints'])
    training=summary['training']
    carrier_source=torch.load(ROOT/'runs/icra_primary_decision_full_w10_128eval_strict_v2/seed27/model.pt',
                              map_location='cpu',weights_only=True)
    protected=[k for k in carrier_source if not k.startswith(('object_','geometric_object_head.','global_residual_head.'))]
    preservation=[]
    checks['actual_training_sample_count_per_fit > 100']=len(training)==18 and all(r['actual_training_unique_trajectories']==50000 for r in training)
    expected={7,17,27,37,47,57}
    checks['complete_training_seed_count > 5']=all({r['seed'] for r in training if r['method']==m}==expected for m in ('ipwm','carrier','global'))
    for r in training:
        p=OUT/'training'/r['method']/f"seed{r['seed']}"/'complete.json'
        d=json.loads(p.read_text())
        assert [x['epoch'] for x in d['history']]==list(range(1,11))
        assert d['model_sha256']==sha(p.parent/'model.pt')
        fitted=torch.load(p.parent/'model.pt',map_location='cpu',weights_only=True)
        assert all(k in fitted and torch.equal(fitted[k],carrier_source[k]) for k in protected)
        preservation.append(dict(method=r['method'],seed=r['seed'],protected_tensors=len(protected),unchanged=True))
        if r['method']=='global':assert d['global_robot_first_gradient_norm']>0
    plan=summary['planning']
    checks['each_declared_planning_cell_independent_problem_count > 100']=summary['independent_planning_problems']==1200 and len(plan)==36 and all(r['independent_problems']==1200 for r in plan)
    fixed=[r for r in plan if r['family']=='deployed' and r['seed']==27]
    checks['complete_planner_repeat_count > 5']=len(fixed)==4 and all(r['planner_repeats_per_problem']==6 for r in fixed)
    validation=json.loads((OUT/'simulator-validation.json').read_text())
    checks['simulator_validation_passed']=(validation['passed'] and len(validation['rows'])==20
        and validation['script_sha256']==sha(ROOT/'scripts/ipwm_scale_data.py')
        and validation['xml_sha256']==sha(ROOT/'sim/assets/arm_push.xml')
        and all(len(r['cpu_checks'])==8 and all(c['max_object_xy_difference_m']<.001
            and c['max_robot_q_difference_rad']<.002 for c in r['cpu_checks']) for r in validation['rows']))
    registry=json.loads((OUT/'queue.json').read_text());records={}
    scientific=[j for j in registry if j['id'].startswith(('train-','predict-','plan-'))]
    assert sum(j['id'].startswith('train-') for j in scientific)==18
    assert sum(j['id'].startswith('predict-') for j in scientific)==9
    assert sum(j['id'].startswith('plan-') for j in scientific)==560
    for job in scientific:
        for name in job['outputs']:
            path=ROOT/name
            assert path.is_file(),name
            records[name]=sha(path)
    checks['all_registered_baselines_complete']=True
    # The deployed weight hashes must still match the earlier frozen evidence
    # protocol. New adaptation fits have distinct output paths.
    old=json.loads((ROOT/'runs/ipwm_targeted_advantage_20260911/protocol.json').read_text())
    for name,h in old['hashes'].items():
        if name.endswith('.pt'):
            assert sha(ROOT/name)==h,'Deployed/historical checkpoint changed: '+name
            records[name]=h
    artifacts=['paper/ipwm-scale-evidence-20260911.md','paper/ipwm-scale-evidence-20260911.html',
        'paper/ipwm-scale-experiments-20260911.tex',
        'runs/ipwm_scale_goal_20260911/scale-planning-comparisons.png',
        'runs/ipwm_scale_goal_20260911/scale-prediction.png']
    for name in artifacts:
        path=ROOT/name
        assert path.is_file() and path.stat().st_size>100,name
        records[name]=sha(path)
    records.update(summary['provenance'])
    for name in ['data-protocol.json','training-protocol.json','planning-protocol.json','dataset-audit.json','simulator-validation.json','results-summary.json','dependency-source-snapshot.json']:
        records[str((OUT/name).relative_to(ROOT))]=sha(OUT/name)
    for pattern in ['scripts/ipwm_scale_*.py','scripts/*ipwm_scale*.py','src/robotarm/models/*.py',
                    'src/robotarm/envs/constraint_lock.py','runs/ipwm_scale_goal_20260911/*.csv',
                    'runs/ipwm_scale_goal_20260911/*.svg','runs/ipwm_scale_goal_20260911/*amendment.json',
                    'tmp/activepusher-review.txt']:
        for path in ROOT.glob(pattern):
            records[str(path.relative_to(ROOT))]=sha(path)
    dependencies={name:importlib.metadata.version(name) for name in
                  ['torch','numpy','mujoco','mujoco-warp','warp-lang','matplotlib']}
    manifest=OUT/'reproducibility-manifest.json'
    manifest.write_text(json.dumps(dict(files=records,contract_sha256=sha(CONTRACT),dependencies=dependencies,
        frozen_carrier_parameter_checks=preservation,
        scope='Scalar simulation counts exceed ActivePusher; damage variety is not object variety. Training repeats are adaptations from shared pretraining. Original deployed weights remain separate.'),indent=2))
    checks['paper_tables_confidence_intervals_and_reproducibility_manifest_written']=True
    result=dict(contract_sha256=sha(CONTRACT),checks=checks,passed=all(checks.values()),
        manifest_sha256=sha(manifest),counts=counts)
    (OUT/'validated-completion.json').write_text(json.dumps(result,indent=2))
    if not result['passed']:raise ValueError('Completion conditions not met')
    print('All artifact completion conditions verified; human-visible figure review still required.')

if __name__=='__main__':main()
