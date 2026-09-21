"""Complete-case result tables, paired problem intervals, explicit repeat counts."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/ipwm_scale_goal_20260911'
METHODS=('ipwm','carrier','global','nominal_projected')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def interval(delta):
    rng=np.random.default_rng(20260911)
    boot=[]
    for _ in range(2000):boot.append(float(delta[rng.integers(0,len(delta),len(delta))].mean()))
    return np.quantile(boot,[.025,.975]).tolist()

def main():
    registry=json.loads((OUT/'queue.json').read_text())
    required=[j for j in registry if j['id'].startswith(('train-','predict-','plan-'))]
    for job in required:
        for path in job['outputs']:
            if not (ROOT/path).is_file():raise FileNotFoundError(path)
    predictions=[];conditions=[];training=[];planning=[];comparisons=[];provenance={};all_problem_ids=set()
    for p in sorted((OUT/'training').glob('*/seed*/complete.json')):
        d=json.loads(p.read_text());assert d['model_sha256']==sha(p.parent/'model.pt')
        training.append({k:d[k] for k in ('method','seed','best_epoch','best_validation_xy_mse',
            'actual_training_unique_trajectories','trainable_parameters','global_robot_first_gradient_norm','seconds')})
        provenance[str(p.relative_to(ROOT))]=sha(p)
    for p in sorted((OUT/'prediction').glob('*/seed*/complete.json')):
        d=json.loads(p.read_text());assert d['rows_sha256']==sha(p.parent/'rows.npz')
        with np.load(p.parent/'rows.npz') as a:
            assert len(a['reset_id'])==6000 and len(set(a['reset_id'].tolist()))==6000
            for method in METHODS:
                for horizon in [10,25,50]:
                    predictions.append(dict(family=d['family'],seed=d['seed'],method=method,horizon=horizon,
                        independent_test_trajectories=6000,
                        xy_rmse_m=float(np.sqrt(a[f'{method}_h{horizon}_xy_sq_m'].mean())),
                        velocity_rmse_mps=float(np.sqrt(a[f'{method}_h{horizon}_velocity_sq_mps'].mean())),
                        free_q_rmse_rad=float(np.sqrt(a[f'{method}_h{horizon}_free_q_sq_rad'].mean()))))
                    for lock in range(5):
                        for profile in range(4):
                            selected=(a['locked_joint']==lock)&np.array([f'-P{profile}-' in x for x in a['reset_id']])
                            assert selected.sum()==300
                            conditions.append(dict(family=d['family'],seed=d['seed'],method=method,horizon=horizon,
                                lock=lock+1,profile=['nominal','high_damping','weak_motor','mixed'][profile],
                                independent_trajectories=300,
                                xy_rmse_m=float(np.sqrt(a[f'{method}_h{horizon}_xy_sq_m'][selected].mean())),
                                velocity_rmse_mps=float(np.sqrt(a[f'{method}_h{horizon}_velocity_sq_mps'][selected].mean()))))
        provenance[str(p.relative_to(ROOT))]=sha(p)
    # Aggregate by model seed. Repeated runs on the same problem are first
    # averaged within problem, then bootstrapped by independent problem ID.
    grouped={};reference={}
    for p in sorted((OUT/'planning').glob('*/seed*/repeat*/*/D*-B*/complete.json')):
        d=json.loads(p.read_text());assert d['rows_sha256']==sha(p.parent/'rows.npz')
        with np.load(p.parent/'rows.npz') as a:
            ids=a['reset_id'].tolist();assert len(ids)==120 and len(set(ids))==120
            assert a['states'].shape==(120,6,14)
            recomputed=np.linalg.norm(a['states'][:,-1,10:12]-a['goal'],axis=1)
            assert np.allclose(recomputed,a['endpoint_m'])
            assert np.array_equal(a['success'],recomputed<.03)
            key=(d['family'],d['seed'],d['method'])
            cells=grouped.setdefault(key,{})
            for i,identity in enumerate(ids):
                all_problem_ids.add(identity)
                start=a['states'][i,0];goal=a['goal'][i]
                if identity in reference:
                    assert np.array_equal(reference[identity][0],start)
                    assert np.array_equal(reference[identity][1],goal)
                else:reference[identity]=(start.copy(),goal.copy())
                by_repeat=cells.setdefault(identity,{})
                assert d['repeat'] not in by_repeat
                by_repeat[d['repeat']]=(float(recomputed[i]),float(a['success'][i]))
        provenance[str(p.relative_to(ROOT))]=sha(p)
    arrays={}
    for (family,seed,method),cells in grouped.items():
        assert len(cells)==1200
        expected=6 if family=='deployed' and seed==27 else 1
        assert all(len(v)==expected for v in cells.values())
        values=np.array([np.array(list(cells[k].values())).mean(0) for k in sorted(cells)])
        arrays[(family,seed,method)]=values
        planning.append(dict(family=family,seed=seed,method=method,independent_problems=1200,
            planner_repeats_per_problem=expected,executed_episodes=1200*expected,
            mean_endpoint_m=float(values[:,0].mean()),success_rate=float(values[:,1].mean())))
    for family,seed in sorted({(k[0],k[1]) for k in arrays}):
        ours=arrays[(family,seed,'ipwm')]
        for baseline in METHODS[1:]:
            base=arrays[(family,seed,baseline)];delta=base[:,0]-ours[:,0]
            ci=interval(delta)
            comparisons.append(dict(family=family,seed=seed,baseline=baseline,
                independent_problems=1200,endpoint_improvement_m=float(delta.mean()),
                endpoint_relative_improvement_pct=float(100*delta.mean()/base[:,0].mean()),
                paired_problem_ci_low_m=ci[0],paired_problem_ci_high_m=ci[1],
                success_difference_pp=float(100*(ours[:,1]-base[:,1]).mean())))
    assert len(training)==18 and len(predictions)==108 and len(planning)==36
    assert len(all_problem_ids)==1200
    write_csv(OUT/'training-results.csv',training)
    write_csv(OUT/'prediction-results.csv',predictions)
    write_csv(OUT/'prediction-condition-results.csv',conditions)
    write_csv(OUT/'planning-results.csv',planning)
    write_csv(OUT/'paired-planning-comparisons.csv',comparisons)
    (OUT/'results-summary.json').write_text(json.dumps(dict(training=training,prediction=predictions,
        planning=planning,paired_comparisons=comparisons,
        independent_planning_problems=1200,
        ci_scope='Paired independent problem bootstrap conditional on each fitted model; six planner repeats of deployed seed27 are averaged within each problem before bootstrap. Not a training-seed confidence interval.',
        provenance=provenance),indent=2))
    print('Complete result tables and paired intervals written')

if __name__=='__main__':main()
