"""Recompute every result table from raw paired rows, without outcome filtering."""
from __future__ import annotations
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import ROOT, OUT, PROTOCOL, load_protocol, read, sha, write

METHODS = ('ipwm','carrier','global')
SEEDS = (7,17,27)


def close(actual, expected, label, atol=1e-10):
    if not np.allclose(actual, expected, rtol=1e-6, atol=atol):
        raise ValueError('Numeric mismatch: ' + label)


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    from driver import verify_frozen
    from train import formal_identity, verify_complete, verify_selection_freeze
    from evaluate import checked_test_data, metric_arrays, summarize_rows
    from data import METRIC_NAMES
    verify_frozen()
    identity=formal_identity()
    selection=verify_selection_freeze()
    spec=load_protocol()
    truth, sources=checked_test_data()
    audit={'passed':False,'protocol_sha256':sha(PROTOCOL),'checks':{},'source_results':{}}
    prediction=[];planning=[];contrasts=[]
    pred_identity=None;problem_reference={};candidate_reference={}
    try:
        for seed in SEEDS:
            for method in METHODS:
                key=f'{method}/seed{seed}'
                fitted=verify_complete(OUT/'training'/key,identity)
                assert fitted['total_updates']==980 and fitted['actual_training_unique_trajectories']==50000
                assert len(fitted['history'])==10 and not fitted['smoke']
                folder=OUT/'prediction'/key;c=read(folder/'complete.json')
                assert c['rows_sha256']==sha(folder/'rows.npz')
                assert c['identity']['model_sha256']==selection['models'][key]
                audit['source_results'][str(folder.relative_to(OUT))]=sha(folder/'complete.json')
                with np.load(folder/'rows.npz',allow_pickle=False) as a:
                    assert len(a['reset_id'])==6000
                    assert np.array_equal(a['reset_id'],truth['reset_id'])
                    assert np.array_equal(a['initial_state_sha256'],truth['initial_state_sha256'])
                    for h in (10,25,50):
                        prefix=f'h{h}_';actual=truth['states'][:,h]
                        close(a[prefix+'truth_state14'],actual,'test truth')
                        locks=a['locked_joint'].astype(int)
                        angles=truth['states'][np.arange(6000),0,locks]
                        expected=metric_arrays(a[prefix+'predicted_state14'],actual,
                                               a[prefix+'reference_state14'],locks,angles)
                        for name,values in expected.items():close(a[prefix+name],values,prefix+name)
                        xy=a[prefix+'object_xy_squared_m2'];vel=a[prefix+'object_velocity_squared_m2_s2']
                        row={'seed':seed,'method':method,'horizon':h,'trajectories':6000,
                             'object_xy_per_coordinate_rmse_mm':float(np.sqrt(xy.mean())*1000),
                             'object_xy_euclidean_rmse_mm':float(np.sqrt(xy.sum(1).mean())*1000),
                             'object_velocity_per_coordinate_rmse_m_s':float(np.sqrt(vel.mean())),
                             'free_q_rmse_rad':float(np.sqrt(a[prefix+'free_q_mean_squared_rad2'].mean())),
                             'free_v_rmse_rad_s':float(np.sqrt(a[prefix+'free_velocity_mean_squared_rad2_s2'].mean())),
                             'fk_pusher_per_coordinate_rmse_mm':float(np.sqrt(a[prefix+'pusher_xy_squared_m2'].mean())*1000),
                             'reference_q_max_abs_rad':float(a[prefix+'reference_q_abs_rad'].max()),
                             'reference_v_max_abs_rad_s':float(a[prefix+'reference_velocity_abs_rad_s'].max()),
                             'reference_pusher_max_abs_mm':float(a[prefix+'reference_pusher_xy_abs_m'].max()*1000),
                             'lock_max_abs_rad':float(a[prefix+'lock_position_abs_rad'].max())}
                        close(row['object_xy_per_coordinate_rmse_mm'],c['summary'][str(h)]['object_position_per_coordinate_rmse_mm'],'prediction summary mm')
                        prediction.append(row)
                episodes=[]
                for lock in range(5):
                    for band in range(2):
                        folder=OUT/'planning'/key/f'D{lock+1}-B{band}';c=read(folder/'complete.json')
                        assert c['rows_sha256']==sha(folder/'rows.npz') and c['resets_sha256']==sha(folder/'resets.json')
                        assert c['identity']['model_sha256']==selection['models'][key]
                        assert c['independent_problems']==120 and c['candidate_budget']==128
                        audit['source_results'][str(folder.relative_to(OUT))]=sha(folder/'complete.json')
                        with np.load(folder/'rows.npz',allow_pickle=False) as a:
                            states=a['states'];goals=a['goal_xy_m']
                            assert states.shape==(120,51,14) and np.isfinite(states).all()
                            assert np.isfinite(a['substep_diagnostic_metrics']).all()
                            close(a['time_s'],np.broadcast_to(np.arange(51)*.005,(120,51)),'observation time')
                            initial=np.linalg.norm(states[:,0,10:12]-goals,axis=1)
                            endpoint=np.linalg.norm(states[:,-1,10:12]-goals,axis=1)
                            assert np.all(initial>0)
                            close(a['initial_distance_m'],initial,'initial distance')
                            close(a['terminal_distance_m'],endpoint,'terminal distance')
                            close(a['progress_m'],initial-endpoint,'goal progress')
                            close(a['relative_terminal_error'],endpoint/initial,'relative endpoint')
                            assert np.array_equal(a['success_at_30mm'],endpoint<.03)
                            close(a['object_net_displacement_m'],np.linalg.norm(states[:,-1,10:12]-states[:,0,10:12],axis=1),'net movement')
                            assert np.array_equal(a['chosen_indices'],a['predicted_candidate_terminal_distance_m'].argmin(axis=2))
                            close(a['predicted_candidate_terminal_distance_m'],np.linalg.norm(a['predicted_candidate_terminal_xy_m']-goals[:,None,None,:],axis=-1),'candidate cost')
                            chosen=a['chosen_segment_commands'][:,:,0,:]
                            close(a['controls'],np.repeat(chosen,10,axis=1),'executed commands')
                            assert np.max(abs(a['controls']))<=.8000001 and np.max(abs(a['controls'][:,:,lock]))==0
                            problem_key=(lock,band)
                            common=(a['reset_id'].copy(),a['initial_state_sha256'].copy(),states[:,0].copy(),goals.copy())
                            if problem_key in problem_reference:
                                assert all(np.array_equal(x,y) for x,y in zip(common,problem_reference[problem_key]))
                            else:problem_reference[problem_key]=common
                            ck=(seed,lock,band);ch=a['candidate_array_sha256'].copy()
                            if ck in candidate_reference:assert np.array_equal(ch,candidate_reference[ck])
                            else:candidate_reference[ck]=ch
                            internal=a['substep_diagnostic_metrics']
                            limits=internal[...,METRIC_NAMES.index('min_joint_margin_rad')]
                            table=internal[...,METRIC_NAMES.index('min_arm_table_m')]
                            z=internal[...,METRIC_NAMES.index('block_z_m')]
                            episodes.append({'e0':initial,'eT':endpoint,'progress':initial-endpoint,'relative':endpoint/initial,
                                'contact':a['first_tool_object_contact_step']>=0,'min_margin':float(limits.min()),
                                'min_table':float(table.min()),'z_min':float(z.min()),'z_max':float(z.max())})
                e0=np.concatenate([e['e0'] for e in episodes]);eT=np.concatenate([e['eT'] for e in episodes])
                planning.append({'seed':seed,'method':method,'problems':len(eT),
                    'initial_distance_mean_mm':float(e0.mean()*1000),'terminal_distance_mean_mm':float(eT.mean()*1000),
                    'progress_mean_mm':float((e0-eT).mean()*1000),'relative_terminal_error_mean':float((eT/e0).mean()),
                    'success_at_30mm':float((eT<.03).mean()),'positive_progress_fraction':float((eT<e0).mean()),
                    'tool_object_contact_fraction':float(np.concatenate([e['contact'] for e in episodes]).mean()),
                    'min_dynamic_joint_margin_rad':min(e['min_margin'] for e in episodes),
                    'min_dynamic_arm_table_gap_mm':min(e['min_table'] for e in episodes)*1000,
                    'z_min_mm':min(e['z_min'] for e in episodes)*1000,'z_max_mm':max(e['z_max'] for e in episodes)*1000})
        audit['checks'].update(nine_complete_fits=True,all_prediction_rows_recomputed=True,
            ninety_planning_cells_recomputed=True,all_problem_and_candidate_pairs_match=True,
            all_seeds_and_methods_retained=True,units_and_observation_timing_verified=True)
        aggregate=[]
        for method in METHODS:
            values=[r for r in planning if r['method']==method]
            row={'method':method,'adaptation_seeds':3,'independent_planning_problems':1200}
            for metric in ('terminal_distance_mean_mm','progress_mean_mm','relative_terminal_error_mean','success_at_30mm'):
                x=np.asarray([r[metric] for r in values]);row[metric+'_mean']=float(x.mean());row[metric+'_seed_std']=float(x.std(ddof=1))
            aggregate.append(row)
        for seed in SEEDS:
            bymethod={r['method']:r for r in planning if r['seed']==seed}
            for baseline in ('carrier','global'):
                base=bymethod[baseline]['terminal_distance_mean_mm'];ours=bymethod['ipwm']['terminal_distance_mean_mm']
                contrasts.append({'seed':seed,'comparison':'ipwm_vs_'+baseline,
                    'terminal_distance_reduction_pct':100*(1-ours/base) if base else None})
        summary={'protocol_sha256':sha(PROTOCOL),'prediction':prediction,'planning':planning,
            'planning_across_seeds':aggregate,'paired_seed_contrasts':contrasts,
            'scope':'Corrected simulation adaptation, one common pretraining source; no real-robot or isolation attribution.',
            'counting':'6000 unique heldout trajectories and 1200 unique planning problems; repeated methods/seeds do not multiply unique starts.',
            'numeric_definition':'XY per-coordinate RMSE averages both coordinates; Euclidean RMSE sums XY squared errors before averaging. Planning uses mean Euclidean terminal distance.',
            'physics_gate':read(OUT/'physics-gate.json')}
        write(OUT/'summary.json',summary)
        csv_write(OUT/'prediction-all-seeds.csv',prediction);csv_write(OUT/'planning-all-seeds.csv',planning)
        lines=['# 修复后核心对照：全部结果','',
               '三种适配种子共享同一预训练来源；所有方法和种子完整保留。',
               '预测位置误差以下为每坐标RMSE；任务终点误差为二维欧氏距离均值，两者不能混为同一指标。','',
               '| 种子 | 方法 | H50位置RMSE (mm/坐标) | 终点距离 (mm) | 朝目标进展 (mm) | eT/e0 | 成功率 |',
               '|---|---|---:|---:|---:|---:|---:|']
        for r in planning:
            p=next(p for p in prediction if p['seed']==r['seed'] and p['method']==r['method'] and p['horizon']==50)
            lines.append(f"| {r['seed']} | {r['method']} | {p['object_xy_per_coordinate_rmse_mm']:.3f} | {r['terminal_distance_mean_mm']:.3f} | {r['progress_mean_mm']:.3f} | {r['relative_terminal_error_mean']:.4f} | {r['success_at_30mm']:.3%} |")
        lines += ['', '完整H10/H25/H50、速度、机器人/FK参考偏差及动态诊断见JSON/CSV和原始rows.npz。',
                  '本组仿真不自动证明隔离机制的任务增益，不构成新增真机对照；数值与物理模型的限定见physics-gate.json。']
        (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        audit['passed']=True
        audit['outputs']={name:sha(OUT/name) for name in ('summary.json','summary.md','prediction-all-seeds.csv','planning-all-seeds.csv')}
        write(OUT/'completion-audit.json',audit)
        print('All raw-row, pairing and completeness checks passed.',flush=True)
    except BaseException as error:
        audit['error']=repr(error);write(OUT/'completion-audit.json',audit);raise


if __name__=='__main__':main()
