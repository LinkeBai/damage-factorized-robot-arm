"""Independently inspect development evidence and record its precise scope."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from common import ROOT,OUT,PROTOCOL,load_protocol,read,write,sha
from robotarm.envs import supported_push_reset as old, resolved_push_reset as new


def main():
    spec=load_protocol();target=OUT/'physics-gate.json'
    if target.exists():raise FileExistsError(target)
    report_path=OUT/spec['development_report'];report=read(report_path)
    folder=OUT/spec['development_directory'];records=read(folder/'records.json');manifest=read(folder/'manifest.json')
    checks={}
    checks['exact_development_membership']=len(records)==320 and len({(r['lock'],r['profile'],r['index'],r['control_kind']) for r in records})==320
    checks['source_integrity']=report['protocol_sha256']==sha(PROTOCOL) and report['source_unchanged_during_run'] and all(sha(ROOT/p)==h for p,h in report['source_sha256'].items())
    checks['raw_file_hashes']=manifest['npz_sha256']==sha(folder/'data.npz') and manifest['records_sha256']==sha(folder/'records.json')
    checks['all_initial_audits_and_finite_states']=report['hard_gate_passed'] and all(r['reset_record']['passed'] and r['diagnostics']['finite'] for r in records)
    checks['correct_actuator_mapping']=all(r['diagnostics']['max_actuator_torque_error_Nm']<1e-12 for r in records)
    checks['no_object_motion_before_contact']=all(r['diagnostics']['max_object_displacement_without_prior_contact_m']<1e-10 for r in records)
    disabled=('shoulder_geom','upper_geom','forearm_geom','wrist_pitch_geom','wrist_roll_geom')
    checks['noncolliding_visible_arm_clear_of_table']=all(r['diagnostics']['arm_table_minima_m'][name]>0 for r in records for name in disabled)
    with np.load(folder/'data.npz',allow_pickle=False) as arrays:
        checks['full_internal_trace_recorded']=arrays['substep_diagnostic_metrics'].shape[:3]==(320,50,20) and np.isfinite(arrays['substep_diagnostic_metrics']).all()
        checks['duration_and_internal_dt']=np.allclose(arrays['substep_time_s'][:,-1,-1],.25,atol=1e-10) and all(r['diagnostics']['internal_timestep_s']==.00025 for r in records)
        checks['commands_match_declared_range']=np.max(abs(arrays['segment_actions']))<=.8
        for i,r in enumerate(records):
            if np.max(abs(arrays['segment_actions'][i,:,r['lock']]))!=0:checks['commands_match_declared_range']=False
    properties=('body_mass','body_inertia','body_pos','geom_size','geom_friction','geom_pos','geom_quat','geom_contype','geom_conaffinity',
                'actuator_gear','actuator_ctrlrange','actuator_forcerange','jnt_range','dof_damping','dof_armature','eq_solref','eq_solimp')
    changes={}
    for lock in range(5):
        for profile in new.PROFILES:
            a=old.make_model(lock,profile);b=new.make_model(lock,profile)
            changes[f'{lock}/{profile}']={name:bool(np.array_equal(getattr(a,name),getattr(b,name))) for name in properties}
            assert b.opt.timestep==.00025 and np.array_equal(b.geom_solref[0],[.004,1.])
    checks['declared_parameters_only_changed']=all(all(v.values()) for v in changes.values())
    write(OUT/'physics-model-preservation.json',changes)
    resolution=read(OUT/'solver-development/stiff_contact_limit_025ms.json')
    groups={}
    for kind in ('zero','random','positive','negative'):
        x=np.array([r['endpoint_object_xy_difference_m'] for r in resolution['rows'] if r['kind']==kind])
        groups[kind]={'mean_endpoint_step_halving_difference_mm':float(x.mean()*1000),
                      'p95_mm':float(np.quantile(x,.95)*1000),'max_mm':float(x.max()*1000)}
    result={'passed':bool(all(checks.values())), 'protocol_sha256':sha(PROTOCOL),
        'development_report_sha256':sha(report_path), 'checks':{k:bool(v) for k,v in checks.items()},
        'source_sha256':report['source_sha256'],
        'scope':'Release of a declared idealized soft-contact CPU benchmark after geometry/support/command/internal-state checks; not hardware calibration.',
        'development_dynamics':{k:v for k,v in report.items() if k not in ('source_sha256','raw_directory')},
        'step_halving_sensitivity':groups,
        'uniform_submillimeter_convergence_demonstrated':False,
        'global_physical_accuracy_demonstrated':False,
        'model_performance_evaluated':False,
        'review':'Initial invalid overlaps and missing normal support were corrected. Contact/limit softness was explicitly revised and integration refined. Finite soft-contact/limit violations remain and are archived at every internal step; no trace was discarded or success-conditioned. Random-command resolution sensitivity is much smaller on average than the stress-case maximum; both are retained. Findings must be reported at the registered discretization and cannot support claims of universally submillimeter physical accuracy.',
        'limitations':['Fixed object orientation.','Learned 14-D state omits archived block z/vz.',
                       'Raw torque with gravity can move the arm/object even at zero commands.',
                       'Small performance differences may be comparable to numerical/contact sensitivity.'],
        'model_preservation_sha256':sha(OUT/'physics-model-preservation.json')}
    write(target,result)
    print(result['passed'],result['checks'],flush=True)
    if not result['passed']:raise RuntimeError('Physics review failed; results retained')


if __name__=='__main__':main()
