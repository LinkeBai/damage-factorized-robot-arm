"""Same stiff-contact development replay at 0.5 ms for resolution comparison."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_probe as base
from common import OUT, read, write, sha
import mujoco
import numpy as np
from robotarm.envs import checked_push_reset as checked, supported_push_reset as supported
from robotarm.envs.constraint_lock import activate_joint_lock


def main():
    folder=OUT/'solver-development'; target=folder/'stiff_contact_limit_05ms.json'
    if target.exists(): raise FileExistsError(target)
    write(folder/'resolution-protocol.json', {'scope':'Same contact/limit parameters and all 320 development inputs; resolution check only',
        'dt_s':.0005,'duration_s':.25,'script_sha256':sha(__file__),
        'coarse_report_sha256':sha(folder/'stiff_contact_limit_1ms.json')})
    records=read(OUT/'data-development/records.json')
    with np.load(OUT/'data-development/data.npz') as a:
        q,v,actions=a['full_qpos'][:,0],a['full_qvel'][:,0],a['segment_actions']
    with np.load(folder/'stiff_contact_limit_1ms.npz') as a: coarse=a['traces']
    rows=[]; traces=[]
    for i,record in enumerate(records):
        lock,profile=record['lock'],record['profile']
        m=supported.make_model(lock,profile);base.configure(m,'stiff_contact_limit_1ms');m.opt.timestep=.0005
        d=mujoco.MjData(m);d.qpos[:]=q[i];d.qvel[:]=v[i]
        d.qpos[int(m.joint('block_z').qposadr[0])]=base.settled_z('stiff_contact_limit_1ms',profile)
        activate_joint_lock(m,d,checked.JOINTS[lock],record['reset_record']['lock_angle_rad'])
        capture=[]
        for step in range(501):
            if step:
                d.ctrl[:]=actions[i,(step-1)//100];mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d);g=checked.geometry(m,d);s=supported.support_measurement(m,d)
            capture.append([float(d.time),min(g['joint_margins_rad'].values()),
                min(v for n,v in g['arm_table_m'].items() if n!='base_geom'),min(g['arm_block_m'].values()),g['block_table_m'],
                s['z_displacement_m'],s['z_velocity_m_s'],*d.qpos,*d.qvel])
        a=np.asarray(capture);traces.append(a)
        bq=[int(m.joint(j).qposadr[0]) for j in ('block_x','block_y')]
        paired=a[::2];bc=[7+j for j in bq]
        xy=np.linalg.norm(paired[:,bc]-coarse[i][:,bc],axis=-1)
        rows.append({'identity':record['reset_record']['reset_id'],'kind':record['control_kind'],
            'min_joint_margin_rad':float(a[:,1].min()),'min_arm_table_m':float(a[:,2].min()),
            'min_arm_block_m':float(a[:,3].min()),'min_block_table_m':float(a[:,4].min()),
            'max_object_xy_difference_m':float(xy.max()),'endpoint_object_xy_difference_m':float(xy[-1]),
            'finite':bool(np.isfinite(a).all())})
    np.savez_compressed(folder/'stiff_contact_limit_05ms.npz',traces=np.asarray(traces))
    report={'rows':rows,'trace_sha256':sha(folder/'stiff_contact_limit_05ms.npz'),
        'summary':{k:min(r[k] for r in rows) for k in ('min_joint_margin_rad','min_arm_table_m','min_arm_block_m','min_block_table_m')}}
    report['summary'].update(max_object_xy_difference_m=max(r['max_object_xy_difference_m'] for r in rows),
        endpoint_object_xy_difference_m_max=max(r['endpoint_object_xy_difference_m'] for r in rows),
        endpoint_object_xy_difference_m_mean=float(np.mean([r['endpoint_object_xy_difference_m'] for r in rows])),
        all_finite=all(r['finite'] for r in rows))
    write(target,report);print(report['summary'],flush=True)


if __name__=='__main__': main()
