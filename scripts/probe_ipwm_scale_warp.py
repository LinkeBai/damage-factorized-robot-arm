"""CPU/Warp contact-rollout parity probe; never counted as paper evidence."""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp
from scripts.generate_primary_sequence_candidates import JOINTS, CONTACT_QPOS, set_arm_state
from robotarm.envs.constraint_lock import model_with_inactive_joint_locks, activate_joint_lock

def main():
    out = ROOT / 'runs/ipwm_scale_goal_20260911'
    out.mkdir(parents=True, exist_ok=True)
    wp.init()
    model = model_with_inactive_joint_locks(ROOT / 'sim/assets/arm_push.xml', JOINTS)
    data = mujoco.MjData(model)
    set_arm_state(model, data, CONTACT_QPOS, np.zeros(5))
    activate_joint_lock(model, data, 'j3', CONTACT_QPOS[2])
    n = 128
    actions = np.random.default_rng(91112001).uniform(-.8, .8, (5, n, 5)).astype(np.float32)
    actions[:, :, 2] = 0
    wm = mjw.put_model(model)
    wd = mjw.put_data(model, data, nworld=n)
    # Compile before measuring steady-state execution.
    mjw.step(wm, wd)
    wp.synchronize()
    wd = mjw.put_data(model, data, nworld=n)
    start = time.perf_counter()
    for segment in actions:
        wd.ctrl.assign(segment)
        for _ in range(10):
            mjw.step(wm, wd)
    wp.synchronize()
    gpu_seconds = time.perf_counter() - start
    gpu = wd.qpos.numpy()
    cpu = []
    start = time.perf_counter()
    for i in range(n):
        d = mujoco.MjData(model)
        d.qpos[:] = data.qpos
        d.qvel[:] = data.qvel
        d.eq_active[:] = data.eq_active
        mujoco.mj_forward(model, d)
        for segment in actions:
            d.ctrl[:] = segment[i]
            mujoco.mj_step(model, d, nstep=10)
        cpu.append(d.qpos.copy())
    cpu_seconds = time.perf_counter() - start
    cpu = np.array(cpu)
    block = [int(model.joint(x).qposadr[0]) for x in ('block_x', 'block_y')]
    error = np.linalg.norm(gpu[:, block] - cpu[:, block], axis=1)
    result = dict(probe_only=True, n=n, horizon=50, gpu_seconds=gpu_seconds,
                  cpu_seconds=cpu_seconds, endpoint_difference_max_m=float(error.max()),
                  endpoint_difference_mean_m=float(error.mean()),
                  finite=bool(np.isfinite(gpu).all()),
                  preliminary_parity_pass=bool(np.isfinite(gpu).all() and error.max() < .001),
                  threshold_m=.001,
                  note='One contact pose and D3 only. Other locks and batch graph speed still require verification.')
    (out / 'warp-probe.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
