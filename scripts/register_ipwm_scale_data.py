"""Freeze the sampling protocol and register the complete data queue once."""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from ipwm_scale_data import ROOT, OUT, sha, SPLITS, PROFILES

def main():
    path = OUT/'data-protocol.json'
    if path.exists():
        raise FileExistsError('Protocol already frozen')
    check = json.loads((OUT/'simulator-validation.json').read_text())
    assert check['passed'] and check['script_sha256'] == sha(ROOT/'scripts/ipwm_scale_data.py')
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(ROOT/'scripts/ipwm_scale_data.py'),
        xml_sha256=sha(ROOT/'sim/assets/arm_push.xml'),
        conditions=['D1','D2','D3','D4','D5'], profiles=PROFILES,
        per_condition_counts={k:v[0] for k,v in SPLITS.items()},
        independence='One unique reset seed, randomized arm pose/velocity and object translation, one independent action sequence per trajectory. Windows are not independent examples.',
        units='robot radians and radians/second; object world xy meters and linear velocity meters/second; no mixed-unit RMSE as headline metric',
        horizons=[10,25,50], steps=50, timestep_seconds=.005,
        commands='5 independent uniform [-0.8,0.8] torque-command segments, 10 steps each; locked actuator zero; solver-native equality lock at sampled initial angle.',
        scope='Contact-neighborhood box pushing. Five damage conditions are not five object geometries. This set tests local prediction and adaptation, not approach success.',
        split_policy='No test or validation trajectories used for training. Formal downstream fitting and planning protocols frozen separately before any test result inspection.',
        checkpoint_policy='Existing deployed checkpoints are immutable; newly trained repetitions are a distinct scale study without new real-robot validation.')
    path.write_text(json.dumps(protocol, indent=2))
    queue=[]
    for split in SPLITS:
        for lock in range(1,6):
            queue.append(dict(id=f'collect-{split}-D{lock}',
                command=[sys.executable, 'scripts/ipwm_scale_data.py', '--split',split,'--lock',str(lock)],
                outputs=[f'runs/ipwm_scale_goal_20260911/data/{split}/D{lock}/manifest.json']))
    (OUT/'queue.json').write_text(json.dumps(queue,indent=2))
    print('Registered',len(queue),'jobs, 50,000 pool + 2,000 validation + 6,000 test resets')

if __name__=='__main__':
    main()
