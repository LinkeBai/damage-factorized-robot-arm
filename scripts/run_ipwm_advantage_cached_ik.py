"""Execution-only optimization: memoize deterministic IK, preserve all trial inputs."""
import json, runpy, sys, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import scripts.run_push_benchmark as collector
original=collector.solve_reach_reference
cache={}
def memo(target,ranges,*,locked_joints=None,config=None):
    key=(np.asarray(target,dtype=np.float64).tobytes(),np.asarray(ranges,dtype=np.float64).tobytes(),
         tuple(sorted((locked_joints or {}).items())),repr(config))
    if key not in cache: cache[key]=original(target,ranges,locked_joints=locked_joints,config=config)
    q,error=cache[key]
    return q.copy(),error
collector.solve_reach_reference=memo
out=ROOT/'runs/ipwm_targeted_advantage_20260911'
(out/'execution-amendment.json').write_text(json.dumps({
    'reason':'Memoize repeated deterministic IK solves; no changes to inputs, sample sizes, metrics, checkpoints or thresholds.',
    'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'solver_sha256':hashlib.sha256((ROOT/'src/robotarm/training/controllers.py').read_bytes()).hexdigest(),
},indent=2))
runpy.run_path(str(ROOT/'scripts/audit_ipwm_targeted_advantage.py'),run_name='__main__')
