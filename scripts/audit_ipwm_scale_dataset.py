"""Audit every stored trajectory, not merely manifest totals."""
import json
import sys
import hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/ipwm_scale_goal_20260911'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    protocol=json.loads((OUT/'data-protocol.json').read_text())
    seen=set(); fingerprints=set(); counts={}; shards=[]
    for split,expected in protocol['per_condition_counts'].items():
        counts[split]={}
        for lock in range(1,6):
            folder=OUT/'data'/split/f'D{lock}'
            manifest=json.loads((folder/'manifest.json').read_text())
            count=0
            for row in manifest['shards']:
                path=ROOT/row['path']
                assert sha(path)==row['sha256']
                with np.load(path) as a:
                    s=a['states']; u=a['segment_actions']; ids=a['reset_id']
                    assert s.shape==(len(ids),51,14) and u.shape==(len(ids),5,5)
                    assert np.isfinite(s).all() and np.isfinite(u).all()
                    assert np.all(a['locked_joint']==lock-1) and np.all(u[:,:,lock-1]==0)
                    assert len(set(ids.tolist()))==len(ids)
                    assert not seen.intersection(ids.tolist())
                    seen.update(ids.tolist())
                    for i in range(len(ids)):
                        fingerprint=hashlib.sha256(s[i,0].tobytes()+u[i].tobytes()).hexdigest()
                        assert fingerprint not in fingerprints, 'Duplicate physical reset/action sequence'
                        fingerprints.add(fingerprint)
                    count+=len(ids)
                    # Movement is descriptive only; motionless samples are retained.
                    displacement=np.linalg.norm(s[:,-1,10:12]-s[:,0,10:12],axis=1)
                    shards.append(dict(path=row['path'],sha256=row['sha256'],count=len(ids),
                        moving_over_1mm=int((displacement>.001).sum())))
            assert count==expected==manifest['count']
            counts[split][f'D{lock}']=count
    result=dict(passed=True,counts=counts,total_unique_reset_ids=len(seen),
                total_unique_reset_action_fingerprints=len(fingerprints),
                data_protocol_sha256=sha(OUT/'data-protocol.json'),shards=shards)
    (OUT/'dataset-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='shards'},indent=2))

if __name__=='__main__':main()
