"""Run registered planning cells in one interpreter without changing simulation.

Calls the same frozen run() entry point with the exact registered arguments.
Completed cells are verified before reuse; no results are sampled or filtered.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/ipwm_scale_goal_20260911'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def arguments(job):
    args=job['command'][2:]
    assert len(args)==12 and all(args[i].startswith('--') for i in range(0,len(args),2))
    values={args[i][2:]:args[i+1] for i in range(0,len(args),2)}
    assert set(values)=={'family','method','seed','repeat','lock','band'}
    for key in ['seed','repeat','lock','band']:values[key]=int(values[key])
    return argparse.Namespace(**values)

def verified(job):
    paths=[ROOT/p for p in job['outputs']]
    if not all(p.is_file() for p in paths):return False
    metadata=next(p for p in paths if p.name=='complete.json')
    rows=next(p for p in paths if p.name=='rows.npz')
    result=json.loads(metadata.read_text())
    for key,value in vars(arguments(job)).items():
        assert result[key]==value,(job['id'],key)
    assert result['protocol_sha256']==sha(OUT/'planning-protocol.json')
    assert result['rows_sha256']==sha(rows),job['id']
    assert result['independent_problems']==120
    return True

def main():
    sys.path.insert(0,str(ROOT/'scripts'))
    from ipwm_scale_planning import run
    jobs=[j for j in json.loads((OUT/'queue.json').read_text()) if j['id'].startswith('plan-')]
    assert len(jobs)==560
    for i,job in enumerate(jobs):
        if verified(job):continue
        print('BEGIN',job['id'],flush=True)
        status={'current':job['id'],'registered_cells':len(jobs),'index':i,'started':time.time()}
        (OUT/'planning-bundle-progress.json').write_text(json.dumps(status,indent=2))
        run(arguments(job))
        assert verified(job)
        print('COMPLETE',job['id'],flush=True)
        if (i+1)%10==0:
            from report_ipwm_scale_progress import main as report_progress
            report_progress()
    (OUT/'planning-bundle-progress.json').write_text(json.dumps({'status':'complete','registered_cells':len(jobs)},indent=2))

if __name__=='__main__':main()
