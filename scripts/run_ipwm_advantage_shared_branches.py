"""Exact evaluation reuse: selective publication joins two already evaluated branches."""
import argparse,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import audit_ipwm_targeted_advantage as audit
from scripts import replicate_ipwm_legacy_policy as legacy
from scripts import replicate_ipwm_policy_shift as directional
def reuse(original):
    def evaluate(models,traj,domain,device):
        atomic={k:v for k,v in models.items() if k not in ['selective','routed']}
        rows=original(atomic,traj,domain,device)
        carrier={(r['trajectory'],r['start'],r['horizon']):r for r in rows if r['method']=='carrier'}
        selected=[]
        for full in [r for r in rows if r['method']=='full']:
            base=carrier[(full['trajectory'],full['start'],full['horizon'])]
            selected.append(dict(full,method='selective',free_mse=base['free_mse'],
                robot_carrier_max=0.,lock_violation=base['lock_violation']))
        rows.extend(selected)
        if 'routed' in models:
            source=selected if models['routed'] is models['selective'] else list(carrier.values())
            rows.extend([dict(r,method='routed') for r in source])
        return rows
    return evaluate
def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',choices=['spectrum','legacy','directional'])
    args,remaining=parser.parse_known_args();sys.argv=[sys.argv[0],*remaining]
    # Restore the appropriate collector after importing both policy modules.
    if args.experiment=='legacy':legacy.collector.directional_push_waypoints=legacy.old_waypoints
    record=dict(optimization='Evaluate carrier/full branches once, then exactly join per-block metrics for selective publication. Reuse identical routed branch. No changes to checkpoints, trajectories, horizons or metrics.',
        launcher_sha256=audit.digest(__file__),
        verification='Compare with previously completed explicit-wrapper evaluations before accepting output.')
    (audit.OUT/'shared-branch-execution-amendment.json').write_text(json.dumps(record,indent=2))
    audit.evaluate=reuse(audit.evaluate);legacy.evaluate=reuse(legacy.evaluate)
    if args.experiment=='spectrum':audit.main()
    elif args.experiment=='legacy':legacy.main()
    else:directional.main()
if __name__=='__main__':main()
