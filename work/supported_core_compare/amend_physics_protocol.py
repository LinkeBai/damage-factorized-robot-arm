"""Archive the initial registration and register physics correction before fits."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,OUT,PROTOCOL,read,write,sha


def main():
    if (OUT/'training').exists() or (OUT/'data').exists():
        raise RuntimeError('Amendment must precede formal data/fits')
    old=read(PROTOCOL)
    if old['version']!='supported-core-v1':raise RuntimeError('Already amended')
    folder=OUT/'protocol-history';folder.mkdir(exist_ok=True)
    for name in ('protocol.json','protocol-frozen.json'):
        target=folder/('v1-'+name)
        if target.exists():raise FileExistsError(target)
        shutil.copy2(OUT/name,target)
    previous=sha(PROTOCOL)
    spec=dict(old)
    spec.update(version='supported-core-v2',model_revision='supported-resolved-contact-v2',
                previous_protocol_sha256=previous,
                physics_amended_utc=datetime.now(timezone.utc).isoformat(),
                development_directory='data-development-v2',development_report='development-v2-report.json',
                numerics={'internal_timestep_s':.00025,'observation_timestep_s':.005,
                    'internal_steps_per_observation':20,'geom_and_limit_solref':[.004,1.],
                    'geom_and_limit_solimp_first_three':[.99,.9999,.001],
                    'scope':'Explicit soft-contact/limit model revision plus numerical substepping; not measured hardware stiffness.'})
    spec['source_sha256']=dict(old['source_sha256'])
    spec['source_sha256']['src/robotarm/envs/resolved_push_reset.py']=sha(ROOT/'src/robotarm/envs/resolved_push_reset.py')
    spec['limitations']=old['limitations']+[
        'Soft contacts and soft joint limits are finite-compliance approximations; all internal-step violations are retained.',
        'Step-halving development sensitivity is reported; this does not establish uniform submillimeter convergence.']
    proof=['development-report.json','solver-development/unchanged_softness_1ms.json',
           'solver-development/stiff_contact_limit_1ms.json','solver-development/stiff_contact_limit_05ms.json',
           'solver-development/stiff_contact_limit_025ms.json']
    amendment={'previous_protocol_sha256':previous,'reason':'Formal-action development exposed excessive default contact/limit softness; resolve before collecting formal comparison data.',
        'basis':{name:sha(OUT/name) for name in proof},
        'unchanged':['methods','seeds','training budget','training/validation selection','test count','planning task count',
                     'action amplitude','action/observation durations','mass','geometry','friction','damping','actuators'],
        'no_learned_outcome_consulted':True,
        'documentation':'https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters'}
    write(PROTOCOL,spec)
    amendment['new_protocol_sha256']=sha(PROTOCOL)
    write(folder/'physics-amendment-v2.json',amendment)
    write(OUT/'protocol-frozen.json',{'protocol_sha256':sha(PROTOCOL),'previous_protocol_sha256':previous,
                                   'created_utc':spec['physics_amended_utc']})
    print(sha(PROTOCOL))


if __name__=='__main__':main()
