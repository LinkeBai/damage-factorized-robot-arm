"""Freeze the complete finite experiment queue after development checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,OUT,PROTOCOL,sha,read,write


def main():
    target=OUT/'implementation-frozen.json'
    if target.exists():raise FileExistsError(target)
    gate=read(OUT/'physics-gate.json')
    assert gate['passed'] and gate['protocol_sha256']==sha(PROTOCOL)
    paths=[ROOT/'work/supported_core_compare'/name for name in
           ('common.py','data.py','train.py','evaluate.py','planning.py','driver.py','report.py','review_physics.py')]
    paths += [ROOT/'src/robotarm/envs/resolved_push_reset.py']
    value={'protocol_sha256':sha(PROTOCOL),'physics_gate_sha256':sha(OUT/'physics-gate.json'),
           'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}
    write(target,value)
    print(sha(target))


if __name__=='__main__':main()
