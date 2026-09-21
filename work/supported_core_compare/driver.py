"""Single-owner finite queue for the registered nine-fit comparison."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, OUT, PROTOCOL, load_protocol, read, sha, write


def alive(pid):
    if not pid:
        return False
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    code = ctypes.c_ulong()
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel.CloseHandle(handle)
    return bool(ok and code.value == 259)


def verify_frozen():
    spec = load_protocol()
    assert read(OUT/'protocol-frozen.json')['protocol_sha256'] == sha(PROTOCOL)
    for path, digest in spec['source_sha256'].items():
        if sha(ROOT/path) != digest:
            raise RuntimeError('Frozen source changed: ' + path)
    implementation = read(OUT/'implementation-frozen.json')
    assert implementation['protocol_sha256'] == sha(PROTOCOL)
    for path, digest in implementation['source_sha256'].items():
        if sha(ROOT/path) != digest:
            raise RuntimeError('Frozen implementation changed: ' + path)
    gate = read(OUT/'physics-gate.json')
    assert gate['passed'] and gate['protocol_sha256'] == sha(PROTOCOL)
    assert gate['development_report_sha256'] == sha(OUT/spec.get('development_report', 'development-report.json'))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    guard = (OUT/'driver.lock').open('a+b')
    guard.seek(0)
    if not guard.read(1):
        guard.write(b'0'); guard.flush()
    guard.seek(0)
    msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
    previous = read(OUT/'driver-state.json', {})
    if previous.get('status') == 'running' and alive(previous.get('child_pid')):
        raise RuntimeError('Existing child is still running; do not duplicate: ' + str(previous['child_pid']))
    verify_frozen()
    state = {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
             'status': 'running', 'protocol_sha256': sha(PROTOCOL),
             'jobs': previous.get('jobs', {}), 'child_pid': None}
    write(OUT/'driver-state.json', state)
    folder = ROOT/'work/supported_core_compare'
    jobs = [
        ('collect-pool', 'data.py', ['--stage', 'collect', '--split', 'pool']),
        ('collect-validation', 'data.py', ['--stage', 'collect', '--split', 'validation']),
        ('collect-test', 'data.py', ['--stage', 'collect', '--split', 'test']),
        ('audit-data', 'data.py', ['--stage', 'audit']),
        ('train-nine-fits', 'train.py', ['--all']),
        ('predict-heldout', 'evaluate.py', ['--all']),
        ('planning-paired', 'planning.py', ['--all']),
        ('summarize-audit', 'report.py', []),
    ]
    try:
        for name, script, arguments in jobs:
            verify_frozen()
            if (OUT/'PAUSE').exists():
                state.update(status='paused', current_job=name, child_pid=None)
                write(OUT/'driver-state.json', state)
                return
            if name == 'audit-data' and (OUT/'data-audit.json').exists():
                audit = read(OUT/'data-audit.json')
                if not audit.get('passed') or audit['protocol_sha256'] != sha(PROTOCOL):
                    raise RuntimeError('Existing data audit is failed or belongs to another protocol')
                if not audit.get('files'):
                    raise RuntimeError('Existing data audit lacks file coverage')
                for relative, digest in audit['files'].items():
                    if sha(OUT/relative) != digest:
                        raise RuntimeError('Audited data changed: ' + relative)
                state['jobs'][name] = {'status':'complete','reused_verified_audit':True,
                                      'audit_sha256':sha(OUT/'data-audit.json')}
                write(OUT/'driver-state.json',state)
                continue
            # Every stage verifies and reuses only complete hashed artifacts.
            # Even previously completed stages are re-entered to audit resume.
            log_path = OUT/'logs'/f'{name}.log'
            log_path.parent.mkdir(exist_ok=True)
            command = [sys.executable, '-u', str(folder/script), *arguments]
            began = time.time()
            with log_path.open('a', encoding='utf-8') as log:
                log.write('\nDriver dispatch ' + datetime.now(timezone.utc).isoformat() + '\n')
                log.flush()
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state.update(current_job=name, child_pid=child.pid, updated_utc=datetime.now(timezone.utc).isoformat())
                state['jobs'][name] = {'status': 'running', 'pid': child.pid,
                    'started_unix': began, 'command': command, 'log': str(log_path.relative_to(ROOT))}
                write(OUT/'driver-state.json', state)
                print(name, 'PID', child.pid, flush=True)
                code = child.wait()
            state['jobs'][name].update(status='complete' if code == 0 else 'failed',
                                      returncode=code, wall_seconds=time.time()-began)
            state['child_pid'] = None
            write(OUT/'driver-state.json', state)
            if code:
                raise RuntimeError(f'{name} failed with {code}; retained log {log_path}')
        audit = read(OUT/'completion-audit.json')
        if not audit.get('passed'):
            raise RuntimeError('Final completeness/numeric audit failed')
        state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(), child_pid=None)
        write(OUT/'driver-state.json', state)
        print('Complete: all nine fits, held-out prediction, paired planning and audit.', flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                     updated_utc=datetime.now(timezone.utc).isoformat())
        write(OUT/'driver-state.json', state)
        raise
    finally:
        guard.close()


if __name__ == '__main__':
    main()
