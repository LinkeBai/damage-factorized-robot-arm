"""Resumable registered-job loop. Missing evidence never counts as completion.

Queue entries: id, command (argument list), outputs (repo-relative paths).
Validated completion is written separately after auditing scientific evidence.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/ipwm_scale_goal_20260911'
CONTRACT = ROOT / 'config/experiment/ipwm_scale_goal_20260911.json'

def read(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def process_alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        code = wintypes.DWORD()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

def acquire_runner():
    handle = (OUT / 'runner.lock').open('a+b')
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b'0'); handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read(OUT / 'runner-lease.json', {})
        pid = previous.get('pid')
        if pid and pid != os.getpid() and process_alive(pid):
            raise RuntimeError(f'An existing scale loop is still running (PID {pid})')
        write(OUT / 'runner-lease.json', {'pid': os.getpid(), 'started': time.time()})
        return handle
    except BaseException:
        handle.close()
        raise

def registered_jobs():
    """Pick up appended stages at batch boundaries without restarting jobs."""
    delivered = set()
    while True:
        current = read(OUT / 'queue.json', [])
        ids = [j['id'] for j in current]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate registered job IDs')
        pending = [j for j in current if j['id'] not in delivered]
        if not pending:
            return
        job = pending[0]
        delivered.add(job['id'])
        yield job

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-jobs', type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runner_guard = acquire_runner()
    state = read(OUT / 'loop-state.json', {'jobs': {}})
    queue = read(OUT / 'queue.json', [])
    completed = 0
    for job in registered_jobs():
        previous = state['jobs'].get(job['id'], {})
        signature = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
        if previous.get('status') == 'complete' and previous.get('signature') == signature:
            if all((ROOT / p).is_file() and digest(ROOT / p) == h
                   for p, h in previous['output_hashes'].items()):
                continue
            raise RuntimeError('Previously completed output changed: ' + job['id'])
        if args.max_jobs and completed >= args.max_jobs:
            break
        state['jobs'][job['id']] = {'status': 'running', 'signature': signature, 'started': time.time()}
        write(OUT / 'loop-state.json', state)
        command = job['command']
        bundled_existing = False
        if job['id'].startswith('plan-'):
            from run_ipwm_scale_planning_bundle import verified
            bundled_existing = verified(job)
            command = [sys.executable, str(ROOT / 'scripts/run_ipwm_scale_planning_bundle.py')]
        if bundled_existing:
            result = subprocess.CompletedProcess(command, 0)
        else:
            with (OUT / (job['id'] + '.log')).open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        item = state['jobs'][job['id']]
        item['returncode'] = result.returncode
        item['execution_command'] = command
        item['reused_verified_bundle_output'] = bundled_existing
        outputs_ok = bool(job['outputs']) and all((ROOT / p).is_file() for p in job['outputs'])
        item['status'] = 'complete' if result.returncode == 0 and outputs_ok else 'failed'
        item['output_hashes'] = {p: digest(ROOT / p) for p in job['outputs'] if (ROOT / p).is_file()}
        item['ended'] = time.time()
        write(OUT / 'loop-state.json', state)
        if item['status'] == 'failed':
            raise RuntimeError('Job failed; inspect retained log: ' + job['id'])
        completed += 1
        if (OUT / 'dataset-audit.json').exists():
            subprocess.run([sys.executable, str(ROOT / 'scripts/report_ipwm_scale_progress.py')],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        write(OUT / 'loop-status.json', {
            'goal_complete': False, 'status': 'running_registered_jobs',
            'queue_jobs': len(read(OUT / 'queue.json', [])),
            'finished_jobs': sum(j.get('status') == 'complete' for j in state['jobs'].values()),
            'last_completed': job['id']})
    # This audit artifact is produced by the independent evidence validation
    # stage, not by the simulator or by output-file existence checks above.
    queue = read(OUT / 'queue.json', [])
    audit = read(OUT / 'validated-completion.json', {})
    contract = read(CONTRACT, {})
    checks = audit.get('checks', {})
    done = (audit.get('contract_sha256') == digest(CONTRACT)
            and all(checks.get(x) is True for x in contract['completion_requires'])
            and bool(queue)
            and all(state['jobs'].get(j['id'], {}).get('status') == 'complete' for j in queue))
    status = {'goal_complete': done, 'queue_jobs': len(queue),
              'finished_jobs': sum(j.get('status') == 'complete' for j in state['jobs'].values()),
              'remaining_checks': [x for x in contract['completion_requires'] if checks.get(x) is not True],
              'status': 'complete' if done else 'needs_next_registered_batch_or_audit'}
    write(OUT / 'loop-status.json', status)
    print(json.dumps(status, indent=2))

if __name__ == '__main__':
    main()
