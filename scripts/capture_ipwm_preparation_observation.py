"""Save a fresh read-only observation for candidate preparation; never move servos."""
import argparse
import json
import time
import urllib.request
from pathlib import Path
from recover_j5 import ServoBus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite observation')
    bus = ServoBus('COM3')
    try:
        joint_raw = [bus.read_u16(i, 56) for i in range(1, 6)]
    finally:
        bus.close()
    with urllib.request.urlopen('http://127.0.0.1:8765/cube-status', timeout=5) as response:
        cube = json.load(response)
    if cube.get('status') != 'PASS':
        raise SystemExit('Cube start gate is not ready')
    payload = dict(schema_version=1, joint_raw=joint_raw,
                   object_pixel=cube['current_px'], goal_pixel=cube['task_goal_px'],
                   captured_monotonic_ns=time.monotonic_ns(), read_only=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload))
