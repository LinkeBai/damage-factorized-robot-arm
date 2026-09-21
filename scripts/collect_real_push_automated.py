"""Supervised, repeatable real-robot Push data collection.

The collector owns the camera preview between trials, waits for the cube reset
gate to remain stable, runs one frozen trajectory through the existing runner,
audits the packet, recovers the arm, and resumes the preview.  It deliberately
does not invent trajectories, move the cube, or alter success thresholds.

Use --execute only with an operator beside the powered robot and a tested power
cut / emergency-stop procedure.  Human reset of the cube is still required.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
ACK = "I_HAVE_CLEARED_WORKSPACE_SUPPORTED_ARM_AND_TESTED_ESTOP"
RECOVERY_ACK = "I_HAVE_CLEARED_WORKSPACE_AND_CAN_CUT_POWER"


@dataclass
class TrialRecord:
    trial_id: str
    condition: str
    trajectory_id: str
    started_at: str
    runner_returncode: int | None = None
    audit_returncode: int | None = None
    tracking_returncode: int | None = None
    recovery_returncode: int | None = None
    packet_exists: bool = False
    tracking_summary_exists: bool = False
    radial_endpoint_error_px: float | None = None
    task_success: bool | None = None
    status: str = "PENDING"


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fetch_status(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/cube-status", timeout=1.5) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None


def run_logged(command: list[str], log_path: Path, cwd: Path = ROOT) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return int(completed.returncode)


def start_preview(log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), "scripts/serve_dual_camera_preview.py"],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._collector_log = log  # type: ignore[attr-defined]
    return process


def stop_preview(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=4)
    log = getattr(process, "_collector_log", None)
    if log is not None:
        log.close()


def wait_preview_ready(process: subprocess.Popen[str], url: str, timeout_s: float = 25.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"preview exited with code {process.returncode}")
        if fetch_status(url) is not None:
            return
        time.sleep(0.25)
    raise RuntimeError("preview did not become ready")


def wait_for_stable_reset(
    process: subprocess.Popen[str], url: str, stable_s: float, poll_s: float
) -> dict:
    stable_since: float | None = None
    last_print = 0.0
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"preview exited with code {process.returncode}")
        status = fetch_status(url)
        now = time.monotonic()
        passed = bool(status and status.get("status") == "PASS")
        if passed:
            stable_since = stable_since or now
            if now - stable_since >= stable_s:
                return status or {}
        else:
            stable_since = None
        if now - last_print >= 2.0:
            instruction = (status or {}).get("instruction", "waiting for camera")
            print(f"[{stamp()}] WAIT_RESET: {instruction}", flush=True)
            last_print = now
        time.sleep(poll_s)


def announce(message: str, success: bool) -> None:
    """Report collection status in the terminal without playing audio."""
    print(message, flush=True)


def parse_tracking(trial_dir: Path, record: TrialRecord) -> None:
    path = trial_dir / "offline_yellow_cube_tracking" / "yellow_cube_summary.json"
    record.tracking_summary_exists = path.is_file()
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    image_task = payload.get("image_task", {})
    value = image_task.get("endpoint_error_radial_px")
    if isinstance(value, (int, float)):
        record.radial_endpoint_error_px = float(value)
    value = image_task.get("success")
    if isinstance(value, bool):
        record.task_success = value


def build_schedule(conditions: list[str], repeats: int, prefix: str) -> list[tuple[str, str, str]]:
    result = []
    for repetition in range(1, repeats + 1):
        for condition in conditions:
            result.append((condition, f"{condition}_45px_v1", f"{prefix}-{condition}-{repetition:03d}"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=["intact", "D2", "D3"], default=["D2", "D3"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--trial-prefix", default="final45-day2")
    parser.add_argument("--output-root", type=Path, default=Path("data/real_robot/session_20260901/pilot_trials"))
    parser.add_argument("--waypoints", type=Path, default=Path("data/real_robot/session_20260901/setup/trajectory_candidates_45px_v1.csv"))
    parser.add_argument("--preview-url", default="http://127.0.0.1:8765")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--reset-stable-s", type=float, default=4.0,
                        help="Hands-clear settling time after reset gate passes")
    parser.add_argument("--poll-s", type=float, default=0.2)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--open-browser", action="store_true",
                        help="Open the persistent dual-camera page in the default browser")
    parser.add_argument("--acknowledge-risk", default="")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.execute and args.acknowledge_risk != ACK:
        raise SystemExit(f"--execute requires --acknowledge-risk {ACK}")
    if not args.waypoints.is_file():
        raise SystemExit(f"missing waypoint file: {args.waypoints}")

    schedule = build_schedule(args.conditions, args.repeats, args.trial_prefix)
    session_dir = ROOT / "data/real_robot/session_20260901/automated_collection" / (
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    session_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": stamp(),
        "execute": args.execute,
        "conditions": args.conditions,
        "repeats": args.repeats,
        "schedule": [dict(condition=c, trajectory_id=t, trial_id=i) for c, t, i in schedule],
        "records": [],
        "status": "RUNNING" if args.execute else "DRY_RUN_PASS",
    }
    write_json(session_dir / "collection_manifest.json", manifest)
    if not args.execute:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    preview: subprocess.Popen[str] | None = None
    collection_complete = False
    try:
        preview = start_preview(session_dir / "preview.log")
        wait_preview_ready(preview, args.preview_url)
        if args.open_browser:
            webbrowser.open(args.preview_url + "/?layout=standalone-collector", new=2)
        for condition, trajectory_id, trial_id in schedule:
            trial_dir = ROOT / args.output_root / trial_id
            if trial_dir.exists():
                raise RuntimeError(f"refusing to overwrite existing trial: {trial_dir}")
            record = TrialRecord(trial_id, condition, trajectory_id, stamp())
            manifest["records"].append(asdict(record))
            write_json(session_dir / "collection_manifest.json", manifest)

            reset = wait_for_stable_reset(preview, args.preview_url, args.reset_stable_s, args.poll_s)
            write_json(session_dir / f"{trial_id}_reset_gate.json", reset)
            stop_preview(preview)
            preview = None

            runner = [
                str(PYTHON), "scripts/run_real_push_fixed_trajectory.py",
                "--waypoints", str(args.waypoints), "--trajectory-id", trajectory_id,
                "--condition", condition, "--trial-id", trial_id,
                "--output-root", str(args.output_root), "--port", args.port,
                "--telemetry-hz", "5", "--video-fps", "12", "--pre-roll-s", "1", "--post-roll-s", "1",
                "--maximum-speed-deg-s", "5", "--maximum-start-error-deg", "2",
                "--startup-powered-handoff", "--keep-torque-enabled-after-success", "--execute",
                "--acknowledge-risk", ACK,
            ]
            record.runner_returncode = run_logged(runner, session_dir / f"{trial_id}_runner.log")
            record.packet_exists = trial_dir.is_dir()

            if record.runner_returncode == 0 and trial_dir.is_dir():
                record.audit_returncode = run_logged(
                    [str(PYTHON), "scripts/audit_real_robot_trial_packet.py", str(trial_dir)],
                    session_dir / f"{trial_id}_audit.log",
                )
                record.tracking_returncode = run_logged(
                    [str(PYTHON), "scripts/track_yellow_cube_video.py", str(trial_dir),
                     "--output-dir", str(trial_dir / "offline_yellow_cube_tracking"),
                     "--task-goal-px", "1147.34,580.78", "--task-goal-box-wh-px", "40,40",
                     "--write-endpoint-images"],
                    session_dir / f"{trial_id}_tracking.log",
                )
                parse_tracking(trial_dir, record)

            recovery_dir = ROOT / "data/real_robot/session_20260901/setup" / f"auto_recover_{trial_id}"
            record.recovery_returncode = run_logged(
                [str(PYTHON), "scripts/recover_to_frozen_start.py", "--port", args.port,
                 "--target", "2085,2635,2603,2740,2077", "--speed-deg-s", "1.5",
                 "--period-s", "0.1", "--settle-timeout-s", "20",
                 "--load-compensation-max-ticks", "40", "--leave-torque-enabled-on-success",
                 "--acknowledge-risk", RECOVERY_ACK, "--output-dir", str(recovery_dir)],
                session_dir / f"{trial_id}_recovery.log",
            )
            record.status = "PASS" if (
                record.runner_returncode == 0 and record.audit_returncode == 0
                and record.tracking_returncode == 0 and record.recovery_returncode == 0
            ) else "FAIL"
            manifest["records"][-1] = asdict(record)
            write_json(session_dir / "collection_manifest.json", manifest)
            announce("试验成功了，请复位方块" if record.status == "PASS" else "任务失败了，采集器已停止", record.status == "PASS")
            if record.status != "PASS":
                raise RuntimeError(f"trial {trial_id} failed; see {session_dir}")

            preview = start_preview(session_dir / "preview.log")
            wait_preview_ready(preview, args.preview_url)

        manifest["status"] = "COMPLETE"
        manifest["completed_at"] = stamp()
        write_json(session_dir / "collection_manifest.json", manifest)
        announce("全部试验采集完成", True)
        collection_complete = True
        return 0
    except KeyboardInterrupt:
        manifest["status"] = "INTERRUPTED"
        manifest["ended_at"] = stamp()
        write_json(session_dir / "collection_manifest.json", manifest)
        return 130
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = str(exc)
        manifest["ended_at"] = stamp()
        write_json(session_dir / "collection_manifest.json", manifest)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        # After a clean completion leave the final preview child alive so the
        # operator never loses the two camera views. On failures/interruption,
        # close it deterministically and leave the hardware state explicit.
        if collection_complete and preview is not None and preview.poll() is None:
            log = getattr(preview, "_collector_log", None)
            if log is not None:
                log.close()
        else:
            stop_preview(preview)


if __name__ == "__main__":
    raise SystemExit(main())
