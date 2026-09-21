"""Serve a live dual-camera MJPEG alignment preview on localhost."""

from __future__ import annotations

import threading
import time
import json
import math
import os
import sys
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

try:
    from .audit_daheng_camera import configure_sdk
except ImportError:  # Direct script execution keeps scripts/ on sys.path.
    from audit_daheng_camera import configure_sdk


def fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox a raw camera frame without drawing recording overlays."""
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, None, fx=scale, fy=scale)
    canvas = np.zeros((height, width, 3), np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


FRAME = None
RAW_OVERHEAD = None
RAW_WRIST = None
LOCK = threading.Lock()
FRAME_NUMBER = 0
FRAME_UPDATED_MONOTONIC = None
DEFAULT_SETTINGS = {"daheng_exposure": 3000.0, "daheng_gain": 12.0,
                    "second_exposure": -4.0, "second_brightness": 0.0}
SETTING_RANGES = {
    "daheng_exposure": (200.0, 12000.0),
    "daheng_gain": (0.0, 24.0),
    "second_exposure": (-8.0, -1.0),
    "second_brightness": (-64.0, 64.0),
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
from robotarm.analysis.yellow_cube_tracker import TrackerConfig, detect_yellow_cube

FROZEN_SETTINGS_PATH = REPOSITORY_ROOT / "results" / "real_robot" / "camera_settings_selected.json"
PREVIEW_SESSION_SETTINGS_PATH = (
    REPOSITORY_ROOT / "results" / "real_robot" / "camera_settings_preview_session.json"
)
SETTINGS = dict(DEFAULT_SETTINGS)
PENDING = dict(SETTINGS)
SETTINGS_SOURCE = "built_in_defaults_not_initialized"
PERSIST_LOCK = threading.Lock()
# Original arm home is retained.  The task overlay is re-aligned to the measured
# gripper push-face row after homing; robot configuration is not redefined here.
JOINT_TARGET = (2085, 2635, 2603, 2740, 2077)
JOINT_STATUS = {"status": "starting", "target_raw": JOINT_TARGET}
CUBE_TARGET_PX = tuple(float(v) for v in os.environ.get(
    "ROBOTARM_PREVIEW_START_PX", "1086.31,570.00").split(","))
if len(CUBE_TARGET_PX) != 2 or not all(np.isfinite(v) for v in CUBE_TARGET_PX):
    raise ValueError("preview start must be two finite pixel coordinates")
# Manual reset is an initialization gate, not the task-success criterion.
# Five pixels avoids wasting trials on sub-pixel detector jitter while every
# run still records its measured start for paired endpoint analysis.
CUBE_TOLERANCE_PX = float(os.environ.get("ROBOTARM_PREVIEW_START_TOLERANCE_PX", "5.0"))
if not 0 < CUBE_TOLERANCE_PX <= 5:
    raise ValueError("preview start tolerance must be in (0,5]")
CUBE_TARGET_BOX_WH_PX = (40, 40)
TASK_GOAL_PX = (1106.31, 570.00)  # Final-day intact/single-lock: 20 px.
TASK_GOAL_BOX_WH_PX = (40, 40)
TASK_GOAL_TOLERANCE_PX = 5.0
TASK_GOAL_GATE = "axis_aligned_per_axis"
CUBE_STATUS = {"status": "starting", "target_px": CUBE_TARGET_PX,
               "tolerance_px": CUBE_TOLERANCE_PX,
               "target_box_wh_px": CUBE_TARGET_BOX_WH_PX}
GRIPPER_STATUS = {"status": "starting"}
MICROSTEP_QUEUE: queue.Queue = queue.Queue()
MICROSTEP_STATUS = {"status": "idle"}
MICROSTEP_ACK = "I_ACKNOWLEDGE_SUPERVISED_MICROSTEP"
MICROSTEP_ENABLED = False  # Manual cube reset restored; keep preview read-only.
FORMAL_TRIAL_PREFIX = "push30-v6-"
# Final on-site scope amendment (2026-09-03): one genuine qualitative
# feasibility trial per condition. Earlier n=10 and n=3 protocols remain on
# disk and must not be described as completed by this display.
FORMAL_REQUIRED_PER_CONDITION = 1
FORMAL_CONDITIONS = ("intact", "D2", "D3")
FORMAL_TRAJECTORY_IDS = {
    "intact": "intact_45px_v1",
    "D2": "D2_cartesian_level_freewrist_60mm_v1",
    "D3": "D3_cartesian_level_v1",
}
FORMAL_TRIAL_ROOT = (
    REPOSITORY_ROOT / "data" / "real_robot" / "session_20260901"
    / "confirmatory_trials"
)
ALL_EVIDENCE_LEDGER = (
    REPOSITORY_ROOT / "results" / "real_robot"
    / "all_real_push_evidence_ledger_20260903.json"
)


def formal_experiment_progress() -> dict:
    """Read formal v6 acquisition progress without mutating trial evidence."""
    counts = {condition: 0 for condition in FORMAL_CONDITIONS}
    outcomes = {
        condition: {"success": 0, "fail": 0, "unassessed": 0}
        for condition in FORMAL_CONDITIONS
    }
    trial_ids: list[str] = []
    if FORMAL_TRIAL_ROOT.is_dir():
        for trial in sorted(FORMAL_TRIAL_ROOT.glob(f"{FORMAL_TRIAL_PREFIX}*")):
            manifest_path = trial / "run_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                packet_audit = json.loads(
                    (trial / "packet_audit.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            condition = manifest.get("condition")
            if manifest.get("trajectory_id") != FORMAL_TRAJECTORY_IDS.get(condition):
                continue
            acquisition_valid = manifest.get("status") == "ACQUISITION_COMPLETE_UNASSESSED"
            adjudicated_valid = False
            adjudication_path = trial / "post_acquisition_adjudication.json"
            if not acquisition_valid and adjudication_path.is_file():
                try:
                    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
                    adjudicated_valid = adjudication.get("recovered_valid_trial") is True
                    acquisition_valid = adjudicated_valid
                except (OSError, json.JSONDecodeError):
                    acquisition_valid = False
            if (acquisition_valid and packet_audit.get("packet_integrity_status") == "PASS"
                    and (packet_audit.get("valid_trial") is True or adjudicated_valid)
                    and condition in counts):
                counts[condition] += 1
                trial_ids.append(str(manifest.get("trial_id", trial.name)))
                summary_path = (
                    trial / "offline_cube_tracking_v1" / "yellow_cube_summary.json"
                )
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    task = summary["image_task"]
                    confidence_ok = summary.get("confidence_gate", {}).get("pass") is True
                    assessed = task.get("assessed") is True and confidence_ok
                    success = task.get("success") is True
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    assessed = False
                    success = False
                if not assessed:
                    outcomes[condition]["unassessed"] += 1
                elif success:
                    outcomes[condition]["success"] += 1
                else:
                    outcomes[condition]["fail"] += 1
    total_attempts = sum(counts.values())
    # Quota is condition-balanced: repeated successes in one condition cannot
    # stand in for a missing intact/D2/D3 condition.
    completed_successes = sum(
        min(item["success"], FORMAL_REQUIRED_PER_CONDITION)
        for item in outcomes.values()
    )
    required_total = FORMAL_REQUIRED_PER_CONDITION * len(FORMAL_CONDITIONS)
    retained_counts = {}
    try:
        retained_counts = json.loads(
            ALL_EVIDENCE_LEDGER.read_text(encoding="utf-8")
        ).get("counts", {})
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "protocol": "formal_push30_protocol_v6_amendment_20260903_n1",
        "formal_trial_prefix": FORMAL_TRIAL_PREFIX,
        "completed_success_quota": completed_successes,
        "required_success_quota": required_total,
        "percent": round(100.0 * completed_successes / required_total, 1),
        "valid_attempts": total_attempts,
        "counts": counts,
        "outcomes": outcomes,
        "required_per_condition": FORMAL_REQUIRED_PER_CONDITION,
        "trial_ids": trial_ids,
        "all_retained_evidence": retained_counts,
        "counting_rule": (
            "complete v6 acquisition, authorized condition-specific trajectory, "
            "and passing packet audit only; calibration and aborts excluded"
        ),
    }


def detect_gripper_push_face(frame: np.ndarray, cube_xy: tuple[float, float] | None) -> dict:
    """Locate the red distal gripper component nearest the yellow cube.

    This is deliberately an observation, not a claim of metric hand-eye
    calibration. The distal red component has two fingers; their vertical
    midline is the image-space push centre. A single closest red pixel can snap
    to one finger whenever the cube is slightly off-axis.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array((0, 135, 75)), np.array((12, 255, 255)))
    red |= cv2.inRange(hsv, np.array((168, 135, 75)), np.array((179, 255, 255)))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(red)
    components = [i for i in range(1, count) if 120 <= stats[i, cv2.CC_STAT_AREA] <= 25000]
    if cube_xy is None or not components:
        return {"status": "NOT_DETECTED", "component_count": len(components)}
    cube = np.asarray(cube_xy, dtype=np.float64)
    component = min(components, key=lambda i: float(np.linalg.norm(centroids[i] - cube)))
    ys, xs = np.where(labels == component)
    points = np.column_stack((xs, ys)).astype(np.float64)
    max_x = float(np.max(points[:, 0]))
    distal = points[points[:, 0] >= max_x - 8.0]
    low_y, high_y = np.percentile(distal[:, 1], (10, 90))
    push_face = np.asarray((max_x, (low_y + high_y) / 2.0), dtype=np.float64)
    relative = cube - push_face
    return {
        "status": "DETECTED",
        "component_area_px": int(stats[component, cv2.CC_STAT_AREA]),
        "component_centroid_px": [round(float(v), 2) for v in centroids[component]],
        "push_face_px": [round(float(v), 2) for v in push_face],
        "cube_minus_push_face_px": [round(float(v), 2) for v in relative],
        "distance_to_cube_px": round(float(np.linalg.norm(relative)), 2),
    }


def annotate_cube_reset(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    """Draw reset guidance on a preview copy; formal raw video stays untouched."""
    preview = frame.copy()
    # The v2 start places the beige/yellow cube partly inside the open gripper.
    # A frozen task ROI and slightly wider hue bound separate it from the red
    # arm and prevent the old full-field detector from latching floor marks.
    detection = detect_yellow_cube(
        frame,
        TrackerConfig(
            hsv_lower=(10, 80, 90),
            hsv_upper=(35, 255, 255),
            min_area_px=300,
            min_component_confidence=0.35,
            roi_xywh=(1000, 450, 350, 250),
        ),
    )
    tx, ty = CUBE_TARGET_PX
    target_w, target_h = CUBE_TARGET_BOX_WH_PX
    # Equal-size, axis-aligned start/goal footprints. These overlays exist only
    # in the live preview; raw formal videos remain untouched.
    cv2.rectangle(
        preview,
        (round(tx - target_w / 2), round(ty - target_h / 2)),
        (round(tx + target_w / 2), round(ty + target_h / 2)),
        (0, 255, 0), 3,
    )
    cv2.drawMarker(preview, (round(tx), round(ty)), (0, 255, 0),
                   cv2.MARKER_CROSS, 22, 3)
    gx, gy = TASK_GOAL_PX
    goal_w, goal_h = TASK_GOAL_BOX_WH_PX
    goal_pass = bool(
        detection.detected
        and detection.centroid is not None
        and abs(detection.centroid[0] - gx) <= TASK_GOAL_TOLERANCE_PX
        and abs(detection.centroid[1] - gy) <= TASK_GOAL_TOLERANCE_PX
    )
    goal_color = (0, 255, 0) if goal_pass else (255, 0, 255)
    # A high-contrast datum makes the re-aligned row unmistakable even when the
    # old and new y coordinates differ by only a few pixels.
    guide_x0 = max(0, round(tx - target_w / 2 - 100))
    guide_x1 = min(preview.shape[1] - 1, round(gx + goal_w / 2 + 60))
    cv2.line(preview, (guide_x0, round(ty)), (guide_x1, round(ty)),
             (255, 255, 0), 3, cv2.LINE_AA)
    cv2.rectangle(
        preview,
        (round(gx - goal_w / 2), round(gy - goal_h / 2)),
        (round(gx + goal_w / 2), round(gy + goal_h / 2)),
        goal_color, 4,
    )
    cv2.drawMarker(preview, (round(gx), round(gy)), goal_color,
                   cv2.MARKER_CROSS, 22, 3)
    cv2.rectangle(
        preview,
        (round(gx - TASK_GOAL_TOLERANCE_PX),
         round(gy - TASK_GOAL_TOLERANCE_PX)),
        (round(gx + TASK_GOAL_TOLERANCE_PX),
         round(gy + TASK_GOAL_TOLERANCE_PX)),
        goal_color, 3,
    )
    cv2.line(preview, (round(tx + target_w / 2), round(ty)),
             (round(gx - goal_w / 2), round(gy)), (255, 255, 255), 2,
             cv2.LINE_AA)
    cv2.putText(preview, "RESET START", (round(tx - 72), round(ty + target_h / 2 + 30)),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2, cv2.LINE_AA)
    goal_label = "TASK SUCCESS" if goal_pass else "TASK GOAL"
    cv2.putText(preview, goal_label, (round(gx - 65), round(gy - goal_h / 2 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, .75, goal_color, 2, cv2.LINE_AA)
    coord_label = (f"ALIGNED ROW y={ty:.0f}  START=({tx:.0f},{ty:.0f})  "
                   f"GOAL=({gx:.0f},{gy:.0f})  DIST={gx - tx:.0f}px")
    cv2.rectangle(preview, (10, 70), (850, 112), (0, 0, 0), -1)
    cv2.putText(preview, coord_label, (22, 100), cv2.FONT_HERSHEY_SIMPLEX,
                .68, (255, 255, 0), 2, cv2.LINE_AA)
    if not detection.detected or detection.centroid is None:
        status = {"status": "NOT_DETECTED", "target_px": [tx, ty],
                  "tolerance_px": CUBE_TOLERANCE_PX,
                  "task_success": None,
                  "task_goal_gate": TASK_GOAL_GATE,
                  "task_goal_tolerance_each_axis_px": TASK_GOAL_TOLERANCE_PX}
        label = "CUBE NOT DETECTED"
        color = (0, 0, 255)
    else:
        cx, cy = detection.centroid
        dx, dy = cx - tx, cy - ty
        goal_dx, goal_dy = cx - TASK_GOAL_PX[0], cy - TASK_GOAL_PX[1]
        passed = max(abs(dx), abs(dy)) <= CUBE_TOLERANCE_PX
        status = {"status": "PASS" if passed else "ADJUST",
                  "current_px": [round(cx, 2), round(cy, 2)],
                  "target_px": [tx, ty], "error_px": [round(dx, 2), round(dy, 2)],
                  "task_goal_px": list(TASK_GOAL_PX),
                  "task_goal_error_px": [round(goal_dx, 2), round(goal_dy, 2)],
                  "task_success": goal_pass,
                  "task_goal_gate": TASK_GOAL_GATE,
                  "task_goal_tolerance_each_axis_px": TASK_GOAL_TOLERANCE_PX,
                  "tolerance_px": CUBE_TOLERANCE_PX,
                  "instruction": ("READY" if passed else
                    f"move {'left' if dx > 0 else 'right'} {abs(dx):.0f}px, "
                    f"{'up' if dy > 0 else 'down'} {abs(dy):.0f}px")}
        color = (0, 255, 0) if passed else (0, 165, 255)
        if None not in (
            detection.bbox_x_px, detection.bbox_y_px,
            detection.bbox_width_px, detection.bbox_height_px,
        ):
            x, y = int(detection.bbox_x_px), int(detection.bbox_y_px)
            w, h = int(detection.bbox_width_px), int(detection.bbox_height_px)
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 3)
        cv2.circle(preview, (round(cx), round(cy)), 14, color, 4)
        task_label = "TASK SUCCESS" if goal_pass else "TASK NOT YET SUCCESS"
        label = (f"{task_label} | {status['status']} cube=({cx:.0f},{cy:.0f}) "
                 f"start=({dx:+.0f},{dy:+.0f}) goal=({goal_dx:+.0f},{goal_dy:+.0f}) "
                 f"{status['instruction']}")
    cv2.rectangle(preview, (10, 10), (980, 62), (0, 0, 0), -1)
    cv2.putText(preview, label, (22, 48), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, cv2.LINE_AA)
    return preview, status


def validate_camera_settings(payload: object, source: Path | str) -> dict[str, float]:
    """Validate a complete settings document before it can reach the cameras."""
    if not isinstance(payload, dict):
        raise ValueError(f"camera settings in {source} must be a JSON object")
    expected = set(SETTING_RANGES)
    supplied = set(payload)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(f"camera settings in {source} have missing={missing}, extra={extra}")
    validated = {}
    for key, (minimum, maximum) in SETTING_RANGES.items():
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} in {source} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
            raise ValueError(f"{key} in {source} must be within [{minimum}, {maximum}]")
        validated[key] = numeric
    return validated


def read_settings_file(path: Path) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required camera settings file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in camera settings file {path}: {error}") from error
    return validate_camera_settings(payload, path)


def load_startup_settings(
    frozen_path: Path = FROZEN_SETTINGS_PATH,
    session_path: Path = PREVIEW_SESSION_SETTINGS_PATH,
) -> tuple[dict[str, float], str]:
    """Load the validated frozen baseline, then an optional preview-only session."""
    frozen = read_settings_file(frozen_path)
    if session_path.exists():
        return read_settings_file(session_path), "preview_session"
    return frozen, "formal_frozen"


def initialize_settings(
    frozen_path: Path = FROZEN_SETTINGS_PATH,
    session_path: Path = PREVIEW_SESSION_SETTINGS_PATH,
) -> dict[str, float]:
    global SETTINGS_SOURCE
    loaded, source = load_startup_settings(frozen_path, session_path)
    with LOCK:
        SETTINGS.clear(); SETTINGS.update(loaded)
        PENDING.clear(); PENDING.update(loaded)
        SETTINGS_SOURCE = source
    return dict(loaded)


def persist_preview_settings(path: Path | None = None) -> None:
    """Atomically persist desired preview values without touching the frozen file."""
    global SETTINGS_SOURCE
    destination = path if path is not None else PREVIEW_SESSION_SETTINGS_PATH
    if destination.resolve() == FROZEN_SETTINGS_PATH.resolve():
        raise ValueError("preview settings must not overwrite the formal frozen settings")
    with PERSIST_LOCK:
        with LOCK:
            snapshot = validate_camera_settings(dict(PENDING), "in-memory preview settings")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        SETTINGS_SOURCE = "preview_session"


def apply_camera_settings(features, second, settings: dict[str, float]) -> None:
    """Apply one validated settings snapshot to both camera backends."""
    validated = validate_camera_settings(settings, "camera apply request")
    # Older/mock Galaxy feature surfaces may expose only numeric controls.
    # Keep preview compatibility while enabling continuous AWB whenever the
    # connected Daheng SDK exposes the enum feature.
    if hasattr(features, "get_enum_feature"):
        features.get_enum_feature("BalanceWhiteAuto").set("Continuous")
    features.get_float_feature("ExposureTime").set(validated["daheng_exposure"])
    features.get_float_feature("Gain").set(validated["daheng_gain"])
    second.set(cv2.CAP_PROP_EXPOSURE, validated["second_exposure"])
    second.set(cv2.CAP_PROP_BRIGHTNESS, validated["second_brightness"])


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        request_path = urlsplit(self.path).path
        if request_path != "/microstep":
            self.send_error(404); return
        if not MICROSTEP_ENABLED:
            self.send_error(410, "robot motion endpoint disabled; manual reset protocol active"); return
        query = parse_qs(urlsplit(self.path).query)
        try:
            joint = int(query.get("joint", [""])[0])
            delta_raw = int(query.get("delta_raw", [""])[0])
            ack = query.get("ack", [""])[0]
        except ValueError:
            self.send_error(400, "joint and delta_raw must be integers"); return
        if ack != MICROSTEP_ACK:
            self.send_error(403, "explicit supervised microstep acknowledgement required"); return
        if joint not in range(1, 6) or delta_raw == 0 or abs(delta_raw) > 50:
            self.send_error(400, "joint must be 1..5 and abs(delta_raw) must be 1..50"); return
        request = {"joint": joint, "delta_raw": delta_raw, "done": threading.Event()}
        MICROSTEP_QUEUE.put(request)
        if not request["done"].wait(timeout=3.0):
            self.send_error(504, "microstep controller timeout"); return
        payload = json.dumps(request["result"]).encode()
        status = 200 if request["result"].get("status") == "PASS" else 409
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self._no_cache_headers(); self.send_header("Content-Length", str(len(payload)))
        self.end_headers(); self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        request_path = urlsplit(self.path).path
        if request_path in {"/", "/index.html"}:
            body = b"""<!doctype html><meta charset=utf-8><title>Dual camera alignment</title>
<style>body{margin:0;background:#111;color:#eee;font:15px sans-serif;text-align:center}img{display:block;width:100vw;height:auto}.controls{display:flex;gap:18px;justify-content:center;flex-wrap:wrap;padding:10px}.c{white-space:nowrap}input{width:160px}</style>
<img id=v><div class=controls>
<label class=c>Daheng exposure <input id=de type=range min=200 max=12000 step=100 disabled><span></span></label>
<label class=c>Daheng gain <input id=dg type=range min=0 max=24 step=.5 disabled><span></span></label>
<label class=c>Second exposure <input id=se type=range min=-8 max=-1 step=1 disabled><span></span></label>
<label class=c>Second brightness <input id=sb type=range min=-64 max=64 step=1 disabled><span></span></label>
</div><div id=progress>Formal experiment progress: loading...</div><div id=live>Camera preview: connecting...</div><div id=cube>Cube reset gate: starting</div><div id=gripper>Gripper visual servo: starting</div><div id=j>Joint read-only status: starting</div>
<script>
const v=document.getElementById('v');
const live=document.getElementById('live');
let currentObjectUrl=null;
async function refreshFrame(){
  let delayMs=120, pendingObjectUrl=null;
  const controller=new AbortController();
  const timeout=setTimeout(()=>controller.abort(),2500);
  try{
    const response=await fetch('/snapshot.jpg?t='+Date.now(),{
      cache:'no-store',signal:controller.signal,headers:{'Cache-Control':'no-cache'}
    });
    if(!response.ok)throw new Error('HTTP '+response.status);
    pendingObjectUrl=URL.createObjectURL(await response.blob());
    const previousObjectUrl=currentObjectUrl;
    await new Promise((resolve,reject)=>{
      const decodeTimeout=setTimeout(()=>{
        v.onload=null;v.onerror=null;reject(new Error('JPEG decode timeout'));
      },1500);
      v.onload=()=>{clearTimeout(decodeTimeout);resolve()};
      v.onerror=()=>{clearTimeout(decodeTimeout);reject(new Error('JPEG decode failed'))};
      v.src=pendingObjectUrl;
    });
    currentObjectUrl=pendingObjectUrl;pendingObjectUrl=null;
    if(previousObjectUrl)URL.revokeObjectURL(previousObjectUrl);
    live.textContent='Camera preview: live (frame '+(response.headers.get('X-Frame-Number')||'?')+')';
  }catch(error){
    delayMs=500;
    live.textContent='Camera preview: reconnecting ('+(error.name==='AbortError'?'request timeout':error.message)+')';
  }finally{
    clearTimeout(timeout);
    if(pendingObjectUrl)URL.revokeObjectURL(pendingObjectUrl);
    setTimeout(refreshFrame,delayMs);
  }
}
refreshFrame();
const ids=['de','dg','se','sb'], names=['daheng_exposure','daheng_gain','second_exposure','second_brightness'];
async function initializeControls(){
  try{
    const response=await fetch('/settings?t='+Date.now(),{cache:'no-store'});
    if(!response.ok)throw new Error('HTTP '+response.status);
    const settings=await response.json();
    ids.forEach((id,i)=>{
      const e=document.getElementById(id),s=e.nextElementSibling;
      e.value=String(settings[names[i]]);s.textContent=' '+e.value;e.disabled=false;
      let sendTimer=null;
      e.oninput=()=>{
        s.textContent=' '+e.value;clearTimeout(sendTimer);
        sendTimer=setTimeout(()=>{
          const query=new URLSearchParams({[names[i]]:e.value});
          fetch('/settings?'+query.toString(),{cache:'no-store'}).catch(()=>{});
        },80);
      };
    });
  }catch(error){setTimeout(initializeControls,500)}
}
initializeControls();
async function cube(){try{const x=await (await fetch('/cube-status?t='+Date.now(),{cache:'no-store'})).json();document.getElementById('cube').textContent='Cube reset gate: '+x.status+(x.current_px?' current '+x.current_px.join(', ')+' target '+x.target_px.join(', ')+' error '+x.error_px.join(', ')+' px; '+x.instruction:'')}catch(e){}setTimeout(cube,200)}cube();
async function gripper(){try{const x=await (await fetch('/gripper-status?t='+Date.now(),{cache:'no-store'})).json();document.getElementById('gripper').textContent=x.status==='DETECTED'?'Gripper push face '+x.push_face_px.join(', ')+' cube-face error '+x.cube_minus_push_face_px.join(', ')+' px; distance '+x.distance_to_cube_px+' px':'Gripper visual servo: '+x.status}catch(e){}setTimeout(gripper,200)}gripper();
async function joints(){try{const x=await (await fetch('/joint-status?t='+Date.now(),{cache:'no-store'})).json();document.getElementById('j').textContent=x.status==='ok'?'Read-only J1-J5 error to frozen start (deg): '+x.error_deg.join(' | '):'Joint read-only status: '+x.status}catch(e){}setTimeout(joints,300)}joints();
async function progress(){try{const x=await (await fetch('/experiment-progress?t='+Date.now(),{cache:'no-store'})).json();const o=x.outcomes,r=x.all_retained_evidence||{};document.getElementById('progress').textContent='V6 formal: '+x.completed_success_quota+'/'+x.required_success_quota+' ('+x.percent+'%) | valid attempts '+x.valid_attempts+' | intact S'+o.intact.success+' F'+o.intact.fail+' | D2 S'+o.D2.success+' F'+o.D2.fail+' | D3 S'+o.D3.success+' F'+o.D3.fail+' || ALL RETAINED: '+(r.all_packets??'?')+' packets, '+(r.successful_under_recorded_criterion??'?')+' valid successes, '+(r.failed_under_recorded_criterion??'?')+' valid failures'}catch(e){}setTimeout(progress,1000)}progress();
</script>"""
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self._no_cache_headers()
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        elif request_path == "/snapshot.jpg":
            with LOCK:
                frame = FRAME
                frame_number = FRAME_NUMBER
                updated_at = FRAME_UPDATED_MONOTONIC
            if frame is None or updated_at is None:
                self.send_error(503); return
            if time.monotonic() - updated_at > 2.5:
                self.send_error(503, "Camera frame is stale"); return
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                self.send_error(500); return
            data = encoded.tobytes()
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self._no_cache_headers()
            self.send_header("X-Frame-Number", str(frame_number))
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        elif request_path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with LOCK:
                        frame = FRAME
                    if frame is None:
                        time.sleep(.05); continue
                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        data = encoded.tobytes()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    time.sleep(.04)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif request_path in {"/overhead-raw.jpg", "/wrist-raw.jpg"}:
            with LOCK:
                frame = RAW_OVERHEAD if request_path.startswith("/overhead") else RAW_WRIST
            if frame is None:
                self.send_error(503, "raw frame not ready"); return
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                self.send_error(500, "jpeg encode failed"); return
            payload = encoded.tobytes()
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self._no_cache_headers(); self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        elif request_path == "/settings":
            query = parse_qs(urlsplit(self.path).query)
            changed = any(key in query for key in SETTING_RANGES)
            try:
                with LOCK:
                    candidate = dict(PENDING)
                    for key in SETTING_RANGES:
                        if key in query:
                            candidate[key] = float(query[key][0])
                    validated = validate_camera_settings(candidate, "HTTP settings request")
                    if changed:
                        PENDING.clear(); PENDING.update(validated)
                    payload = json.dumps(PENDING).encode()
                if changed:
                    persist_preview_settings()
            except (TypeError, ValueError) as error:
                self.send_error(400, str(error)); return
            except OSError as error:
                self.send_error(500, f"could not persist preview settings: {error}"); return
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self._no_cache_headers(); self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        elif request_path == "/joint-status":
            with LOCK:
                payload = json.dumps(JOINT_STATUS).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        elif request_path == "/cube-status":
            with LOCK:
                payload = json.dumps(CUBE_STATUS).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self._no_cache_headers(); self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        elif request_path == "/experiment-progress":
            payload = json.dumps(formal_experiment_progress()).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self._no_cache_headers(); self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        elif request_path == "/gripper-status":
            with LOCK:
                payload = json.dumps(GRIPPER_STATUS).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self._no_cache_headers(); self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        elif request_path == "/microstep-status":
            with LOCK:
                payload = json.dumps(MICROSTEP_STATUS).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self._no_cache_headers(); self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        elif request_path == "/favicon.ico":
            self.send_response(204); self._no_cache_headers(); self.end_headers()
        else:
            self.send_error(404)

    def _no_cache_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def log_message(self, *_):
        return


def capture() -> None:
    global FRAME, RAW_OVERHEAD, RAW_WRIST, FRAME_NUMBER, FRAME_UPDATED_MONOTONIC
    global CUBE_STATUS, GRIPPER_STATUS
    configure_sdk(Path(r"D:\GalaxySDK"))
    import gxipy as gx  # type: ignore
    manager = gx.DeviceManager(); manager.update_device_list(1500)
    daheng = manager.open_device_by_sn("FDE23080341")
    features = daheng.get_remote_device_feature_control()
    features.get_enum_feature("TriggerMode").set("Off")
    features.get_enum_feature("BalanceWhiteAuto").set("Continuous")
    with LOCK:
        initial_settings = dict(PENDING)
    second = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    apply_camera_settings(features, second, initial_settings)
    with LOCK:
        SETTINGS.clear(); SETTINGS.update(initial_settings)
    daheng.stream_on()
    try:
        while True:
            with LOCK:
                requested = dict(PENDING)
            if requested != SETTINGS:
                apply_camera_settings(features, second, requested)
                with LOCK:
                    SETTINGS.update(requested)
            raw = daheng.data_stream[0].get_image(1000); ok, other = second.read()
            if raw is None or not ok: continue
            first = cv2.cvtColor(raw.convert("RGB").get_numpy_array(), cv2.COLOR_RGB2BGR)
            annotated_first, cube_status = annotate_cube_reset(first)
            cube_xy = tuple(cube_status["current_px"]) if "current_px" in cube_status else None
            gripper_status = detect_gripper_push_face(first, cube_xy)
            # Keep the overhead field large enough for precise manual reset;
            # place the secondary view below instead of shrinking both side by side.
            combined = np.vstack((fit(annotated_first, 960, 600),
                                  fit(other, 960, 540)))
            with LOCK:
                FRAME = combined
                RAW_OVERHEAD = first.copy()
                RAW_WRIST = other.copy()
                CUBE_STATUS = cube_status
                GRIPPER_STATUS = gripper_status
                FRAME_NUMBER += 1
                FRAME_UPDATED_MONOTONIC = time.monotonic()
            time.sleep(.03)
    finally:
        daheng.stream_off(); daheng.close_device(); second.release()


def read_joints() -> None:
    global JOINT_STATUS, MICROSTEP_STATUS
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from recover_j5 import ServoBus
    while True:
        bus = None
        try:
            bus = ServoBus("COM3")
            while True:
                current = tuple(bus.read_u16(i, 56) for i in range(1, 6))
                errors = [round(abs(a - b) * 360 / 4096, 3)
                          for a, b in zip(current, JOINT_TARGET)]
                with LOCK:
                    JOINT_STATUS = {"status": "ok", "current_raw": current,
                                    "target_raw": JOINT_TARGET, "error_deg": errors,
                                    "read_only": True}
                try:
                    request = MICROSTEP_QUEUE.get_nowait()
                except queue.Empty:
                    request = None
                if request is not None:
                    joint = request["joint"]
                    delta_raw = request["delta_raw"]
                    result = {"status": "FAIL", "joint": joint, "delta_raw": delta_raw}
                    try:
                        mode = bus.read_u8(joint, 33)
                        torque = bus.read_u8(joint, 40)
                        voltage_v = bus.read_u8(joint, 62) / 10
                        temperature_c = bus.read_u8(joint, 63)
                        current_raw = bus.read_u16(joint, 69)
                        if current_raw >= 32768:
                            current_raw -= 65536
                        if mode != 0 or torque != 1:
                            raise RuntimeError(f"mode={mode} torque={torque}; powered position hold required")
                        if voltage_v < 6 or temperature_c >= 50 or abs(current_raw) > 400:
                            raise RuntimeError(
                                f"electrical gate voltage={voltage_v} temp={temperature_c} current={current_raw}"
                            )
                        before = bus.read_u16(joint, 56)
                        target = before + delta_raw
                        if not 900 <= target <= 3200:
                            raise RuntimeError(f"target {target} outside conservative envelope")
                        bus.write_u16(joint, 42, target)
                        time.sleep(0.05)
                        bus.write_u16(joint, 42, target)
                        time.sleep(0.45)
                        after = bus.read_u16(joint, 56)
                        result.update({"status": "PASS", "before_raw": before,
                                       "target_raw": target, "after_raw": after,
                                       "tracking_error_raw": after - target,
                                       "voltage_v": voltage_v,
                                       "temperature_c": temperature_c,
                                       "current_raw": current_raw})
                    except Exception as error:
                        result["error"] = str(error)
                    with LOCK:
                        MICROSTEP_STATUS = result
                    request["result"] = result
                    request["done"].set()
                    MICROSTEP_QUEUE.task_done()
                time.sleep(.25)
        except Exception as error:
            with LOCK:
                JOINT_STATUS = {"status": str(error), "target_raw": JOINT_TARGET,
                                "read_only": True}
            time.sleep(1.0)
        finally:
            if bus is not None:
                bus.close()


if __name__ == "__main__":
    initialize_settings()
    threading.Thread(target=capture, daemon=True).start()
    # Formal trial executors need exclusive ownership of the STS bus.  Keep the
    # cameras live without opening COM3 when this environment flag is set.
    if os.environ.get("ROBOTARM_PREVIEW_CAMERA_ONLY", "0") != "1":
        threading.Thread(target=read_joints, daemon=True).start()
    else:
        with LOCK:
            JOINT_STATUS = {
                "status": "camera-only preview; COM3 released for trial executor",
                "target_raw": JOINT_TARGET,
                "read_only": True,
            }
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
