"""Read-only browser monitor fed by formal trial recorders.

Raw frames are written by the recorder before they are copied here.  All
boxes, labels, and trails therefore exist only in this disposable preview.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2
import numpy as np

from robotarm.analysis.yellow_cube_tracker import TrackerConfig, detect_yellow_cube


class LiveTrialMonitor:
    def __init__(self, start_px=(1086.31, 570.00), task_goal_px=(1116.31, 570.00), target_box_wh=(40, 40), goal_tolerance_px=5.0, gate_mode="axiswise", stable_goal_frames=3, port=8765):
        self.reset_start_px = tuple(start_px)
        self.task_goal_px = tuple(task_goal_px)
        self.target_box_wh = tuple(target_box_wh)
        self.goal_tolerance_px = float(goal_tolerance_px)
        if gate_mode not in {"axiswise", "radial"}:
            raise ValueError("gate_mode must be axiswise or radial")
        self.gate_mode = gate_mode
        self.stable_goal_frames = int(stable_goal_frames)
        if self.stable_goal_frames < 1:
            raise ValueError("stable_goal_frames must be positive")
        self.port = int(port)
        self.lock = threading.Lock()
        self.frames: dict[str, np.ndarray] = {}
        self.published_frame_sequences: dict[str, int] = {}
        self.preview: np.ndarray | None = None
        self.status = {"status": "STARTING", "reset_start_px": list(self.reset_start_px), "task_goal_px": list(self.task_goal_px)}
        self.start_px: tuple[float, float] | None = None
        self.last_px: tuple[float, float] | None = None
        self.trail: list[tuple[int, int]] = []
        self.gate_samples: list[dict] = []
        self.goal_pass_streak = 0
        self.maximum_goal_pass_streak = 0
        self.stable_goal_ever = False
        self.first_stable_goal_monotonic_ns: int | None = None
        self.stop_event = threading.Event()
        self.frame_number = 0
        self.server: ThreadingHTTPServer | None = None
        self.server_thread: threading.Thread | None = None
        self.worker = threading.Thread(target=self._work, daemon=True)

    def publish(self, camera: str, frame: np.ndarray) -> None:
        with self.lock:
            self.frames[camera] = frame.copy()
            self.published_frame_sequences[camera] = (
                self.published_frame_sequences.get(camera, 0) + 1
            )

    def gate_summary(self) -> dict:
        """Return an immutable snapshot of the live 3-pixel gate evidence."""
        with self.lock:
            samples = [dict(sample) for sample in self.gate_samples]
            return {
                "criterion": (
                    f"at least {self.stable_goal_frames} consecutive reliable overhead "
                    f"detections with {self.gate_mode} endpoint error <= {self.goal_tolerance_px:g} px"
                ),
                "goal_px": list(self.task_goal_px),
                "gate_mode": self.gate_mode,
                "tolerance_px": self.goal_tolerance_px,
                "required_consecutive_frames": self.stable_goal_frames,
                "processed_overhead_samples": len(samples),
                "maximum_consecutive_pass_frames": self.maximum_goal_pass_streak,
                "stable_goal_ever": self.stable_goal_ever,
                "first_stable_goal_monotonic_ns": self.first_stable_goal_monotonic_ns,
                "samples": samples,
            }

    def stable_goal_reached(self) -> bool:
        """Return whether the frozen live endpoint gate has ever passed."""
        with self.lock:
            return bool(self.stable_goal_ever)

    def goal_plane_crossed(self) -> bool:
        """Return whether the cube reached the plane through the task goal.

        This is only a motion-stop guard.  It never changes the radial success
        criterion, so an off-axis crossing remains a recorded task failure.
        """
        with self.lock:
            current = self.last_px
        if current is None:
            return False
        direction = np.asarray(self.task_goal_px) - np.asarray(self.reset_start_px)
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            return False
        progress = float(
            np.dot(np.asarray(current) - np.asarray(self.reset_start_px), direction / length)
        )
        return progress >= length

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                path = urlsplit(self.path).path
                if path in {"/", "/index.html"}:
                    body = b"""<!doctype html><meta charset=utf-8><title>Formal trial live monitor</title>
<style>body{margin:0;background:#111;color:#eee;font:15px sans-serif;text-align:center}img{display:block;width:100vw;height:auto}div{padding:8px}</style>
<img id=v><div id=s>Formal trial monitor: reconnecting...</div><script>
const v=document.getElementById('v'),s=document.getElementById('s');let old=null;
async function go(){let wait=100,n=null;try{const r=await fetch('/snapshot.jpg?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);n=URL.createObjectURL(await r.blob());await new Promise((a,b)=>{v.onload=a;v.onerror=b;v.src=n});if(old)URL.revokeObjectURL(old);old=n;n=null;const q=await(await fetch('/cube-status?t='+Date.now(),{cache:'no-store'})).json();s.textContent='Cube: '+q.status+(q.current_px?' current '+q.current_px.join(', ')+' observed start '+q.observed_start_px.join(', ')+' task goal '+q.task_goal_px.join(', '):'')}catch(e){wait=400;s.textContent='Formal trial monitor: reconnecting...'}finally{if(n)URL.revokeObjectURL(n);setTimeout(go,wait)}}go();
</script>"""
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body)))
                    self.end_headers(); self.wfile.write(body); return
                if path == "/snapshot.jpg":
                    with owner.lock:
                        frame = None if owner.preview is None else owner.preview.copy()
                        number = owner.frame_number
                    if frame is None:
                        self.send_error(503); return
                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        self.send_error(500); return
                    data = encoded.tobytes(); self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Frame-Number", str(number)); self.send_header("Content-Length", str(len(data)))
                    self.end_headers(); self.wfile.write(data); return
                if path == "/cube-status":
                    with owner.lock: data = json.dumps(owner.status).encode()
                    self.send_response(200); self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(data)))
                    self.end_headers(); self.wfile.write(data); return
                if path == "/favicon.ico":
                    self.send_response(204); self.end_headers(); return
                self.send_error(404)

            def log_message(self, *_):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start(); self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.server is not None:
            self.server.shutdown(); self.server.server_close()
        self.worker.join(timeout=2)

    @staticmethod
    def _fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        scale = min(width / frame.shape[1], height / frame.shape[0])
        resized = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        x = (width - resized.shape[1]) // 2; y = (height - resized.shape[0]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        return canvas

    def _work(self) -> None:
        last_processed_overhead_sequence = 0
        while not self.stop_event.wait(0.10):
            with self.lock:
                overhead = self.frames.get("overhead")
                wrist = self.frames.get("wrist")
                overhead_sequence = self.published_frame_sequences.get("overhead", 0)
                if overhead is not None: overhead = overhead.copy()
                if wrist is not None: wrist = wrist.copy()
            if (overhead is None or wrist is None
                    or overhead_sequence == last_processed_overhead_sequence):
                continue
            last_processed_overhead_sequence = overhead_sequence
            detection = detect_yellow_cube(
                overhead,
                TrackerConfig(
                    hsv_lower=(10, 80, 90), hsv_upper=(35, 255, 255),
                    min_area_px=300, min_component_confidence=0.35,
                    roi_xywh=(1000, 450, 350, 250),
                ),
                self.last_px,
            )
            view = overhead.copy(); sx0, sy0 = self.reset_start_px; tx, ty = self.task_goal_px; tw, th = self.target_box_wh
            if detection.detected and detection.centroid is not None:
                dx = float(detection.centroid[0] - tx)
                dy = float(detection.centroid[1] - ty)
                goal_pass = bool(
                    max(abs(dx), abs(dy)) <= self.goal_tolerance_px
                    if self.gate_mode == "axiswise"
                    else np.hypot(dx, dy) <= self.goal_tolerance_px
                )
            else:
                goal_pass = False
            goal_pass_streak = self.goal_pass_streak + 1 if goal_pass else 0
            maximum_goal_pass_streak = max(
                self.maximum_goal_pass_streak, goal_pass_streak
            )
            stable_goal_pass = goal_pass_streak >= self.stable_goal_frames
            stable_goal_ever = self.stable_goal_ever or stable_goal_pass
            first_stable_goal_monotonic_ns = self.first_stable_goal_monotonic_ns
            if stable_goal_pass and first_stable_goal_monotonic_ns is None:
                first_stable_goal_monotonic_ns = time.monotonic_ns()
            goal_color = (0,255,0) if goal_pass else (255,0,255)
            cv2.rectangle(view, (round(sx0-tw/2), round(sy0-th/2)), (round(sx0+tw/2), round(sy0+th/2)), (0,255,0), 4)
            cv2.putText(view, "RESET START", (round(sx0-85), round(sy0-25)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
            cv2.rectangle(view, (round(tx-tw/2), round(ty-th/2)), (round(tx+tw/2), round(ty+th/2)), goal_color, 4)
            gate = round(self.goal_tolerance_px)
            if self.gate_mode == "axiswise":
                cv2.rectangle(view, (round(tx)-gate, round(ty)-gate),
                              (round(tx)+gate, round(ty)+gate), goal_color, 3)
            else:
                cv2.circle(view, (round(tx), round(ty)), gate, goal_color, 3, cv2.LINE_AA)
            cv2.line(view, (round(sx0+tw/2), round(sy0)), (round(tx-tw/2), round(ty)), (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(view, "TASK SUCCESS" if goal_pass else "TASK GOAL", (round(tx+tw/2+8), round(ty)), cv2.FONT_HERSHEY_SIMPLEX, .8, goal_color, 2)
            if detection.detected and detection.centroid is not None:
                cx, cy = detection.centroid; self.last_px = (cx, cy)
                if self.start_px is None: self.start_px = (cx, cy)
                self.trail.append((round(cx), round(cy))); self.trail = self.trail[-200:]
                if None not in (detection.bbox_x_px, detection.bbox_y_px, detection.bbox_width_px, detection.bbox_height_px):
                    x,y,w,h = detection.bbox_x_px,detection.bbox_y_px,detection.bbox_width_px,detection.bbox_height_px
                    cv2.rectangle(view, (x,y), (x+w,y+h), (0,255,0) if goal_pass else (0,165,255), 4)
                state = "TASK_SUCCESS" if stable_goal_pass else ("IN_GATE" if goal_pass else "DETECTED"); current = [round(cx,2), round(cy,2)]
            else:
                state = "OCCLUDED"; current = None
            if self.start_px is not None:
                sx, sy = self.start_px; cv2.rectangle(view, (round(sx-18),round(sy-18)), (round(sx+18),round(sy+18)), (255,100,0), 3)
                cv2.putText(view, "START", (round(sx-60),round(sy-25)), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,100,0), 2)
            if len(self.trail) > 1:
                cv2.polylines(view, [np.asarray(self.trail, dtype=np.int32)], False, (255,0,255), 3)
            if state == "OCCLUDED" and self.last_px is not None:
                cv2.circle(view, tuple(map(round,self.last_px)), 22, (0,0,255), 4)
            cv2.rectangle(view, (10,10), (900,64), (0,0,0), -1)
            cv2.putText(view, f"LIVE {state} (raw videos remain unmodified)", (22,48), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if current else (0,0,255), 2)
            combined = np.vstack((self._fit(view,960,600), self._fit(wrist,960,540)))
            start_error = None if current is None else [current[0]-self.reset_start_px[0], current[1]-self.reset_start_px[1]]
            goal_error = None if current is None else [current[0]-self.task_goal_px[0], current[1]-self.task_goal_px[1]]
            radial_error = None if goal_error is None else float(np.hypot(*goal_error))
            sample = {
                "monotonic_ns": time.monotonic_ns(),
                "overhead_frame_sequence": overhead_sequence,
                "current_px": current,
                "goal_error_px": goal_error,
                "radial_error_px": radial_error,
                "gate_mode": self.gate_mode,
                "detected": current is not None,
                "inside_gate": goal_pass if current is not None else False,
                "consecutive_pass_frames": goal_pass_streak,
                "stable_gate_pass": stable_goal_pass,
            }
            status = {"status":state, "current_px":current, "observed_start_px":None if self.start_px is None else list(self.start_px), "reset_start_px":list(self.reset_start_px), "start_error_px":start_error, "task_goal_px":list(self.task_goal_px), "task_goal_error_px":goal_error, "task_goal_error_radial_px":radial_error, "task_goal_gate_mode":self.gate_mode, "task_goal_tolerance_px":self.goal_tolerance_px, "task_success":stable_goal_pass if current is not None else None, "inside_gate_now":goal_pass if current is not None else False, "consecutive_gate_frames":goal_pass_streak, "required_consecutive_gate_frames":self.stable_goal_frames, "stable_goal_ever":stable_goal_ever, "last_reliable_px":None if self.last_px is None else list(self.last_px)}
            with self.lock:
                self.goal_pass_streak = goal_pass_streak
                self.maximum_goal_pass_streak = maximum_goal_pass_streak
                self.stable_goal_ever = stable_goal_ever
                self.first_stable_goal_monotonic_ns = first_stable_goal_monotonic_ns
                self.gate_samples.append(sample)
                self.preview = combined; self.status = status; self.frame_number += 1
