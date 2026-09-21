import time

import numpy as np

import robotarm.hardware.live_trial_monitor as monitor_module
from robotarm.analysis.yellow_cube_tracker import CubeDetection
from robotarm.hardware.live_trial_monitor import LiveTrialMonitor


def wait_until(predicate, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true before timeout")


def test_live_gate_requires_three_distinct_overhead_frames(monkeypatch):
    monkeypatch.setattr(
        monitor_module,
        "detect_yellow_cube",
        lambda *_args, **_kwargs: CubeDetection(
            detected=True,
            confidence=1.0,
            centroid_x_px=1116.31,
            centroid_y_px=570.0,
            bbox_x_px=1106,
            bbox_y_px=560,
            bbox_width_px=20,
            bbox_height_px=20,
        ),
    )
    monitor = LiveTrialMonitor(stable_goal_frames=3, port=0)
    frame = np.zeros((700, 1400, 3), dtype=np.uint8)
    monitor.worker.start()
    try:
        monitor.publish("wrist", frame)
        monitor.publish("overhead", frame)
        wait_until(lambda: monitor.gate_summary()["processed_overhead_samples"] == 1)

        # Re-reading the cached frame must not advance the consecutive-frame gate.
        time.sleep(0.25)
        assert monitor.gate_summary()["processed_overhead_samples"] == 1
        assert monitor.gate_summary()["stable_goal_ever"] is False

        monitor.publish("overhead", frame)
        wait_until(lambda: monitor.gate_summary()["processed_overhead_samples"] == 2)
        assert monitor.gate_summary()["stable_goal_ever"] is False

        monitor.publish("overhead", frame)
        wait_until(lambda: monitor.gate_summary()["stable_goal_ever"] is True)
        summary = monitor.gate_summary()
        assert summary["maximum_consecutive_pass_frames"] == 3
        assert [sample["overhead_frame_sequence"] for sample in summary["samples"]] == [1, 2, 3]
    finally:
        monitor.stop_event.set()
        monitor.worker.join(timeout=2)


def test_axiswise_gate_accepts_frozen_five_pixel_corner(monkeypatch):
    monkeypatch.setattr(
        monitor_module, "detect_yellow_cube",
        lambda *_args, **_kwargs: CubeDetection(
            detected=True, confidence=1.0,
            centroid_x_px=1121.31, centroid_y_px=575.0,
            bbox_x_px=1111, bbox_y_px=565, bbox_width_px=20, bbox_height_px=20,
        ),
    )
    monitor = LiveTrialMonitor(stable_goal_frames=1, port=0)
    frame = np.zeros((700, 1400, 3), dtype=np.uint8)
    monitor.worker.start()
    try:
        monitor.publish("wrist", frame); monitor.publish("overhead", frame)
        wait_until(lambda: monitor.gate_summary()["processed_overhead_samples"] == 1)
        summary = monitor.gate_summary()
        assert summary["gate_mode"] == "axiswise"
        assert summary["stable_goal_ever"] is True
        assert summary["samples"][0]["radial_error_px"] > 5.0
    finally:
        monitor.stop_event.set(); monitor.worker.join(timeout=2)
