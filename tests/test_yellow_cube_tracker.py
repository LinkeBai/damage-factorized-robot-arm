import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from robotarm.analysis.yellow_cube_tracker import (
    FrozenCalibration,
    TrackerConfig,
    detect_yellow_cube,
    detection_to_row,
    load_frozen_calibration,
    summarize_rows,
)


def yellow_square(center=(80, 60), size=20):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    half = size // 2
    cv2.rectangle(frame, (center[0] - half, center[1] - half),
                  (center[0] + half, center[1] + half), (0, 255, 255), -1)
    return frame


def test_connected_component_detector_finds_compact_yellow_cube():
    frame = yellow_square()
    # An elongated yellow distractor is rejected by the frozen aspect gate.
    cv2.rectangle(frame, (5, 5), (55, 8), (0, 255, 255), -1)
    detection = detect_yellow_cube(
        frame, TrackerConfig(min_area_px=50, max_area_fraction=0.2))
    assert detection.detected
    assert detection.centroid_x_px == pytest.approx(80.0, abs=0.5)
    assert detection.centroid_y_px == pytest.approx(60.0, abs=0.5)
    assert detection.confidence > 0.8


def test_robust_endpoint_median_ignores_one_outlier_and_projects_direction():
    config = TrackerConfig(
        endpoint_window_frames=5, minimum_endpoint_detections=4,
        minimum_detection_rate=0.8, maximum_endpoint_mad_px=1.0,
        stationary_threshold_px=2.0)
    points = [(10, 20), (10, 20), (99, 99), (10, 20), (10, 20),
              (13, 24), (13, 24), (-50, 80), (13, 24), (13, 24)]
    rows = []
    for frame_index, point in enumerate(points):
        detection = detect_yellow_cube(
            yellow_square(point, size=12),
            TrackerConfig(min_area_px=20, max_area_fraction=0.2))
        rows.append(detection_to_row(frame_index, frame_index / 10, detection))
    summary = summarize_rows(rows, config, (3, 4))
    assert summary["confidence_gate"]["pass"] is True
    assert summary["start_px"]["median"] == [10.0, 20.0]
    assert summary["end_px"]["median"] == [13.0, 24.0]
    assert summary["displacement_px"] == pytest.approx(5.0)
    assert summary["projected_displacement_px"] == pytest.approx(5.0)
    assert summary["basically_unmoved_in_pixel_space"] is False
    assert summary["metric"]["displacement_m"] is None
    assert summary["metric_task_success"] is None


def test_missing_frames_fail_confidence_gate_without_fabricating_metric_values():
    config = TrackerConfig(endpoint_window_frames=3,
                           minimum_endpoint_detections=2,
                           minimum_detection_rate=0.8)
    missing = detection_to_row(
        0, 0.0, detect_yellow_cube(np.zeros((60, 80, 3), np.uint8), config))
    rows = [dict(missing, frame_index=i, video_time_s=float(i)) for i in range(6)]
    summary = summarize_rows(rows, config, (1, 0))
    assert summary["confidence_gate"]["pass"] is False
    assert summary["displacement_px"] is None
    assert summary["basically_unmoved_in_pixel_space"] is None
    assert summary["metric"]["available"] is False


def test_frozen_scale_calibration_enables_metric_values():
    calibration = FrozenCalibration(
        kind="scale", source="fixture", sha256="abc", meters_per_pixel=0.002)
    config = TrackerConfig(endpoint_window_frames=2,
                           minimum_endpoint_detections=2)
    rows = []
    for index, point in enumerate([(10, 10), (10, 10), (13, 14), (13, 14)]):
        detection = detect_yellow_cube(
            yellow_square(point, size=12),
            TrackerConfig(min_area_px=20, max_area_fraction=0.2))
        rows.append(detection_to_row(index, index / 10, detection, calibration))
    summary = summarize_rows(rows, config, (3, 4), calibration)
    assert summary["metric"]["available"] is True
    assert summary["metric"]["displacement_m"] == pytest.approx(0.01)
    assert summary["metric"]["projected_displacement_m"] == pytest.approx(0.01)


def test_calibration_loader_rejects_unfrozen_file(tmp_path: Path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "frozen": False, "type": "scale", "meters_per_pixel": 0.001,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen=true"):
        load_frozen_calibration(path)


def test_frozen_homography_maps_image_points_in_meters(tmp_path: Path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "frozen": True,
        "type": "homography",
        "output_unit": "m",
        "image_to_plane_homography": [
            [0.001, 0.0, 0.1], [0.0, 0.002, -0.2], [0.0, 0.0, 1.0],
        ],
    }), encoding="utf-8")
    calibration = load_frozen_calibration(path)
    assert calibration.map_point((100, 200)) == pytest.approx((0.2, 0.2))
