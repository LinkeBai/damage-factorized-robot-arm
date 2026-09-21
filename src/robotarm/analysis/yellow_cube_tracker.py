"""Auditable offline tracking of a yellow cube in an unmodified video.

The core functions in this module are deterministic and have no camera,
serial-port, or robot dependencies.  Image coordinates follow OpenCV's
convention: ``+x`` is right and ``+y`` is down.  Metric values are produced
only when an explicitly frozen calibration is supplied.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    hsv_lower: tuple[int, int, int] = (18, 90, 80)
    hsv_upper: tuple[int, int, int] = (42, 255, 255)
    min_area_px: int = 100
    max_area_fraction: float = 0.02
    min_aspect_ratio: float = 0.40
    max_aspect_ratio: float = 2.50
    morphology_kernel_px: int = 3
    min_component_confidence: float = 0.50
    endpoint_window_frames: int = 15
    minimum_endpoint_detections: int = 5
    minimum_detection_rate: float = 0.90
    maximum_endpoint_mad_px: float = 5.0
    stationary_threshold_px: float = 2.0
    roi_xywh: tuple[int, int, int, int] | None = None

    def validate(self) -> None:
        if any(len(values) != 3 for values in (self.hsv_lower, self.hsv_upper)):
            raise ValueError("HSV bounds must each contain three integers")
        if any(lo < 0 or hi > limit or lo > hi for lo, hi, limit in zip(
                self.hsv_lower, self.hsv_upper, (179, 255, 255))):
            raise ValueError("invalid OpenCV HSV bounds")
        if self.min_area_px <= 0:
            raise ValueError("min_area_px must be positive")
        if not 0 < self.max_area_fraction <= 1:
            raise ValueError("max_area_fraction must be in (0, 1]")
        if not 0 < self.min_aspect_ratio <= self.max_aspect_ratio:
            raise ValueError("invalid aspect-ratio limits")
        if self.morphology_kernel_px <= 0 or self.morphology_kernel_px % 2 == 0:
            raise ValueError("morphology_kernel_px must be a positive odd integer")
        if not 0 <= self.min_component_confidence <= 1:
            raise ValueError("min_component_confidence must be in [0, 1]")
        if self.endpoint_window_frames <= 0 or self.minimum_endpoint_detections <= 0:
            raise ValueError("endpoint window/count must be positive")
        if not 0 <= self.minimum_detection_rate <= 1:
            raise ValueError("minimum_detection_rate must be in [0, 1]")
        if self.maximum_endpoint_mad_px < 0 or self.stationary_threshold_px < 0:
            raise ValueError("MAD and stationary thresholds must be non-negative")
        if self.roi_xywh is not None:
            x, y, width, height = self.roi_xywh
            if min(x, y) < 0 or min(width, height) <= 0:
                raise ValueError("ROI must be non-negative x/y and positive width/height")


@dataclass(frozen=True)
class CubeDetection:
    detected: bool
    confidence: float
    centroid_x_px: float | None = None
    centroid_y_px: float | None = None
    bbox_x_px: int | None = None
    bbox_y_px: int | None = None
    bbox_width_px: int | None = None
    bbox_height_px: int | None = None
    component_area_px: int | None = None
    bbox_extent: float | None = None
    mean_saturation: float | None = None
    mean_value: float | None = None

    @property
    def centroid(self) -> tuple[float, float] | None:
        if self.centroid_x_px is None or self.centroid_y_px is None:
            return None
        return self.centroid_x_px, self.centroid_y_px


@dataclass(frozen=True)
class FrozenCalibration:
    kind: str
    source: str
    sha256: str
    meters_per_pixel: float | None = None
    homography_image_to_plane_m: tuple[tuple[float, float, float], ...] | None = None

    def map_point(self, point_px: Sequence[float]) -> tuple[float, float]:
        x, y = float(point_px[0]), float(point_px[1])
        if self.kind == "scale":
            assert self.meters_per_pixel is not None
            return x * self.meters_per_pixel, y * self.meters_per_pixel
        assert self.homography_image_to_plane_m is not None
        matrix = np.asarray(self.homography_image_to_plane_m, dtype=float)
        mapped = matrix @ np.asarray([x, y, 1.0], dtype=float)
        if abs(float(mapped[2])) < 1e-12:
            raise ValueError("homography maps point to infinity")
        return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_calibration(path: Path) -> FrozenCalibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("frozen") is not True:
        raise ValueError("calibration must explicitly contain frozen=true")
    kind = str(payload.get("type", "")).strip().lower()
    source = str(path.resolve())
    digest = sha256_file(path)
    if kind == "scale":
        scale = float(payload["meters_per_pixel"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("meters_per_pixel must be finite and positive")
        return FrozenCalibration(
            kind=kind, source=source, sha256=digest, meters_per_pixel=scale)
    if kind == "homography":
        if payload.get("output_unit") != "m":
            raise ValueError("homography output_unit must be 'm'")
        matrix = np.asarray(payload["image_to_plane_homography"], dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("image_to_plane_homography must be a finite 3x3 matrix")
        if abs(float(np.linalg.det(matrix))) < 1e-15:
            raise ValueError("homography matrix is singular")
        return FrozenCalibration(
            kind=kind,
            source=source,
            sha256=digest,
            homography_image_to_plane_m=tuple(tuple(float(v) for v in row)
                                                for row in matrix),
        )
    raise ValueError("calibration type must be 'scale' or 'homography'")


def normalized_direction(direction_xy: Sequence[float]) -> tuple[float, float]:
    direction = np.asarray(direction_xy, dtype=float)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)):
        raise ValueError("image direction must contain two finite values")
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise ValueError("image direction cannot be zero")
    direction /= norm
    return float(direction[0]), float(direction[1])


def _candidate_confidence(
        aspect_ratio: float, extent: float, mean_saturation: float,
        mean_value: float) -> float:
    aspect_score = math.exp(-abs(math.log(aspect_ratio)))
    extent_score = min(1.0, extent / 0.75)
    saturation_score = min(1.0, mean_saturation / 255.0)
    value_score = min(1.0, mean_value / 255.0)
    return float(0.25 * aspect_score + 0.35 * extent_score
                 + 0.20 * saturation_score + 0.20 * value_score)


def detect_yellow_cube(
        frame_bgr: np.ndarray, config: TrackerConfig,
        previous_centroid: Sequence[float] | None = None) -> CubeDetection:
    """Detect one compact yellow connected component in a BGR frame."""
    config.validate()
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame must be an HxWx3 BGR image")
    frame_height, frame_width = frame_bgr.shape[:2]
    if config.roi_xywh is None:
        roi_x, roi_y, roi_width, roi_height = 0, 0, frame_width, frame_height
    else:
        roi_x, roi_y, roi_width, roi_height = config.roi_xywh
        if roi_x + roi_width > frame_width or roi_y + roi_height > frame_height:
            raise ValueError("ROI lies outside the frame")
    roi = frame_bgr[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.asarray(config.hsv_lower, dtype=np.uint8),
        np.asarray(config.hsv_upper, dtype=np.uint8))
    kernel = np.ones((config.morphology_kernel_px,
                      config.morphology_kernel_px), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    maximum_area = config.max_area_fraction * frame_width * frame_height
    frame_diagonal = math.hypot(frame_width, frame_height)
    candidates: list[tuple[float, float, CubeDetection]] = []
    for component in range(1, component_count):
        x, y, width, height, area = [int(v) for v in stats[component]]
        if area < config.min_area_px or area > maximum_area or height <= 0:
            continue
        aspect_ratio = width / height
        if not config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio:
            continue
        pixels = labels == component
        mean_saturation = float(np.mean(hsv[..., 1][pixels]))
        mean_value = float(np.mean(hsv[..., 2][pixels]))
        extent = area / float(width * height)
        confidence = _candidate_confidence(
            aspect_ratio, extent, mean_saturation, mean_value)
        center_x = float(centroids[component][0] + roi_x)
        center_y = float(centroids[component][1] + roi_y)
        continuity_penalty = 0.0
        if previous_centroid is not None:
            distance = math.hypot(center_x - float(previous_centroid[0]),
                                  center_y - float(previous_centroid[1]))
            continuity_penalty = 0.25 * min(1.0, distance / frame_diagonal)
        detection = CubeDetection(
            detected=confidence >= config.min_component_confidence,
            confidence=confidence,
            centroid_x_px=center_x,
            centroid_y_px=center_y,
            bbox_x_px=x + roi_x,
            bbox_y_px=y + roi_y,
            bbox_width_px=width,
            bbox_height_px=height,
            component_area_px=area,
            bbox_extent=extent,
            mean_saturation=mean_saturation,
            mean_value=mean_value,
        )
        candidates.append((confidence - continuity_penalty, confidence, detection))
    if not candidates:
        return CubeDetection(detected=False, confidence=0.0)
    selected = max(candidates, key=lambda item: (item[0], item[1]))[2]
    if not selected.detected:
        return CubeDetection(detected=False, confidence=selected.confidence)
    return selected


def robust_endpoint(
        rows: Sequence[dict[str, Any]], frame_indices: Iterable[int],
        coordinate_keys: tuple[str, str]) -> dict[str, Any]:
    indices = set(frame_indices)
    selected = [row for row in rows if row["frame_index"] in indices
                and row["detected"] and row[coordinate_keys[0]] is not None
                and row[coordinate_keys[1]] is not None]
    if not selected:
        return {
            "median": None, "radial_mad": None, "detections": 0,
            "representative_frame_index": None,
        }
    points = np.asarray([[row[coordinate_keys[0]], row[coordinate_keys[1]]]
                         for row in selected], dtype=float)
    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median[None, :], axis=1)
    representative = selected[int(np.argmin(distances))]["frame_index"]
    return {
        "median": [float(median[0]), float(median[1])],
        "radial_mad": float(np.median(distances)),
        "detections": len(selected),
        "representative_frame_index": int(representative),
    }


def summarize_rows(
        rows: Sequence[dict[str, Any]], config: TrackerConfig,
        image_direction_xy: Sequence[float],
        calibration: FrozenCalibration | None = None) -> dict[str, Any]:
    """Summarize per-frame detections without inventing physical units."""
    config.validate()
    if not rows:
        raise ValueError("cannot summarize an empty tracking result")
    direction = normalized_direction(image_direction_xy)
    frame_count = len(rows)
    window = min(config.endpoint_window_frames, frame_count)
    start_indices = range(0, window)
    end_indices = range(frame_count - window, frame_count)
    start_px = robust_endpoint(rows, start_indices,
                               ("centroid_x_px", "centroid_y_px"))
    end_px = robust_endpoint(rows, end_indices,
                             ("centroid_x_px", "centroid_y_px"))
    detected_frames = sum(bool(row["detected"]) for row in rows)
    detection_rate = detected_frames / frame_count
    endpoint_count_pass = (
        start_px["detections"] >= config.minimum_endpoint_detections
        and end_px["detections"] >= config.minimum_endpoint_detections)
    endpoint_stability_pass = (
        start_px["radial_mad"] is not None
        and end_px["radial_mad"] is not None
        and start_px["radial_mad"] <= config.maximum_endpoint_mad_px
        and end_px["radial_mad"] <= config.maximum_endpoint_mad_px)
    gate = {
        "pass": bool(detection_rate >= config.minimum_detection_rate
                     and endpoint_count_pass and endpoint_stability_pass),
        "detection_rate_pass": detection_rate >= config.minimum_detection_rate,
        "endpoint_detection_count_pass": endpoint_count_pass,
        "endpoint_stability_pass": endpoint_stability_pass,
        "minimum_detection_rate": config.minimum_detection_rate,
        "minimum_endpoint_detections": config.minimum_endpoint_detections,
        "maximum_endpoint_radial_mad_px": config.maximum_endpoint_mad_px,
    }
    displacement_vector_px = None
    displacement_px = None
    projected_px = None
    orthogonal_px = None
    stationary = None
    if start_px["median"] is not None and end_px["median"] is not None:
        delta = np.asarray(end_px["median"]) - np.asarray(start_px["median"])
        displacement_vector_px = [float(delta[0]), float(delta[1])]
        displacement_px = float(np.linalg.norm(delta))
        projected_px = float(np.dot(delta, np.asarray(direction)))
        normal = np.asarray([-direction[1], direction[0]])
        orthogonal_px = float(np.dot(delta, normal))
        stationary = bool(gate["pass"]
                          and displacement_px <= config.stationary_threshold_px)

    metric = {
        "available": calibration is not None,
        "calibration": asdict(calibration) if calibration is not None else None,
        "start_median_xy_m": None,
        "end_median_xy_m": None,
        "displacement_vector_m": None,
        "displacement_m": None,
        "projected_displacement_m": None,
    }
    if calibration is not None and start_px["median"] is not None \
            and end_px["median"] is not None:
        start_m = np.asarray(calibration.map_point(start_px["median"]), dtype=float)
        end_m = np.asarray(calibration.map_point(end_px["median"]), dtype=float)
        delta_m = end_m - start_m
        direction_tip_m = np.asarray(calibration.map_point(
            (start_px["median"][0] + direction[0],
             start_px["median"][1] + direction[1])), dtype=float)
        world_direction = direction_tip_m - start_m
        world_direction_norm = float(np.linalg.norm(world_direction))
        projected_m = None
        if world_direction_norm > 1e-12:
            projected_m = float(np.dot(delta_m,
                                       world_direction / world_direction_norm))
        metric.update({
            "start_median_xy_m": start_m.tolist(),
            "end_median_xy_m": end_m.tolist(),
            "displacement_vector_m": delta_m.tolist(),
            "displacement_m": float(np.linalg.norm(delta_m)),
            "projected_displacement_m": projected_m,
        })
    return {
        "coordinate_convention": "+x right, +y down",
        "frame_count": frame_count,
        "detected_frames": detected_frames,
        "detection_rate": detection_rate,
        "confidence_gate": gate,
        "endpoint_window_frames": window,
        "start_px": start_px,
        "end_px": end_px,
        "image_direction_unit_xy": list(direction),
        "displacement_vector_px": displacement_vector_px,
        "displacement_px": displacement_px,
        "projected_displacement_px": projected_px,
        "orthogonal_displacement_px": orthogonal_px,
        "stationary_threshold_px": config.stationary_threshold_px,
        "basically_unmoved_in_pixel_space": stationary,
        "metric": metric,
        "metric_task_success": None,
        "scope_warning": (
            "Pixel displacement is not a metric distance. Metric fields are null "
            "unless a frozen scale or image-to-plane homography is supplied."
        ),
    }


def detection_to_row(
        frame_index: int, video_time_s: float, detection: CubeDetection,
        calibration: FrozenCalibration | None = None) -> dict[str, Any]:
    row = {
        "frame_index": int(frame_index),
        "video_time_s": float(video_time_s),
        "detected": bool(detection.detected),
        "confidence": float(detection.confidence),
        "centroid_x_px": detection.centroid_x_px if detection.detected else None,
        "centroid_y_px": detection.centroid_y_px if detection.detected else None,
        "bbox_x_px": detection.bbox_x_px if detection.detected else None,
        "bbox_y_px": detection.bbox_y_px if detection.detected else None,
        "bbox_width_px": detection.bbox_width_px if detection.detected else None,
        "bbox_height_px": detection.bbox_height_px if detection.detected else None,
        "component_area_px": detection.component_area_px if detection.detected else None,
        "bbox_extent": detection.bbox_extent if detection.detected else None,
        "mean_saturation": detection.mean_saturation if detection.detected else None,
        "mean_value": detection.mean_value if detection.detected else None,
        "centroid_x_m": None,
        "centroid_y_m": None,
    }
    if calibration is not None and detection.detected:
        assert detection.centroid is not None
        row["centroid_x_m"], row["centroid_y_m"] = calibration.map_point(
            detection.centroid)
    return row


def draw_detection(frame: np.ndarray, detection: CubeDetection) -> np.ndarray:
    annotated = frame.copy()
    if detection.detected:
        assert detection.bbox_x_px is not None and detection.bbox_y_px is not None
        assert detection.bbox_width_px is not None and detection.bbox_height_px is not None
        assert detection.centroid is not None
        x, y = detection.bbox_x_px, detection.bbox_y_px
        cv2.rectangle(annotated, (x, y),
                      (x + detection.bbox_width_px, y + detection.bbox_height_px),
                      (0, 255, 0), 2)
        center = tuple(int(round(v)) for v in detection.centroid)
        cv2.drawMarker(annotated, center, (255, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
        cv2.putText(annotated, f"yellow cube conf={detection.confidence:.3f}",
                    (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(annotated, "yellow cube: NOT DETECTED", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                    cv2.LINE_AA)
    return annotated
