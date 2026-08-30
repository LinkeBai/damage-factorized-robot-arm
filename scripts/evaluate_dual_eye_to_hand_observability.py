"""Audit dual eye-to-hand object visibility under frozen extrinsic perturbations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "genkiarm": ("sim/assets/genkiarm_push.xml", "block_geom", None),
    "panda": ("sim/assets/panda_push_grasp.xml", "cube_geom", "task_home"),
}
CAMERAS = ("eye_to_hand_left", "eye_to_hand_right")


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def evaluate(height: int = 240, width: int = 320) -> dict:
    perturbations = [
        ("nominal", np.zeros(3), 0.0),
        ("x_plus_20mm", np.array([0.02, 0, 0]), 0.0),
        ("x_minus_20mm", np.array([-0.02, 0, 0]), 0.0),
        ("y_plus_20mm", np.array([0, 0.02, 0]), 0.0),
        ("y_minus_20mm", np.array([0, -0.02, 0]), 0.0),
        ("z_plus_20mm", np.array([0, 0, 0.02]), 0.0),
        ("z_minus_20mm", np.array([0, 0, -0.02]), 0.0),
        ("yaw_plus_3deg", np.zeros(3), np.deg2rad(3)),
        ("yaw_minus_3deg", np.zeros(3), np.deg2rad(-3)),
    ]
    rows = []
    for robot, (relative, object_geom, keyframe) in MODELS.items():
        model = mujoco.MjModel.from_xml_path(str(ROOT / relative)); data = mujoco.MjData(model)
        if keyframe is not None:
            mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height, width); renderer.enable_segmentation_rendering()
        geom_id = model.geom(object_geom).id
        original_pos = {name: model.cam_pos[model.camera(name).id].copy() for name in CAMERAS}
        original_quat = {name: model.cam_quat[model.camera(name).id].copy() for name in CAMERAS}
        nominal_centroid = {}
        for perturbation, translation, yaw in perturbations:
            per_camera = []
            for camera in CAMERAS:
                camera_id = model.camera(camera).id
                model.cam_pos[camera_id] = original_pos[camera] + translation
                yaw_quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
                model.cam_quat[camera_id] = _quat_mul(yaw_quat, original_quat[camera])
                mujoco.mj_forward(model, data); renderer.update_scene(data, camera=camera)
                segmentation = renderer.render()
                pixels = np.argwhere(segmentation[:, :, 0] == geom_id)
                centroid = pixels.mean(axis=0)[::-1] if len(pixels) else np.array([np.nan, np.nan])
                if perturbation == "nominal": nominal_centroid[camera] = centroid
                shift = float(np.linalg.norm(centroid - nominal_centroid.get(camera, centroid)))
                per_camera.append({"camera": camera, "object_pixels": int(len(pixels)),
                                   "centroid_xy_px": centroid.tolist(), "centroid_shift_px": shift,
                                   "visible": bool(len(pixels) >= 20)})
                model.cam_pos[camera_id] = original_pos[camera]
                model.cam_quat[camera_id] = original_quat[camera]
            rows.append({"robot": robot, "perturbation": perturbation,
                         "both_visible": all(x["visible"] for x in per_camera),
                         "cameras": per_camera})
        renderer.close()
    return {"version": "dual_eye_to_hand_observability_v1", "height": height, "width": width,
            "perturbations": 9, "rows": rows,
            "both_visible_fraction": float(np.mean([row["both_visible"] for row in rows])),
            "scope": "observability_only_not_visual_world_model_accuracy"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/dual_eye_to_hand_observability_v1/summary.json"))
    args = parser.parse_args(); result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__": main()
