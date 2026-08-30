"""Render the two fixed eye-to-hand views for the actual simulation assets."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "genkiarm_push": ROOT / "sim" / "assets" / "genkiarm_push.xml",
    "panda_push_grasp": ROOT / "sim" / "assets" / "panda_push_grasp.xml",
}


def save_png(image: np.ndarray, path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{image.shape[1]}x{image.shape[0]}", "-i", "-", "-frames:v", "1", str(path)],
        input=image.tobytes(), check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "actual-model-eye-to-hand")
    parser.add_argument("--width", type=int, default=640); parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for asset_name, path in ASSETS.items():
        model = mujoco.MjModel.from_xml_path(str(path)); data = mujoco.MjData(model)
        if asset_name == "panda_push_grasp":
            mujoco.mj_resetDataKeyframe(model, data, model.key("task_home").id)
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        for camera_name in ("eye_to_hand_left", "eye_to_hand_right"):
            renderer.update_scene(data, camera=camera_name)
            image = renderer.render().copy()
            output = args.output_dir / f"{asset_name}_{camera_name}.png"; save_png(image, output); panels.append(image)
        renderer.close()
    montage = np.concatenate(
        [np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)], axis=0
    )
    save_png(montage, args.output_dir / "dual_eye_to_hand_actual_models.png")
    print(args.output_dir / "dual_eye_to_hand_actual_models.png")


if __name__ == "__main__": main()
