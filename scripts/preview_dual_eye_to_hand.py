"""Live alignment preview for Daheng and the secondary eye-to-hand camera."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from audit_daheng_camera import configure_sdk


def fit(frame: np.ndarray, width: int = 720, height: int = 480) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, None, fx=scale, fy=scale)
    canvas = np.zeros((height, width, 3), np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def guides(frame: np.ndarray, label: str) -> np.ndarray:
    h, w = frame.shape[:2]
    color = (0, 255, 0)
    cv2.line(frame, (0, h // 2), (w, h // 2), color, 1)
    cv2.line(frame, (w // 2, 0), (w // 2, h), color, 1)
    for fraction in (0.25, 0.75):
        cv2.line(frame, (0, round(h * fraction)), (w, round(h * fraction)), (0, 180, 255), 1)
        cv2.line(frame, (round(w * fraction), 0), (round(w * fraction), h), (0, 180, 255), 1)
    cv2.putText(frame, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secondary-index", type=int, default=1)
    parser.add_argument("--serial", default="FDE23080341")
    args = parser.parse_args()
    configure_sdk(Path(r"D:\GalaxySDK"))
    import gxipy as gx  # type: ignore

    manager = gx.DeviceManager()
    manager.update_device_list(1500)
    daheng = manager.open_device_by_sn(args.serial)
    feature = daheng.get_remote_device_feature_control()
    feature.get_enum_feature("TriggerMode").set("Off")
    secondary = cv2.VideoCapture(args.secondary_index, cv2.CAP_DSHOW)
    if not secondary.isOpened():
        raise RuntimeError("secondary camera could not be opened")
    title = "Dual eye-to-hand alignment - Q to quit, S to save"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1440, 520)
    daheng.stream_on()
    last = None
    try:
        while True:
            raw = daheng.data_stream[0].get_image(1000)
            ok, second = secondary.read()
            if raw is None or not ok:
                continue
            first = raw.convert("RGB").get_numpy_array()
            first = cv2.cvtColor(first, cv2.COLOR_RGB2BGR)
            left = guides(fit(first), "Daheng overhead")
            right = guides(fit(second), "Secondary camera")
            last = np.hstack((left, right))
            cv2.imshow(title, last)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                output = Path("results/real_robot") / f"dual_alignment_{stamp}.png"
                output.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output), last)
                print(f"saved {output.resolve()}", flush=True)
    finally:
        daheng.stream_off()
        daheng.close_device()
        secondary.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
