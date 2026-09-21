"""End-to-end real-system smoke: dual video, telemetry, and a tiny J1 round trip."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2

from audit_daheng_camera import configure_sdk
from recover_j5 import ServoBus


ROOT = Path("results/real_robot/system_smoke_20260901_v2")
FPS = 20


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    settings = json.loads(Path("results/real_robot/camera_settings_selected.json").read_text(encoding="utf-8-sig"))
    configure_sdk(Path(r"D:\GalaxySDK"))
    import gxipy as gx  # type: ignore

    manager = gx.DeviceManager(); manager.update_device_list(1500)
    daheng = manager.open_device_by_sn("FDE23080341")
    features = daheng.get_remote_device_feature_control()
    features.get_enum_feature("TriggerMode").set("Off")
    features.get_float_feature("ExposureTime").set(settings["daheng_exposure"])
    features.get_float_feature("Gain").set(settings["daheng_gain"])
    second = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    second.set(cv2.CAP_PROP_EXPOSURE, settings["second_exposure"])
    second.set(cv2.CAP_PROP_BRIGHTNESS, settings["second_brightness"])
    bus = ServoBus("COM3")
    start_positions = [bus.read_u16(i, 56) for i in range(1, 6)]
    j1_start = start_positions[0]
    j1_target = j1_start + round(2.0 * 4096 / 360)
    first_writer = cv2.VideoWriter(str(ROOT / "daheng.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1920, 1200))
    second_writer = cv2.VideoWriter(str(ROOT / "second.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480))
    log_file = (ROOT / "telemetry.csv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["elapsed_s", "phase", *[f"j{i}_raw" for i in range(1, 6)], "voltage_v", "temperature_c"])
    daheng.stream_on(); started = time.monotonic(); commanded_out = commanded_back = False
    try:
        bus.write_u16(1, 42, j1_start); bus.write_u8(1, 41, 1); bus.write_u16(1, 46, 100); bus.write_u8(1, 40, 1)
        while (elapsed := time.monotonic() - started) < 10.0:
            cycle_started = time.monotonic()
            if elapsed >= 2.0 and not commanded_out:
                bus.write_u16(1, 42, j1_target); commanded_out = True
            if elapsed >= 5.0 and not commanded_back:
                bus.write_u16(1, 42, j1_start); commanded_back = True
            raw = daheng.data_stream[0].get_image(1000); ok, other = second.read()
            if raw is None or not ok:
                raise RuntimeError("camera frame loss")
            first = cv2.cvtColor(raw.convert("RGB").get_numpy_array(), cv2.COLOR_RGB2BGR)
            first_writer.write(first); second_writer.write(other)
            positions = [bus.read_u16(i, 56) for i in range(1, 6)]
            voltage = bus.read_u8(1, 62) / 10; temperature = bus.read_u8(1, 63)
            phase = "settle" if elapsed < 2 else ("out" if elapsed < 5 else "return")
            writer.writerow([f"{elapsed:.4f}", phase, *positions, voltage, temperature]); log_file.flush()
            if voltage < 6.0 or temperature >= 50:
                raise RuntimeError("electrical safety threshold exceeded")
            remaining = 1 / FPS - (time.monotonic() - cycle_started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        try:
            bus.write_u16(1, 42, j1_start); time.sleep(.4); bus.write_u8(1, 40, 0)
        finally:
            bus.close(); daheng.stream_off(); daheng.close_device(); second.release()
            first_writer.release(); second_writer.release(); log_file.close()
    print(ROOT.resolve())


if __name__ == "__main__":
    main()
