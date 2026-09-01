"""Enumerate a Daheng Galaxy camera and optionally save one auditable frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path


def configure_sdk(root: Path) -> None:
    genicam = root / "GenICam"
    os.environ["GALAXY_GENICAM_ROOT"] = str(genicam)
    os.environ["GENICAM_GENTL64_PATH"] = str(root / "GenTL" / "Win64")
    os.environ["GENICAM_GENTL32_PATH"] = str(root / "GenTL" / "Win32")
    sys.path.insert(0, str(root / "Development" / "Samples" / "Python"))
    # Galaxy SDK 2.6 imports numpy.compat.long, removed by recent NumPy.
    compat = types.ModuleType("numpy.compat")
    compat.long = int
    sys.modules.setdefault("numpy.compat", compat)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-root", type=Path, default=Path(r"D:\GalaxySDK"))
    parser.add_argument("--serial", default="")
    parser.add_argument("--frame", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_sdk(args.sdk_root)

    import gxipy as gx  # type: ignore

    manager = gx.DeviceManager()
    count, devices = manager.update_device_list(1500)
    selected = None
    for device in devices:
        if not args.serial or device.get("sn") == args.serial:
            selected = device
            break
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sdk_root": str(args.sdk_root),
        "device_count": count,
        "devices": devices,
        "requested_serial": args.serial or None,
        "selected_device": selected,
        "frame": None,
        "status": "FAIL",
    }
    camera = None
    try:
        if selected is None:
            raise RuntimeError("requested Daheng camera was not enumerated")
        camera = manager.open_device_by_sn(selected["sn"])
        if args.frame:
            from PIL import Image

            feature = camera.get_remote_device_feature_control()
            feature.get_enum_feature("TriggerMode").set("Off")
            camera.stream_on()
            raw = camera.data_stream[0].get_image(3000)
            if raw is None:
                raise RuntimeError("camera returned no frame")
            rgb = raw.convert("RGB")
            array = rgb.get_numpy_array()
            args.frame.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array, "RGB").save(args.frame)
            result["frame"] = {
                "path": args.frame.as_posix(),
                "width": raw.get_width(),
                "height": raw.get_height(),
                "frame_id": raw.get_frame_id(),
                "sha256": sha256(args.frame),
            }
            camera.stream_off()
        result["status"] = "PASS"
    except Exception as error:
        result["error"] = str(error)
    finally:
        if camera is not None:
            try:
                camera.close_device()
            except Exception:
                pass
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(result["status"])
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
