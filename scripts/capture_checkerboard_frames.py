"""Interactively capture diverse checkerboard frames without moving the robot."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--columns", type=int, required=True,
                        help="checkerboard inner-corner columns")
    parser.add_argument("--rows", type=int, required=True,
                        help="checkerboard inner-corner rows")
    parser.add_argument("--output-dir", type=Path, default=Path("calibration/eye_in_hand"))
    parser.add_argument("--target-count", type=int, default=20)
    args = parser.parse_args()
    if args.columns < 3 or args.rows < 3 or args.target_count < 10:
        raise SystemExit("invalid pattern or target count (minimum 10)")
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("install the project real extra: pip install -e .[real]") from exc
    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise SystemExit(f"cannot open camera index {args.camera_index}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    instructions = "SPACE: save detected board | Q/ESC: finish"
    try:
        while len(saved) < args.target_count:
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (args.columns, args.rows))
            preview = frame.copy()
            cv2.drawChessboardCorners(preview, (args.columns, args.rows), corners, found)
            cv2.putText(preview, f"{instructions}  {len(saved)}/{args.target_count}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .6,
                        (0, 220, 0) if found else (0, 0, 255), 2)
            cv2.imshow("Eye-in-hand checkerboard capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32 and found:
                path = args.output_dir/f"checkerboard_{len(saved):03d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"failed to write {path}")
                saved.append(str(path))
                time.sleep(.15)
    finally:
        capture.release(); cv2.destroyAllWindows()
    manifest = {"camera_index": args.camera_index,
                "checkerboard_inner_corners": [args.columns, args.rows],
                "frames": saved, "complete": len(saved) >= 10,
                "note": "Use varied board positions, tilts, distances, and image edges."}
    (args.output_dir/"capture_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
