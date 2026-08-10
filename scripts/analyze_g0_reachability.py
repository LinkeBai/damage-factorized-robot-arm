"""Generate audited position and orientation-constrained G0 reachability."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from robotarm.envs.fk import forward_pose
from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.envs.reachability import joint_ranges_for_damage
from robotarm.training.sim_protocol import damage_from_name

SAMPLES = 100_000
VOXEL = 0.025
MAX_TILT_DEG = 30.0
MORPHOLOGIES = ("intact", "D2", "D3", "D4")


def sample(name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ranges = joint_ranges_for_damage(
        MujocoArmEnv().joint_ranges, damage_from_name(name)
    )
    rng = np.random.default_rng(seed)
    q = rng.uniform(ranges[:, 0], ranges[:, 1], (SAMPLES, 5))
    poses = np.stack([forward_pose(value) for value in q])
    points = poses[:, :3, 3]
    tool_z = poses[:, :3, 2]
    tilt = np.rad2deg(np.arccos(np.clip(tool_z[:, 2], -1.0, 1.0)))
    return points, tilt


def voxel_set(points: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(value) for value in np.floor(points / VOXEL).astype(int)}


def centers(keys: set[tuple[int, int, int]]) -> np.ndarray:
    return (np.asarray(sorted(keys), dtype=float) + 0.5) * VOXEL


def plot(position: np.ndarray, oriented: np.ndarray, path: Path) -> None:
    width, height = 1100, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    panels = ((position, 30, "Position-only common region"), (oriented, 570, "Tilt <= 30 deg common region"))
    for points, left, title in panels:
        draw.rectangle((left, 45, left + 500, 475), outline="#333333", width=2)
        draw.text((left + 10, 15), title, fill="#111111")
        if not len(points):
            continue
        x, z = points[:, 0], points[:, 2]
        xmin, xmax = -0.55, 0.55
        zmin, zmax = 0.0, 0.58
        px = left + 15 + (x - xmin) / (xmax - xmin) * 470
        py = 460 - (z - zmin) / (zmax - zmin) * 395
        color = "#2369a8" if left == 30 else "#d05a35"
        for a, b in zip(px, py):
            draw.ellipse((a - 2, b - 2, a + 2, b + 2), fill=color)
        draw.text((left + 200, 482), "X (m)", fill="#222222")
        draw.text((left + 12, 55), "Z (m)", fill="#222222")
    image.save(path)


def main() -> None:
    clouds = {}
    tilts = {}
    for index, name in enumerate(MORPHOLOGIES):
        clouds[name], tilts[name] = sample(name, 100 + index)
    position_sets = [voxel_set(clouds[name]) for name in MORPHOLOGIES]
    oriented_sets = [
        voxel_set(clouds[name][tilts[name] <= MAX_TILT_DEG])
        for name in MORPHOLOGIES
    ]
    position_common = set.intersection(*position_sets)
    oriented_common = set.intersection(*oriented_sets)
    position_centers = centers(position_common)
    oriented_centers = centers(oriented_common)
    result = {
        "status": "complete",
        "samples_per_morphology": SAMPLES,
        "voxel_size_m": VOXEL,
        "morphologies": list(MORPHOLOGIES),
        "position_only_common_voxels": len(position_common),
        "orientation_constraint": {"tool_z_max_tilt_deg": MAX_TILT_DEG},
        "orientation_constrained_common_voxels": len(oriented_common),
        "orientation_fraction_of_position": (
            len(oriented_common) / len(position_common) if position_common else 0.0
        ),
    }
    out = Path("results/final/g0-reachability-full.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot(position_centers, oriented_centers, Path("reports/g0-reachability.png"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
