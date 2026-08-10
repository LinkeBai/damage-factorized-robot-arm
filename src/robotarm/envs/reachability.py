"""Reachability analysis over morphologies (PROJECT-PLAN-V4 G0 §4).

Computes the set of end-effector target positions that are reachable by a
given morphology (intact or a specific damage config), then the *common*
reachable set across a family of morphologies. Task targets must be sampled
from this common region so that the same target is physically achievable by
every deployment being compared (plan §5.1: "target 只从健康与损坏 morphology
的共同可达域采样").

Works on a kinematic oracle (FK) — no physics needed — which is the right tool
for reachability (position-only). It is environment-agnostic: it takes an FK
callable and joint ranges, so it works for sim or real once FK is measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from .damage import DamageConfig

FkFn = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

_GRID_SIZE = 20_000


@dataclass
class ReachabilityResult:
    """Sampled reachable points for one morphology + its bounding box."""

    kind: str  # e.g. 'intact', 'D2@0.5'
    config_label: str
    points: npt.NDArray  # (N, 3)
    bounds: tuple[npt.NDArray, npt.NDArray]  # (min, max) per axis

    def contains(self, p: npt.NDArray) -> bool:
        p = np.asarray(p, dtype=np.float64)
        return bool(np.all(p >= self.bounds[0]) and np.all(p <= self.bounds[1]))


def _sample_joint_configs(joint_ranges: npt.NDArray, rng: np.random.Generator, n: int) -> npt.NDArray:
    lo = joint_ranges[:, 0]
    hi = joint_ranges[:, 1]
    return rng.uniform(lo, hi, size=(n, joint_ranges.shape[0]))


def reachable_points(
    fk: FkFn,
    joint_ranges: npt.NDArray,
    n: int = _GRID_SIZE,
    rng: np.random.Generator | None = None,
) -> npt.NDArray:
    """Uniformly sample joint configs, return their FK tip positions (N,3)."""
    rng = rng or np.random.default_rng(0)
    q = _sample_joint_configs(np.asarray(joint_ranges, dtype=np.float64), rng, n)
    return np.stack([np.asarray(fk(qi), dtype=np.float64) for qi in q], axis=0)


def joint_ranges_for_damage(
    joint_ranges: npt.ArrayLike,
    damage: DamageConfig | None,
) -> npt.NDArray[np.float64]:
    """Return joint ranges with every locked joint collapsed to its angle."""
    ranges = np.asarray(joint_ranges, dtype=np.float64).copy()
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise ValueError(f"joint_ranges must have shape (dof, 2), got {ranges.shape}")
    if damage is None:
        return ranges
    if damage.dof != ranges.shape[0]:
        raise ValueError(
            f"damage dof={damage.dof} does not match {ranges.shape[0]} joint ranges"
        )
    for joint in damage.locked:
        angle = damage.lock_angle_of(joint)
        lo, hi = ranges[joint]
        if not lo <= angle <= hi:
            raise ValueError(
                f"lock angle {angle:.4f} for joint {joint} is outside [{lo:.4f}, {hi:.4f}]"
            )
        ranges[joint] = angle
    return ranges


def analyze_damage_morphology(
    kind: str,
    fk: FkFn,
    joint_ranges: npt.ArrayLike,
    damage: DamageConfig | None,
    n: int = _GRID_SIZE,
    rng: np.random.Generator | None = None,
) -> ReachabilityResult:
    """Analyze one intact/damaged morphology with exact locked-joint ranges."""
    return analyze_morphology(
        kind,
        fk,
        joint_ranges_for_damage(joint_ranges, damage),
        n=n,
        rng=rng,
    )


def analyze_morphology(
    kind: str,
    fk: FkFn,
    joint_ranges: npt.NDArray,
    n: int = _GRID_SIZE,
    rng: np.random.Generator | None = None,
) -> ReachabilityResult:
    pts = reachable_points(fk, joint_ranges, n, rng)
    return ReachabilityResult(
        kind=kind,
        config_label=kind,
        points=pts,
        bounds=(pts.min(axis=0), pts.max(axis=0)),
    )


def common_reachable_region(
    results: Sequence[ReachabilityResult], voxel_size: float = 0.02
) -> tuple[npt.NDArray, npt.NDArray]:
    """Return (reachable_mask_grid, grid_cell_centers) for the *common* region.

    Discrete the combined bounding box into voxels; a voxel is in the common
    region only if it is hit by *every* morphology's sample cloud. Returns the
    boolean occupancy grid and its cell centers.
    """
    results = list(results)
    if not results:
        raise ValueError("at least one morphology required")

    allmin = np.stack([r.bounds[0] for r in results]).min(axis=0)
    allmax = np.stack([r.bounds[1] for r in results]).max(axis=0)
    # Pad the common box slightly to include most points.
    lo = np.floor(allmin / voxel_size).astype(int)
    hi = np.ceil(allmax / voxel_size).astype(int) + 1
    shape = tuple((hi - lo).astype(int))
    grid = np.ones(shape, dtype=bool)  # start fully "common", AND down

    for r in results:
        idx = np.round((r.points - allmin) / voxel_size).astype(int)
        idx = np.clip(idx, 0, np.array(shape) - 1)
        occ = np.zeros(shape, dtype=bool)
        xi, yi, zi = idx[:, 0], idx[:, 1], idx[:, 2]
        occ[xi, yi, zi] = True
        grid &= occ

    centers_ix = np.argwhere(grid)
    centers = allmin + (centers_ix + 0.5) * voxel_size
    return grid, centers


def sample_targets_from_common(
    results: Sequence[ReachabilityResult],
    n: int,
    *,
    voxel_size: float = 0.02,
    rng: np.random.Generator | None = None,
    margin: float = 0.0,
) -> npt.NDArray:
    """Sample ``n`` target positions from the common reachable region.

    ``margin`` (m) shrinks the region away from voxel boundaries to avoid
    near-boundary targets. Falls back to denser sampling if empty.
    """
    rng = rng or np.random.default_rng(1)
    grid, centers = common_reachable_region(results, voxel_size)
    if not np.any(grid):
        raise RuntimeError("no common reachable region found — increase samples or relax margin")

    # Optional shrink toward inner voxels so targets aren't on the boundary.
    mask = grid
    if margin > 0:
        from scipy import ndimage

        inner = ndimage.binary_erosion(grid, iterations=max(1, int(margin / voxel_size)))
        if np.any(inner):  # keep original only if erosion empties it
            mask = inner

    idx = np.argwhere(mask)
    if len(idx) == 0:
        raise RuntimeError("margin too large; empty inner region")
    chosen = rng.choice(len(idx), size=n, replace=True)

    allmin = np.stack([r.bounds[0] for r in results]).min(axis=0)
    return allmin + (idx[chosen] + 0.5) * voxel_size
