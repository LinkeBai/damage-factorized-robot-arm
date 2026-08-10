"""Append-only episode storage (PROJECT-PLAN-V4 §10.2).

Immutability rules enforced here:

* Raw trajectories are append-only and never edited in place. Writing an
  ``episode_id`` that already exists (either in the manifest or on disk)
  raises ``ValueError`` — there is no silent overwrite.
* Cleaning / exclusion produces a *new* dataset version in a new directory,
  never an in-place edit. ``exclude()`` only *marks* a manifest entry with a
  reason (the audit ledger); the payload file is left untouched.
* A JSON manifest tracks each episode's sha256, sample count, source,
  platform and exclusion reason; it is small enough to live in git. The raw
  ``.npz`` payloads live under ``datasets/`` which is gitignored (§10 note).

Format: one dict-of-arrays ``.npz`` per episode, flattened so that
:py:class:`~robotarm.data.schema.Episode` round-trips losslessly. Scalar text
fields are stored as length-1 object arrays; per-step flexibly-typed
``safety_flags`` / ``hardware_state`` are JSON-encoded into object arrays.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schema import Episode, StepRecord

_MANIFEST_NAME = "manifest.json"
_PAYLOAD_SUFFIX = ".npz"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _flatten_episode(ep: Episode) -> dict[str, np.ndarray]:
    n = len(ep.steps)
    obs_keys = list(ep.steps[0].observation.keys())
    out: dict[str, np.ndarray] = {
        "episode_id": np.array([ep.episode_id], dtype=object),
        "timestamp_ns": np.array([ep.timestamp_ns], dtype=np.int64),
        "platform": np.array([ep.platform], dtype=object),
        "task_id": np.array([ep.task_id], dtype=object),
        "target_id": np.array([ep.target_id], dtype=object),
        "split": np.array([ep.split], dtype=object),
        "damage_id": np.array([ep.damage_id], dtype=object),
        "seed": np.array([ep.seed], dtype=np.int64),
        "config_hash": np.array([ep.config_hash], dtype=object),
        "git_commit": np.array([ep.git_commit], dtype=object),
        "camera_frame_ref": np.array([ep.camera_frame_ref or ""], dtype=object),
        "joint_mask": np.asarray(ep.joint_mask, dtype=np.int64),
        "lock_angle": np.asarray(ep.lock_angle, dtype=np.float64),
        "n_steps": np.array([n], dtype=np.int64),
        "obs_keys": np.array(obs_keys, dtype=object),
        "action_commanded": _stack([s.action_commanded for s in ep.steps]),
        "action_applied": _stack([s.action_applied for s in ep.steps]),
        "reward": np.asarray([s.reward for s in ep.steps], dtype=np.float64),
        "success": np.asarray([int(s.success) for s in ep.steps], dtype=np.int64),
        "done": np.asarray([int(s.done) for s in ep.steps], dtype=np.int64),
        "safety_flags": np.array([json.dumps(s.safety_flags) for s in ep.steps], dtype=object),
        "hardware_state": np.array([json.dumps(s.hardware_state) for s in ep.steps], dtype=object),
    }
    for k in obs_keys:
        out[f"obs_{k}"] = _stack([s.observation[k] for s in ep.steps])
        out[f"nxt_{k}"] = _stack([s.next_observation[k] for s in ep.steps])
    return out


def _stack(arrays: list[np.ndarray]) -> np.ndarray:
    return np.stack(arrays, axis=0)


def _unflatten_episode(data: dict[str, np.ndarray]) -> Episode:
    n = int(data["n_steps"][0])
    obs_keys = [str(k) for k in data["obs_keys"]]
    steps: list[StepRecord] = []
    for i in range(n):
        obs = {k: data[f"obs_{k}"][i] for k in obs_keys}
        nxt = {k: data[f"nxt_{k}"][i] for k in obs_keys}
        steps.append(
            StepRecord(
                observation=obs,
                action_commanded=data["action_commanded"][i],
                action_applied=data["action_applied"][i],
                next_observation=nxt,
                reward=float(data["reward"][i]),
                success=bool(data["success"][i]),
                done=bool(data["done"][i]),
                safety_flags=json.loads(str(data["safety_flags"][i])),
                hardware_state=json.loads(str(data["hardware_state"][i])),
            )
        )
    return Episode(
        episode_id=str(data["episode_id"][0]),
        timestamp_ns=int(data["timestamp_ns"][0]),
        platform=str(data["platform"][0]),  # type: ignore[arg-type]
        task_id=str(data["task_id"][0]),
        target_id=str(data["target_id"][0]),
        split=str(data["split"][0]),  # type: ignore[arg-type]
        damage_id=str(data["damage_id"][0]),
        joint_mask=data["joint_mask"].copy(),
        lock_angle=data["lock_angle"].copy(),
        steps=steps,
        seed=int(data["seed"][0]),
        config_hash=str(data["config_hash"][0]),
        git_commit=str(data["git_commit"][0]),
        camera_frame_ref=str(data["camera_frame_ref"][0]) or None,
    )


@dataclass
class ManifestEntry:
    episode_id: str
    sha256: str
    n_samples: int
    source: str
    platform: str
    excluded: bool = False
    exclusion_reason: str | None = None


@dataclass
class EpisodeDataset:
    """Append-only, directory-backed collection of episodes."""

    root: Path
    version: str = "v1"
    _entries: dict[str, ManifestEntry] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / _MANIFEST_NAME
        if manifest_path.exists():
            self._load_manifest()

    def _payload_path(self, episode_id: str) -> Path:
        return self.root / f"{episode_id}{_PAYLOAD_SUFFIX}"

    # ------------------------------------------------------------------
    # Writing (append-only)
    # ------------------------------------------------------------------
    def add(self, episode: Episode, *, source: str = "sim") -> ManifestEntry:
        if episode.episode_id in self._entries:
            raise ValueError(f"episode {episode.episode_id!r} already exists; append-only")
        episode.validate()
        path = self._payload_path(episode.episode_id)
        if path.exists():
            raise ValueError(f"payload {path.name} already exists; refusing to overwrite")

        payload = _flatten_episode(episode)
        buf = io.BytesIO()
        np.savez(buf, **{k: v for k, v in sorted(payload.items())})
        raw = buf.getvalue()
        path.write_bytes(raw)

        entry = ManifestEntry(
            episode_id=episode.episode_id,
            sha256=_sha256_bytes(raw),
            n_samples=episode.length,
            source=source,
            platform=episode.platform,
        )
        self._entries[episode.episode_id] = entry
        self._write_manifest()
        return entry

    def exclude(self, episode_id: str, reason: str) -> None:
        """Mark an episode excluded with a reason; does NOT delete the payload."""
        if episode_id not in self._entries:
            raise KeyError(episode_id)
        self._entries[episode_id].excluded = True
        self._entries[episode_id].exclusion_reason = reason
        self._write_manifest()

    def sweep_invalid(self) -> list[str]:
        """Validate every stored episode; mark failures excluded; return their ids."""
        cleaned = []
        for ep_id in list(self._entries):
            if self._entries[ep_id].excluded:
                continue
            try:
                self.load(ep_id).validate()
            except Exception as e:  # noqa: BLE001 - validation sweep
                self.exclude(ep_id, f"validation failed: {e}")
                cleaned.append(ep_id)
        return cleaned

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def load(self, episode_id: str, *, validate: bool = True) -> Episode:
        path = self._payload_path(episode_id)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        ep = _unflatten_episode(data)
        if validate:
            ep.validate()
        return ep

    def __getitem__(self, episode_id: str) -> Episode:
        return self.load(episode_id)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterable[Episode]:
        for ep_id in self.ids():
            yield self.load(ep_id)

    def ids(self) -> list[str]:
        return sorted(self._entries)

    def manifest_entries(self) -> list[ManifestEntry]:
        return [self._entries[i] for i in self.ids()]

    # ------------------------------------------------------------------
    # Manifest & integrity
    # ------------------------------------------------------------------
    def _write_manifest(self) -> None:
        meta = {
            "version": self.version,
            "n_episodes": len(self._entries),
            "n_samples_total": sum(e.n_samples for e in self._entries.values()),
            "n_excluded": sum(1 for e in self._entries.values() if e.excluded),
            "episodes": [
                {
                    "episode_id": e.episode_id,
                    "sha256": e.sha256,
                    "n_samples": e.n_samples,
                    "source": e.source,
                    "platform": e.platform,
                    "excluded": e.excluded,
                    "exclusion_reason": e.exclusion_reason,
                }
                for e in self.manifest_entries()
            ],
        }
        text = json.dumps(meta, indent=2, ensure_ascii=False)
        # Atomic on POSIX. On Windows an AV/editor can briefly hold the target,
        # so if the rename fails we fall back to an in-place write rather than
        # corrupting the manifest.
        tmp = self.root / (".manifest.tmp")
        tmp.write_text(text)
        try:
            tmp.replace(self.root / _MANIFEST_NAME)
        except OSError:  # pragma: no cover - Windows-specific lock
            (self.root / _MANIFEST_NAME).write_text(text)
            tmp.unlink(missing_ok=True)

    def _load_manifest(self) -> None:
        meta = json.loads((self.root / _MANIFEST_NAME).read_text())
        self.version = meta.get("version", self.version)
        self._entries = {}
        for item in meta.get("episodes", []):
            self._entries[item["episode_id"]] = ManifestEntry(
                episode_id=item["episode_id"],
                sha256=item["sha256"],
                n_samples=item["n_samples"],
                source=item.get("source", "sim"),
                platform=item.get("platform", "sim"),
                excluded=bool(item.get("excluded", False)),
                exclusion_reason=item.get("exclusion_reason"),
            )

    def verify_integrity(self) -> dict[str, bool]:
        return {
            ep_id: entry.sha256 == _sha256_bytes(self._payload_path(ep_id).read_bytes())
            for ep_id, entry in self._entries.items()
        }

    # ------------------------------------------------------------------
    # Versioning (§10.2: cleaning makes a NEW dataset, never in-place)
    # ------------------------------------------------------------------
    def clean_version(self, new_root: Path, new_version: str) -> "EpisodeDataset":
        new_root = Path(new_root)
        if new_root.exists():
            raise FileExistsError(f"{new_root} already exists; refusing to clobber a dataset version")
        out = EpisodeDataset(root=new_root, version=new_version)
        for ep_id in self.ids():
            entry = self._entries[ep_id]
            if entry.excluded:
                continue
            out.add(self.load(ep_id), source=entry.source)
        return out