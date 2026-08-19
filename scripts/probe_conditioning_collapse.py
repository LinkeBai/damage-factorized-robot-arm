"""Probe whether topology conditioning actually enters the GRU hidden state.

Three probes, from upstream to downstream:

  A. e_topology linear probe   — sanity check; must be ~100% (deterministic encoder)
  B. GRU hidden-state probe    — core question; ≈50% means conditioning collapse
  C. Per-condition error split — do structured/ordinary differ by condition?

Usage (1 seed, fast):
  python scripts/probe_conditioning_collapse.py --seed 7 --epochs 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow `from scripts.run_push_benchmark import ...` when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from scipy.special import expit
from scipy.optimize import minimize

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.g1_mechanism import TOPOLOGY_DIM, encode_damage_batch
from robotarm.training.sim_protocol import damage_from_name, load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_ensemble import (
    ConditionMode,
    TopologyMember,
    conditioning_damages,
    train_topology_ensemble,
)
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


# ── linear probe helpers ─────────────────────────────────────────────────────

def _logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, C: float = 1.0):
    n = len(y)
    logits = X @ w[:-1] + w[-1]
    probs = expit(logits)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    nll = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
    reg = 0.5 / C * np.sum(w[:-1] ** 2) / n
    return nll + reg


def _linear_probe_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    """Stratified k-fold logistic-regression accuracy (numpy/scipy, no sklearn)."""
    rng = np.random.default_rng(0)
    # manual stratified split
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    folds0 = np.array_split(idx0, n_splits)
    folds1 = np.array_split(idx1, n_splits)
    accs = []
    for k in range(n_splits):
        val_idx = np.concatenate([folds0[k], folds1[k]])
        train_idx = np.concatenate(
            [folds0[j] for j in range(n_splits) if j != k]
            + [folds1[j] for j in range(n_splits) if j != k]
        )
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        # z-score on train stats
        mu = X_tr.mean(0)
        std = X_tr.std(0) + 1e-8
        X_tr = (X_tr - mu) / std
        X_va = (X_va - mu) / std
        w0 = np.zeros(X_tr.shape[1] + 1)
        res = minimize(_logistic_loss, w0, args=(X_tr, y_tr.astype(float)),
                       method="L-BFGS-B", options={"maxiter": 500})
        w = res.x
        preds = (X_va @ w[:-1] + w[-1] > 0).astype(int)
        accs.append(np.mean(preds == y_va))
    return float(np.mean(accs))


# ── probe A: e_topology ───────────────────────────────────────────────────────

def probe_topology_encoder(
    encoder: torch.nn.Module,
    joint_ranges: np.ndarray,
    device: torch.device,
    n_samples: int = 200,
) -> float:
    """Check that D2 vs D3 is linearly separable in e_topology space."""
    X, y = [], []
    for label, name in enumerate(["D2", "D3"]):
        damages = [damage_from_name(name) for _ in range(n_samples)]
        with torch.no_grad():
            emb = encode_damage_batch(encoder, damages, joint_ranges, device)
        X.append(emb.cpu().numpy())
        y.extend([label] * n_samples)
    X_arr = np.concatenate(X, axis=0)
    y_arr = np.array(y)
    return _linear_probe_cv(X_arr, y_arr)


# ── probe B: GRU hidden states ────────────────────────────────────────────────

@torch.no_grad()
def collect_hidden_states(
    member: TopologyMember,
    trajectories_d2: list,
    trajectories_d3: list,
    joint_ranges: np.ndarray,
    device: torch.device,
    condition_mode: ConditionMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Run rollouts and collect per-step GRU hidden states + labels."""
    hiddens, labels = [], []

    for condition_label, (trajs, cond_name) in enumerate([
        (trajectories_d2, "D2"),
        (trajectories_d3, "D3"),
    ]):
        damages = conditioning_damages(
            [damage_from_name(cond_name)] * len(trajs), condition_mode
        )
        states = torch.stack([t.states for t in trajs]).to(device)   # (N, T, S)
        actions = torch.stack([t.actions for t in trajs]).to(device)  # (N, T, A)
        context = encode_damage_batch(member.encoder, damages, joint_ranges, device)

        N, T, _ = states.shape
        hidden = None
        for t in range(T - 1):
            _, hidden = member.world_model.step(
                states[:, t], actions[:, t], context, hidden
            )
            hiddens.append(hidden.cpu().numpy())          # (N, H)
            labels.extend([condition_label] * N)

    return np.concatenate(hiddens, axis=0), np.array(labels)


def probe_hidden_states(
    ensemble: list[TopologyMember],
    trajectories_d2: list,
    trajectories_d3: list,
    joint_ranges: np.ndarray,
    device: torch.device,
    condition_mode: ConditionMode,
) -> list[float]:
    """Probe each member separately; return per-member accuracy."""
    accs = []
    for member in ensemble:
        X, y = collect_hidden_states(
            member, trajectories_d2, trajectories_d3, joint_ranges, device,
            condition_mode,
        )
        accs.append(_linear_probe_cv(X, y))
    return accs


# ── probe C: per-condition RMSE ───────────────────────────────────────────────

@torch.no_grad()
def per_condition_rmse(
    ensemble: list[TopologyMember],
    trajectories_d2: list,
    trajectories_d3: list,
    joint_ranges: np.ndarray,
    device: torch.device,
    condition_mode: ConditionMode,
    horizon: int = 10,
) -> dict[str, float]:
    results = {}
    for cond_name, trajs in [("D2", trajectories_d2), ("D3", trajectories_d3)]:
        damages = conditioning_damages(
            [damage_from_name(cond_name)] * len(trajs), condition_mode
        )
        states = torch.stack([t.states for t in trajs]).to(device)
        actions = torch.stack([t.actions for t in trajs]).to(device)
        contexts = [
            encode_damage_batch(member.encoder, damages, joint_ranges, device)
            for member in ensemble
        ]
        sq_errors = []
        T = actions.shape[1]
        h = min(horizon, T)
        for start in range(0, T - h + 1, h):
            preds = [states[:, start].clone() for _ in ensemble]
            hidden = [None] * len(ensemble)
            for offset in range(h):
                means = []
                for i, member in enumerate(ensemble):
                    out, hidden[i] = member.world_model.step(
                        preds[i], actions[:, start + offset], contexts[i], hidden[i]
                    )
                    preds[i] = out["mean"]
                    means.append(out["mean"])
                mean_pred = torch.stack(means).mean(0)
                target = states[:, start + offset + 1]
                sq_errors.append((mean_pred - target).pow(2).mean().item())
        results[cond_name] = float(np.sqrt(np.mean(sq_errors)))
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/experiment/g2_push_ensemble_v1.yaml"),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    epochs = args.epochs or int(config["epochs"])
    steps = int(config["steps"])
    members = int(config["members"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  seed={args.seed}  epochs={epochs}  members={members}")

    protocol = load_g1_protocol(Path(config["protocol"]))
    targets = load_target_split(Path(config["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    block_initial_xy = np.asarray(config["block_initial_xy"], dtype=float)
    joint_ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

    # ── training data ─────────────────────────────────────────────────────────
    print("\n[train] collecting trajectories …", flush=True)
    train_trajs = collect_push_domains(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps,
        seed=args.seed * 10_000,
        targets=calibration,
        excitation="goal",
        block_initial_xy=block_initial_xy,
    )

    # ── train both ensembles ──────────────────────────────────────────────────
    ensembles: dict[str, list[TopologyMember]] = {}
    for method, mode in config["condition_modes"].items():
        print(f"\n[train] {method} …", flush=True)
        t0 = time.perf_counter()
        ensembles[method] = train_topology_ensemble(
            train_trajs, joint_ranges, members=members, epochs=epochs,
            device=device, seed=args.seed, condition_mode=mode,
        )
        print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)

    # ── test trajectories per condition ───────────────────────────────────────
    print("\n[eval] collecting test trajectories …", flush=True)
    test_by_cond: dict[str, list] = {}
    for domain in protocol.test:
        cond = domain.domain_id.split("__")[0]
        trajs = collect_push_domains(
            (domain,),
            trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            steps=steps,
            seed=args.seed * 100_000 + 500,
            targets=evaluation,
            excitation="goal",
            block_initial_xy=block_initial_xy,
        )
        test_by_cond.setdefault(cond, []).extend(trajs)
    d2_trajs = test_by_cond.get("D2", [])
    d3_trajs = test_by_cond.get("D3", [])
    print(f"  D2: {len(d2_trajs)} trajectories,  D3: {len(d3_trajs)} trajectories")

    # ── probes ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PROBE A — e_topology linear separability (sanity check)")
    print("=" * 60)
    for method, ensemble in ensembles.items():
        acc = probe_topology_encoder(ensemble[0].encoder, joint_ranges, device)
        print(f"  {method}: {acc * 100:.1f}%  (expect ~100%)")

    print("\n" + "=" * 60)
    print("PROBE B — GRU hidden-state linear probe")
    print("  Q: does the hidden state carry D2 vs D3 signal?")
    print("  50% = conditioning collapse; 100% = perfect signal")
    print("=" * 60)
    for method, mode in config["condition_modes"].items():
        ensemble = ensembles[method]
        accs = probe_hidden_states(
            ensemble, d2_trajs, d3_trajs, joint_ranges, device, mode,
        )
        mean_acc = np.mean(accs)
        print(f"  {method}:")
        for i, a in enumerate(accs):
            print(f"    member {i}: {a * 100:.1f}%")
        print(f"    mean: {mean_acc * 100:.1f}%")

    print("\n" + "=" * 60)
    print("PROBE C — per-condition ensemble RMSE")
    print("  Q: does topology conditioning help more on one condition?")
    print("=" * 60)
    for method, mode in config["condition_modes"].items():
        ensemble = ensembles[method]
        rmse = per_condition_rmse(
            ensemble, d2_trajs, d3_trajs, joint_ranges, device, mode,
        )
        print(f"  {method}:  D2={rmse['D2']:.4f}  D3={rmse['D3']:.4f}")

    print("\n[done]")


if __name__ == "__main__":
    main()
