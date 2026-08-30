"""Held-out-joint deployable propagation gate for exact-prefix contact data.

Train only on j2/j4 lock interventions and test on unseen j3 locks from held-
out episodes.  Both models receive the same deployable state, action and
continuous lock location/features.  The structured model predicts each output
node with a shared propagation function; the baseline predicts the full vector
with a parameter-matched unstructured MLP.  Solver forces are labels for the
mechanism audit and are deliberately not loaded here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


LOCK_TO_JOINT = np.array([1, 2, 3], dtype=np.int64)  # dataset: j2, j3, j4
NODE_OUTPUT_INDEX = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)


def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((actual - actual.mean(axis=0, keepdims=True)) ** 2))
    if denominator <= 1e-12:
        return float("nan")
    return 1.0 - float(np.sum((actual - predicted) ** 2)) / denominator


def base_features(state: np.ndarray, action: np.ndarray, lock: np.ndarray) -> np.ndarray:
    joint = LOCK_TO_JOINT[lock]
    row = np.arange(len(lock))
    lock_coord = joint[:, None] / 4.0
    lock_local = np.stack([
        state[row, joint], state[row, 5 + joint], action[row, joint]
    ], axis=1)
    return np.concatenate([state, action, lock_coord, lock_local], axis=1)


def structured_features(
    state: np.ndarray, action: np.ndarray, lock: np.ndarray
) -> np.ndarray:
    """Return [samples, seven output nodes, features]."""
    base = base_features(state, action, lock)
    lock_joint = LOCK_TO_JOINT[lock]
    lock_coord = lock_joint / 4.0
    rows: list[np.ndarray] = []
    for node in range(7):
        if node < 2:  # block x/y velocity response
            node_coord = np.full(len(lock), 1.25)
            node_type = np.ones(len(lock))
            axis = np.full(len(lock), float(node))
            downstream = np.ones(len(lock))
            local = np.stack([
                state[:, 10 + node], state[:, 12 + node], np.zeros(len(lock))
            ], axis=1)
        else:
            joint = node - 2
            node_coord = np.full(len(lock), joint / 4.0)
            node_type = np.zeros(len(lock))
            axis = np.zeros(len(lock))
            downstream = (joint >= lock_joint).astype(np.float64)
            local = np.stack([
                state[:, joint], state[:, 5 + joint], action[:, joint]
            ], axis=1)
        signed_distance = node_coord - lock_coord
        structural = np.stack([
            node_coord, signed_distance, np.maximum(signed_distance, 0.0),
            downstream, node_type, axis,
        ], axis=1)
        rows.append(np.concatenate([base, structural, local], axis=1))
    return np.stack(rows, axis=1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _standardize(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, [(x - mean) / std for x in others], mean, std


def train_model(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    seed: int,
    epochs: int = 2500,
) -> nn.Module:
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    tx = torch.as_tensor(train_x, dtype=torch.float32)
    ty = torch.as_tensor(train_y, dtype=torch.float32)
    vx = torch.as_tensor(val_x, dtype=torch.float32)
    vy = torch.as_tensor(val_y, dtype=torch.float32)
    best_loss = float("inf")
    best = None
    patience = 300
    stale = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = torch.mean((model(tx) - ty) ** 2)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((model(vx) - vy) ** 2))
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best is not None:
        model.load_state_dict(best)
    return model


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "all_rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "all_r2": r2(actual, predicted),
        "object_rmse": float(np.sqrt(np.mean((actual[:, :2] - predicted[:, :2]) ** 2))),
        "object_r2": r2(actual[:, :2], predicted[:, :2]),
        "robot_rmse": float(np.sqrt(np.mean((actual[:, 2:] - predicted[:, 2:]) ** 2))),
        "robot_r2": r2(actual[:, 2:], predicted[:, 2:]),
    }


def run(dataset: Path, *, seed: int) -> dict[str, object]:
    with np.load(dataset) as data:
        state = np.asarray(data["state"], dtype=np.float64)
        action = np.asarray(data["action"], dtype=np.float64)
        lock = np.asarray(data["lock"], dtype=np.int64)
        episode = np.asarray(data["episode"], dtype=np.int64)
        target = np.asarray(data["actual_full"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    episodes = np.unique(episode)
    shuffled = rng.permutation(episodes)
    train_episodes = set(int(x) for x in shuffled[: max(1, int(0.7 * len(episodes)))])
    train = np.array([(e in train_episodes) and (fault != 1) for e, fault in zip(episode, lock)])
    validation = np.array([(e not in train_episodes) and (fault != 1) for e, fault in zip(episode, lock)])
    test = np.array([(e not in train_episodes) and (fault == 1) for e, fault in zip(episode, lock)])
    if min(train.sum(), validation.sum(), test.sum()) == 0:
        raise ValueError("empty train, validation, or held-out-j3 split")

    bx = base_features(state, action, lock)
    sx = structured_features(state, action, lock)
    bx_train, (bx_val, bx_test), bx_mean, bx_std = _standardize(
        bx[train], bx[validation], bx[test]
    )
    flat_train = sx[train].reshape(-1, sx.shape[-1])
    flat_val = sx[validation].reshape(-1, sx.shape[-1])
    flat_test = sx[test].reshape(-1, sx.shape[-1])
    sx_train, (sx_val, sx_test), sx_mean, sx_std = _standardize(
        flat_train, flat_val, flat_test
    )
    y_train, (y_val, y_test), y_mean, y_std = _standardize(
        target[train], target[validation], target[test]
    )

    unstructured = MLP(bx_train.shape[1], 7, hidden=64)
    structured = MLP(sx_train.shape[1], 1, hidden=64)
    unstructured = train_model(
        unstructured, bx_train, y_train, bx_val, y_val, seed=seed
    )
    structured = train_model(
        structured,
        sx_train,
        y_train.reshape(-1, 1),
        sx_val,
        y_val.reshape(-1, 1),
        seed=seed,
    )
    with torch.no_grad():
        up = unstructured(torch.as_tensor(bx_test, dtype=torch.float32)).numpy()
        sp = structured(torch.as_tensor(sx_test, dtype=torch.float32)).numpy().reshape(-1, 7)
    up = up * y_std + y_mean
    sp = sp * y_std + y_mean
    actual = target[test]
    unstructured_metrics = evaluate(actual, up)
    structured_metrics = evaluate(actual, sp)
    relative = (
        unstructured_metrics["all_rmse"] - structured_metrics["all_rmse"]
    ) / unstructured_metrics["all_rmse"]
    return {
        "version": "deployable_constraint_propagation_gate_v1",
        "dataset": str(dataset),
        "seed": seed,
        "split": {
            "training_faults": ["j2", "j4"],
            "heldout_fault": "j3",
            "training_episodes": sorted(train_episodes),
            "test_episodes": sorted(set(int(x) for x in episodes) - train_episodes),
            "train_rows": int(train.sum()),
            "validation_rows": int(validation.sum()),
            "test_rows": int(test.sum()),
        },
        "deployment_inputs": ["14D state", "5D action", "continuous lock location/local state-action"],
        "solver_force_input": False,
        "parameter_counts": {
            "unstructured": parameter_count(unstructured),
            "structured": parameter_count(structured),
            "relative_difference": abs(parameter_count(unstructured) - parameter_count(structured))
            / parameter_count(unstructured),
        },
        "zero": evaluate(actual, np.zeros_like(actual)),
        "unstructured": unstructured_metrics,
        "structured": structured_metrics,
        "structured_relative_all_rmse_improvement": float(relative),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dataset, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
