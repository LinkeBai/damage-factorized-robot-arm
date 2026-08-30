"""Deployable cross-arm contact counterfactual object-response Gate."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robotarm.models.variable_dof_ipwm import VariableDofInterventionCore


MAX_DOF = 7


class StructuredObjectResponse(nn.Module):
    def __init__(self, hidden_dim: int = 64, object_hidden: int = 96):
        super().__init__()
        self.core = VariableDofInterventionCore(hidden_dim=hidden_dim)
        self.object_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 16, object_hidden), nn.SiLU(),
            nn.Linear(object_hidden, object_hidden), nn.SiLU(),
            nn.Linear(object_hidden, 9),
        )

    def forward(self, state, action, mask, angle, valid, axes, origins, object_features):
        _, _, hidden = self.core.encode_nodes(
            state, action, mask, angle, valid, axes, origins
        )
        valid_f = valid.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
        locked = (hidden * mask.unsqueeze(-1)).sum(dim=1)
        return self.object_head(torch.cat([pooled, locked, object_features], dim=-1))


class FlatObjectResponse(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 9),
        )

    def forward(self, value):
        return self.net(value)


def count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def standardizer(train: np.ndarray):
    mean, std = train.mean(axis=0, keepdims=True), train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean, std


def flat_features(data) -> np.ndarray:
    valid = np.arange(MAX_DOF)[None] < data["dof"][:, None]
    node = np.concatenate([
        data["state"], data["action"][..., None], data["mask"][..., None],
        data["angle"][..., None], data["axes"], data["origins"], valid[..., None],
    ], axis=-1).reshape(len(valid), -1)
    objects = np.concatenate([
        data["object_pose"], data["object_twist"], data["ee_object_relative"]
    ], axis=-1)
    return np.concatenate([node, objects], axis=-1)


def metrics(actual, predicted, robots, select):
    result = {}
    for robot in ("genkiarm", "panda"):
        rows = select & (robots == robot)
        error = actual[rows] - predicted[rows]
        result[robot] = {
            "all_rmse": float(np.sqrt(np.mean(error ** 2))),
            "position_rmse": float(np.sqrt(np.mean(error[:, :3] ** 2))),
            "twist_rmse": float(np.sqrt(np.mean(error[:, 3:] ** 2))),
            "target_rms": float(np.sqrt(np.mean(actual[rows] ** 2))),
        }
    error = actual[select] - predicted[select]
    result["pooled"] = {"all_rmse": float(np.sqrt(np.mean(error ** 2)))}
    return result


def run(dataset: Path, *, seed: int, epochs: int = 1200):
    with np.load(dataset) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    robots = data["robot"].astype(str)
    rng = np.random.default_rng(seed)
    train_prefix = {}
    for robot in ("genkiarm", "panda"):
        ids = np.unique(data["prefix_id"][robots == robot])
        shuffled = rng.permutation(ids)
        train_prefix[robot] = set(int(x) for x in shuffled[: int(0.7 * len(ids))])
    middle = np.where(robots == "genkiarm", 2, 3)
    prefix_is_train = np.array([
        int(prefix) in train_prefix[robot] for prefix, robot in zip(data["prefix_id"], robots)
    ])
    is_middle = data["lock_index"] == middle
    train = prefix_is_train & ~is_middle
    validation = ~prefix_is_train & ~is_middle
    test = ~prefix_is_train & is_middle

    valid = np.arange(MAX_DOF)[None] < data["dof"][:, None]
    object_features = np.concatenate([
        data["object_pose"], data["object_twist"], data["ee_object_relative"]
    ], axis=-1).astype(np.float32)
    object_mean, object_std = standardizer(object_features[train])
    object_features = (object_features - object_mean) / object_std
    target = data["object_delta"].astype(np.float32)
    target_mean, target_std = standardizer(target[train])
    normalized_target = (target - target_mean) / target_std
    flat_x = flat_features(data).astype(np.float32)
    flat_mean, flat_std = standardizer(flat_x[train]); flat_x = (flat_x - flat_mean) / flat_std
    tensors = {
        key: torch.as_tensor(data[key], dtype=torch.float32) for key in
        ("state", "action", "mask", "angle", "axes", "origins")
    }
    tensors["valid"] = torch.as_tensor(valid)
    tensors["object"] = torch.as_tensor(object_features, dtype=torch.float32)
    tensors["target"] = torch.as_tensor(normalized_target, dtype=torch.float32)
    tensors["flat"] = torch.as_tensor(flat_x, dtype=torch.float32)

    torch.manual_seed(seed)
    structured = StructuredObjectResponse()
    candidates = [FlatObjectResponse(flat_x.shape[1], hidden) for hidden in range(64, 321)]
    flat = min(candidates, key=lambda model: abs(count(model) - count(structured)))
    parameter_difference = abs(count(flat) - count(structured)) / count(structured)
    if parameter_difference > 0.05:
        raise RuntimeError("unable to parameter-match flat baseline")

    def structured_predict(model, indices):
        return model(
            tensors["state"][indices], tensors["action"][indices],
            tensors["mask"][indices], tensors["angle"][indices],
            tensors["valid"][indices], tensors["axes"][indices],
            tensors["origins"][indices], tensors["object"][indices],
        )

    def flat_predict(model, indices):
        return model(tensors["flat"][indices])

    def fit(model, predictor):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        train_indices = torch.as_tensor(np.flatnonzero(train), dtype=torch.long)
        val_indices = torch.as_tensor(np.flatnonzero(validation), dtype=torch.long)
        best, best_loss, stale = None, float("inf"), 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = torch.mean((predictor(model, train_indices) - tensors["target"][train_indices]) ** 2)
            loss.backward(); optimizer.step()
            if epoch % 10:
                continue
            with torch.no_grad():
                val = float(torch.mean((predictor(model, val_indices) - tensors["target"][val_indices]) ** 2))
            if val < best_loss - 1e-7:
                best_loss, stale, best = val, 0, copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= 25:
                break
        model.load_state_dict(best)
        return model

    structured = fit(structured, structured_predict)
    flat = fit(flat, flat_predict)
    all_indices = torch.arange(len(target))
    with torch.no_grad():
        structured_prediction = structured_predict(structured, all_indices).numpy() * target_std + target_mean
        flat_prediction = flat_predict(flat, all_indices).numpy() * target_std + target_mean
    zero = np.zeros_like(target)
    structured_metrics = metrics(target, structured_prediction, robots, test)
    flat_metrics = metrics(target, flat_prediction, robots, test)
    return {
        "version": "cross_arm_contact_gate_v1", "seed": seed,
        "split": {"train_rows": int(train.sum()), "validation_rows": int(validation.sum()),
                  "test_rows": int(test.sum()), "grouped_by_prefix": True,
                  "heldout_locks": {"genkiarm": "j3", "panda": "joint4"}},
        "parameters": {"structured": count(structured), "flat": count(flat),
                       "relative_difference": parameter_difference},
        "zero": metrics(target, zero, robots, test),
        "flat": flat_metrics, "structured": structured_metrics,
        "relative_pooled_improvement": (
            flat_metrics["pooled"]["all_rmse"] - structured_metrics["pooled"]["all_rmse"]
        ) / flat_metrics["pooled"]["all_rmse"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = run(args.dataset, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
