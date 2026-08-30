"""Frozen cross-arm contact action-effect Gate.

The structured model predicts a contact-frame object response as a context
bias plus a low-rank linear operator on the analytically lock-projected
end-effector candidate-action response.  The flat baseline receives exactly
the same observables and is parameter matched.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def standardizer(value: np.ndarray, train: np.ndarray):
    mean = value[train].mean(axis=0, keepdims=True)
    std = value[train].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean, std


def count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class ContactTransferOperator(nn.Module):
    def __init__(self, context_dim: int, hidden: int = 96, rank: int = 3):
        super().__init__()
        self.rank = rank
        self.context = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.bias = nn.Linear(hidden, 9)
        self.left = nn.Linear(hidden, 9 * rank)
        self.right = nn.Linear(hidden, 3 * rank)

    def forward(self, context, ee_response):
        hidden = self.context(context)
        left = self.left(hidden).view(-1, 9, self.rank)
        right = self.right(hidden).view(-1, self.rank, 3)
        transfer = torch.bmm(left, right)
        return self.bias(hidden) + torch.bmm(transfer, ee_response[..., None]).squeeze(-1)


class FlatResponse(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 9),
        )

    def forward(self, value):
        return self.net(value)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def ranking_metrics(actual, predicted, data, select):
    directions = np.asarray(((1, 0), (0, 1), (-1, 0), (0, -1)), dtype=np.float64)
    correlations, regrets = [], []
    keys = np.stack([data["prefix_id"], data["lock_index"]], axis=1)
    for robot in ("genkiarm", "panda"):
        robot_rows = np.flatnonzero(select & (data["robot"].astype(str) == robot))
        for key in np.unique(keys[robot_rows], axis=0):
            rows = robot_rows[np.all(keys[robot_rows] == key, axis=1)]
            if len(rows) < 3:
                continue
            for direction in directions:
                truth = actual[rows, :2] @ direction
                estimate = predicted[rows, :2] @ direction
                rt, rp = rankdata(truth), rankdata(estimate)
                if np.std(rt) > 0 and np.std(rp) > 0:
                    correlations.append(float(np.corrcoef(rt, rp)[0, 1]))
                chosen = int(np.argmax(estimate))
                scale = max(float(np.ptp(truth)), 1e-8)
                regrets.append(float((np.max(truth) - truth[chosen]) / scale))
    return {
        "mean_spearman": float(np.mean(correlations)),
        "normalized_top1_regret": float(np.mean(regrets)),
        "groups": len(correlations),
    }


def prediction_metrics(actual, predicted, robots, select):
    result = {}
    for robot in ("genkiarm", "panda"):
        rows = select & (robots == robot)
        result[robot] = float(np.sqrt(np.mean((actual[rows] - predicted[rows]) ** 2)))
    result["pooled"] = float(np.sqrt(np.mean((actual[select] - predicted[select]) ** 2)))
    return result


def run(dataset: Path, *, seed: int, epochs: int = 1600):
    with np.load(dataset) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    robots = data["robot"].astype(str)
    rng = np.random.default_rng(seed)
    train_prefix = {}
    for robot in ("genkiarm", "panda"):
        ids = np.unique(data["prefix_id"][robots == robot])
        shuffled = rng.permutation(ids)
        train_prefix[robot] = set(map(int, shuffled[: int(0.7 * len(ids))]))
    prefix_train = np.asarray([
        int(prefix) in train_prefix[robot]
        for prefix, robot in zip(data["prefix_id"], robots)
    ])
    middle = np.where(robots == "genkiarm", 2, 3)
    heldout = data["lock_index"] == middle
    train = prefix_train & ~heldout
    validation = ~prefix_train & ~heldout
    test = ~prefix_train & heldout

    lock_depth = data["lock_index"][:, None] / np.maximum(data["dof"][:, None] - 1, 1)
    lock_angle = np.sum(data["angle"] * data["mask"], axis=1, keepdims=True)
    context = np.concatenate([
        data["object_pose"], data["object_twist"], data["ee_object_relative"],
        data["ee_action_delta"], lock_depth, lock_angle,
    ], axis=1).astype(np.float32)
    response = data["ee_projected_action"].astype(np.float32)
    target = data["locked_object_step"].astype(np.float32)
    context_mean, context_std = standardizer(context, train)
    response_mean, response_std = standardizer(response, train)
    target_mean, target_std = standardizer(target, train)
    context_n = (context - context_mean) / context_std
    response_n = (response - response_mean) / response_std
    target_n = (target - target_mean) / target_std
    flat_x = np.concatenate([context_n, response_n], axis=1)

    tensors = {
        "context": torch.as_tensor(context_n), "response": torch.as_tensor(response_n),
        "flat": torch.as_tensor(flat_x), "target": torch.as_tensor(target_n),
    }
    torch.manual_seed(seed)
    structured = ContactTransferOperator(context_n.shape[1])
    candidates = [FlatResponse(flat_x.shape[1], hidden) for hidden in range(32, 321)]
    flat = min(candidates, key=lambda model: abs(count(model) - count(structured)))
    relative_parameter_difference = abs(count(flat) - count(structured)) / count(structured)
    if relative_parameter_difference > 0.05:
        raise RuntimeError("unable to parameter-match flat baseline")

    def fit(model, predict):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        ti = torch.as_tensor(np.flatnonzero(train)); vi = torch.as_tensor(np.flatnonzero(validation))
        best, best_loss, stale = None, float("inf"), 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = torch.mean((predict(model, ti) - tensors["target"][ti]) ** 2)
            loss.backward(); optimizer.step()
            if epoch % 10:
                continue
            with torch.no_grad():
                value = float(torch.mean((predict(model, vi) - tensors["target"][vi]) ** 2))
            if value < best_loss - 1e-7:
                best, best_loss, stale = copy.deepcopy(model.state_dict()), value, 0
            else:
                stale += 1
            if stale >= 25:
                break
        model.load_state_dict(best)
        return model

    structured_predict = lambda model, rows: model(tensors["context"][rows], tensors["response"][rows])
    flat_predict = lambda model, rows: model(tensors["flat"][rows])
    fit(structured, structured_predict); fit(flat, flat_predict)
    all_rows = torch.arange(len(target))
    with torch.no_grad():
        sp = structured_predict(structured, all_rows).numpy() * target_std + target_mean
        fp = flat_predict(flat, all_rows).numpy() * target_std + target_mean
    sm, fm = prediction_metrics(target, sp, robots, test), prediction_metrics(target, fp, robots, test)
    sr, fr = ranking_metrics(target, sp, data, test), ranking_metrics(target, fp, data, test)
    return {
        "version": "ipwm_contact_action_effect_gate_v1", "seed": seed,
        "split": {"train": int(train.sum()), "validation": int(validation.sum()), "test": int(test.sum()),
                  "grouped_by_prefix": True, "heldout_locks": {"genkiarm": "j3", "panda": "joint4"}},
        "parameters": {"structured": count(structured), "flat": count(flat),
                       "relative_difference": relative_parameter_difference},
        "flat": {"prediction": fm, "ranking": fr},
        "structured": {"prediction": sm, "ranking": sr},
        "relative_pooled_rmse_improvement": (fm["pooled"] - sm["pooled"]) / fm["pooled"],
        "spearman_absolute_improvement": sr["mean_spearman"] - fr["mean_spearman"],
        "both_robots_prediction_improve": all(sm[r] < fm[r] for r in ("genkiarm", "panda")),
        "lower_top1_regret": sr["normalized_top1_regret"] < fr["normalized_top1_regret"],
    }


def main():
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
