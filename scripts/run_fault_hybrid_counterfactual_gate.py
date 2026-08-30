"""Run the frozen H10 fault-conditioned hybrid counterfactual Gate."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def standardize(value, train):
    mean = value[train].mean(0, keepdims=True)
    std = value[train].std(0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean, std


class HybridMixture(nn.Module):
    def __init__(self, input_dim, hidden=96):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())
        self.mode = nn.Linear(hidden, 1)
        self.contact = nn.Linear(hidden, 9)
        self.detached = nn.Linear(hidden, 9)

    def forward(self, value):
        hidden = self.trunk(value)
        logit = self.mode(hidden).squeeze(-1)
        probability = torch.sigmoid(logit).unsqueeze(-1)
        contact, detached = self.contact(hidden), self.detached(hidden)
        prediction = probability * contact + (1.0 - probability) * detached
        return prediction, logit, contact, detached


class FlatMultiTask(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())
        self.response, self.mode = nn.Linear(hidden, 9), nn.Linear(hidden, 1)

    def forward(self, value):
        hidden = self.trunk(value)
        return self.response(hidden), self.mode(hidden).squeeze(-1)


class FlatResponse(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 9))

    def forward(self, value):
        return self.net(value)


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values)); ranks[order] = np.arange(len(values))
    return ranks


def action_metrics(actual, predicted, data, select):
    directions = np.asarray(((1, 0), (0, 1), (-1, 0), (0, -1)))
    correlations, regrets = [], []
    robots = data["robot"].astype(str)
    keys = np.stack([(robots == "panda").astype(int), data["prefix_id"], data["lock_index"]], 1)
    selected = np.flatnonzero(select)
    for key in np.unique(keys[selected], axis=0):
        rows = selected[np.all(keys[selected] == key, axis=1)]
        for direction in directions:
            truth, estimate = actual[rows, :2] @ direction, predicted[rows, :2] @ direction
            rt, rp = rankdata(truth), rankdata(estimate)
            if np.std(rt) > 0 and np.std(rp) > 0:
                correlations.append(float(np.corrcoef(rt, rp)[0, 1]))
            choice, scale = int(np.argmax(estimate)), max(float(np.ptp(truth)), 1e-8)
            regrets.append(float((np.max(truth) - truth[choice]) / scale))
    return {"mean_spearman": float(np.mean(correlations)),
            "normalized_top1_regret": float(np.mean(regrets)), "groups": len(correlations)}


def response_metrics(actual, predicted, robots, select):
    result = {}
    for robot in ("genkiarm", "panda"):
        rows = select & (robots == robot)
        result[robot] = float(np.sqrt(np.mean((actual[rows] - predicted[rows]) ** 2)))
    result["pooled"] = float(np.sqrt(np.mean((actual[select] - predicted[select]) ** 2)))
    return result


def mode_metrics(label, probability, select):
    y, p = label[select], probability[select]
    predicted = p >= 0.5
    tpr = np.mean(predicted[y == 1]) if np.any(y == 1) else np.nan
    tnr = np.mean(~predicted[y == 0]) if np.any(y == 0) else np.nan
    return {"balanced_accuracy": float(np.nanmean([tpr, tnr])),
            "brier_score": float(np.mean((p - y) ** 2)),
            "positive_fraction": float(np.mean(y))}


def run(dataset: Path, seed: int, epochs=1600):
    with np.load(dataset) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    robots = data["robot"].astype(str)
    rng = np.random.default_rng(seed)
    train_ids = {}
    for robot in ("genkiarm", "panda"):
        ids = rng.permutation(np.unique(data["prefix_id"][robots == robot]))
        train_ids[robot] = set(map(int, ids[:int(0.7 * len(ids))]))
    prefix_train = np.asarray([int(prefix) in train_ids[robot]
                               for prefix, robot in zip(data["prefix_id"], robots)])
    middle = np.where(robots == "genkiarm", 2, 3)
    heldout = data["lock_index"] == middle
    train, validation, test = prefix_train & ~heldout, ~prefix_train & ~heldout, ~prefix_train & heldout
    depth = data["lock_index"][:, None] / np.maximum(data["dof"][:, None] - 1, 1)
    angle = np.sum(data["angle"] * data["mask"], axis=1, keepdims=True)
    features = np.concatenate([data["object_pose"], data["object_twist"],
        data["ee_object_relative"], data["ee_action_delta"],
        data["ee_projected_action"], depth, angle], 1).astype(np.float32)
    target = data["locked_object_step"].astype(np.float32)
    label = data["locked_contact_after"].astype(np.float32)
    fm, fs = standardize(features, train); tm, ts = standardize(target, train)
    x, y = (features - fm) / fs, (target - tm) / ts
    tx, ty, tl = torch.as_tensor(x), torch.as_tensor(y), torch.as_tensor(label)
    torch.manual_seed(seed)
    hybrid = HybridMixture(x.shape[1])
    multi_candidates = [FlatMultiTask(x.shape[1], h) for h in range(32, 321)]
    plain_candidates = [FlatResponse(x.shape[1], h) for h in range(32, 321)]
    multi = min(multi_candidates, key=lambda m: abs(count(m) - count(hybrid)))
    plain = min(plain_candidates, key=lambda m: abs(count(m) - count(hybrid)))
    differences = {"multitask": abs(count(multi)-count(hybrid))/count(hybrid),
                   "non_mixture": abs(count(plain)-count(hybrid))/count(hybrid)}
    if max(differences.values()) > 0.05:
        raise RuntimeError("parameter matching failed")
    ti, vi = torch.as_tensor(np.flatnonzero(train)), torch.as_tensor(np.flatnonzero(validation))
    bce = nn.BCEWithLogitsLoss()

    def hybrid_loss(model, rows):
        prediction, logit, contact, detached = model(tx[rows])
        mode = tl[rows]
        conditional = ((contact - ty[rows])**2 * mode[:, None] +
                       (detached - ty[rows])**2 * (1-mode[:, None])).mean()
        return ((prediction-ty[rows])**2).mean() + conditional + 0.2*bce(logit, mode)
    def multi_loss(model, rows):
        prediction, logit = model(tx[rows])
        return ((prediction-ty[rows])**2).mean() + 0.2*bce(logit, tl[rows])
    def plain_loss(model, rows):
        return ((model(tx[rows])-ty[rows])**2).mean()
    def fit(model, loss_fn):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best, best_loss, stale = None, float("inf"), 0
        for epoch in range(epochs):
            optimizer.zero_grad(); loss_fn(model, ti).backward(); optimizer.step()
            if epoch % 10: continue
            with torch.no_grad(): value = float(loss_fn(model, vi))
            if value < best_loss - 1e-7:
                best, best_loss, stale = copy.deepcopy(model.state_dict()), value, 0
            else: stale += 1
            if stale >= 25: break
        model.load_state_dict(best)
    fit(hybrid, hybrid_loss); fit(multi, multi_loss); fit(plain, plain_loss)
    with torch.no_grad():
        hp, hl, _, _ = hybrid(tx); mp, ml = multi(tx); pp = plain(tx)
    predictions = {"hybrid": hp.numpy()*ts+tm, "multitask": mp.numpy()*ts+tm,
                   "non_mixture": pp.numpy()*ts+tm}
    result = {"version": "ipwm_fault_hybrid_counterfactual_gate_v1", "seed": seed,
      "split": {"train": int(train.sum()), "validation": int(validation.sum()), "test": int(test.sum()),
                "grouped_by_prefix": True},
      "parameters": {"hybrid": count(hybrid), "multitask": count(multi), "non_mixture": count(plain),
                     "relative_differences": differences}, "methods": {}}
    for name, prediction in predictions.items():
        result["methods"][name] = {"prediction": response_metrics(target, prediction, robots, test),
                                    "action": action_metrics(target, prediction, data, test)}
    result["methods"]["hybrid"]["mode"] = mode_metrics(label, torch.sigmoid(hl).numpy(), test)
    result["methods"]["multitask"]["mode"] = mode_metrics(label, torch.sigmoid(ml).numpy(), test)
    strongest_rmse = min(result["methods"][n]["prediction"]["pooled"] for n in ("multitask","non_mixture"))
    strongest_spearman = max(result["methods"][n]["action"]["mean_spearman"] for n in ("multitask","non_mixture"))
    best_regret = min(result["methods"][n]["action"]["normalized_top1_regret"] for n in ("multitask","non_mixture"))
    hm = result["methods"]["hybrid"]
    result["gate_terms"] = {
      "rmse_improvement": (strongest_rmse-hm["prediction"]["pooled"])/strongest_rmse,
      "spearman_improvement": hm["action"]["mean_spearman"]-strongest_spearman,
      "lower_regret": hm["action"]["normalized_top1_regret"] < best_regret,
      "both_robots_improve": all(hm["prediction"][r] < min(result["methods"][n]["prediction"][r] for n in ("multitask","non_mixture")) for r in ("genkiarm","panda")),
      "mode_balanced_accuracy": hm["mode"]["balanced_accuracy"], "mode_brier": hm["mode"]["brier_score"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    result = run(args.dataset, args.seed); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
