"""Train the preregistered contact action ranker on carrier-policy branches."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from robotarm.models.contact_action_ranker import ContactActionRanker, pairwise_ranking_loss
from scripts.diagnose_ipwm_action_ranking import spearman


SCALARS = (
    "carrier_predicted_cost_m", "selective_predicted_cost_m", "predicted_cost_delta_m",
    "action_deviation_rms", "first_action_deviation_l2", "action_effort_rms",
    "tool_block_distance_m", "goal_distance_m", "block_speed_mps", "nominal_action_l2",
)


def load_rows(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))["candidate_rows"]


def vector(row: dict) -> np.ndarray:
    return np.asarray(
        [*row["state"], *row["carrier_sequence"], *row["candidate_delta"],
         *(float(row[key]) for key in SCALARS)], dtype=np.float32,
    )


def groups(rows: list[dict]):
    result: dict[str, list[dict]] = {}
    for row in rows:
        result.setdefault(row["target"], []).append(row)
    return result


def tensor_groups(rows, mean, scale, device):
    output = []
    for target, items in groups(rows).items():
        x = np.stack([vector(item) for item in items])
        y = np.asarray([item["true_cost_m"] for item in items], dtype=np.float32) * 1000.0
        output.append((
            target,
            torch.as_tensor((x - mean) / scale, device=device),
            torch.as_tensor(y, device=device),
            items,
        ))
    return output


@torch.no_grad()
def evaluate(model, data):
    metrics = []
    for target, x, cost, items in data:
        score = model(x).cpu().numpy()
        actual = cost.cpu().numpy()
        chosen, oracle = int(np.argmin(score)), int(np.argmin(actual))
        baseline = next(i for i, item in enumerate(items) if int(item["candidate_index"]) == 0)
        metrics.append({
            "target": target, "spearman": spearman(score, actual),
            "chosen_index": int(items[chosen]["candidate_index"]),
            "oracle_index": int(items[oracle]["candidate_index"]),
            "chosen_cost_m": float(actual[chosen] / 1000.0),
            "baseline_cost_m": float(actual[baseline] / 1000.0),
            "oracle_cost_m": float(actual[oracle] / 1000.0),
            "improvement_over_baseline_pct": float(
                100.0 * (actual[baseline] - actual[chosen]) / max(actual[baseline], 1e-8)
            ),
            "regret_m": float((actual[chosen] - actual[oracle]) / 1000.0),
        })
    return metrics


def aggregate(metrics):
    return {
        "mean_spearman": float(np.mean([row["spearman"] for row in metrics])),
        "mean_improvement_over_baseline_pct": float(np.mean([
            row["improvement_over_baseline_pct"] for row in metrics
        ])),
        "positive_fraction": float(np.mean([
            row["improvement_over_baseline_pct"] > 0 for row in metrics
        ])),
        "mean_regret_m": float(np.mean([row["regret_m"] for row in metrics])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows, validation_rows = load_rows(args.train), load_rows(args.validation)
    train_x = np.stack([vector(row) for row in train_rows])
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train = tensor_groups(train_rows, mean, scale, device)
    validation = tensor_groups(validation_rows, mean, scale, device)
    model = ContactActionRanker(train_x.shape[1], (64, 32)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        losses = []
        for _, x, cost, items in train:
            score = model(x)
            rank_loss = pairwise_ranking_loss(score, cost)
            centered_cost = cost - cost.mean()
            regression = torch.nn.functional.mse_loss(score - score.mean(), centered_cost)
            baseline = next(i for i, item in enumerate(items) if int(item["candidate_index"]) == 0)
            worse = cost > cost[baseline] + 1e-6
            conservative = (
                torch.nn.functional.softplus(score[baseline] - score[worse]).mean()
                if worse.any() else score.sum() * 0.0
            )
            losses.append(rank_loss + 0.25 * regression + 0.10 * conservative)
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_metrics = evaluate(model, validation)
            val_aggregate = aggregate(val_metrics)
            record = {"epoch": epoch, "loss": float(loss.detach().cpu()), **val_aggregate}
            history.append(record)
            key = (
                val_aggregate["mean_spearman"],
                val_aggregate["positive_fraction"],
                val_aggregate["mean_improvement_over_baseline_pct"],
                -val_aggregate["mean_regret_m"],
            )
            if best is None or key > best[0]:
                best = (key, epoch, copy.deepcopy(model.state_dict()))
    assert best is not None
    model.load_state_dict(best[2])
    model.eval()
    train_metrics, val_metrics = evaluate(model, train), evaluate(model, validation)
    payload = {
        "version": "contact_action_ranker_v1", "development_only": True,
        "seed": args.seed, "input_dim": int(train_x.shape[1]),
        "hidden_dims": [64, 32], "scalars": list(SCALARS),
        "best_epoch": best[1], "normalization_mean": mean.tolist(),
        "normalization_scale": scale.tolist(),
        "training_metrics": train_metrics, "training_aggregate": aggregate(train_metrics),
        "validation_metrics": val_metrics, "validation_aggregate": aggregate(val_metrics),
        "history": history,
    }
    if args.test is not None:
        test = tensor_groups(load_rows(args.test), mean, scale, device)
        test_metrics = evaluate(model, test)
        payload["test_metrics"] = test_metrics
        payload["test_aggregate"] = aggregate(test_metrics)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "input_dim": train_x.shape[1],
        "hidden_dims": (64, 32), "normalization_mean": torch.as_tensor(mean),
        "normalization_scale": torch.as_tensor(scale), "scalars": list(SCALARS),
    }, args.output_model)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_epoch": best[1], "training": payload["training_aggregate"],
        "validation": payload["validation_aggregate"],
        **({"test": payload["test_aggregate"]} if "test_aggregate" in payload else {}),
    }, indent=2))


if __name__ == "__main__":
    main()
