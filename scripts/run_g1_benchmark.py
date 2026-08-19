"""Run the held-out topology-residual G1 mechanism benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.g1_mechanism import (
    evaluate_test_domain,
    train_mechanism_models,
)
from robotarm.training.sim_data import collect_controller_domains, collect_domains
from robotarm.training.sim_protocol import build_g1_protocol, load_g1_protocol
from robotarm.training.target_split import load_target_split


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def aggregate_rows(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), int(row["shots"]))].append(row)
    aggregates = []
    for (model, shots), values in sorted(grouped.items()):
        nll = np.asarray([float(value["eval_nll"]) for value in values])
        rmse = np.asarray([float(value["eval_rmse"]) for value in values])
        adaptation = np.asarray(
            [float(value["adaptation_seconds"]) for value in values]
        )
        aggregates.append(
            {
                "model": model,
                "shots": shots,
                "n": len(values),
                "eval_nll_mean": float(nll.mean()),
                "eval_nll_std": float(nll.std()),
                "eval_rmse_mean": float(rmse.mean()),
                "eval_rmse_std": float(rmse.std()),
                "adaptation_seconds_mean": float(adaptation.mean()),
            }
        )
    return aggregates


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(path: Path, aggregates: list[dict[str, object]]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return _plot_with_pillow(path, aggregates)
    else:
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        for model in (
            "topology_only",
            "history_encoder",
            "parameter_matched",
            "residual_only",
            "monolithic_matched",
            "dfwm",
        ):
            values = [row for row in aggregates if row["model"] == model]
            axis.errorbar(
                [int(row["shots"]) for row in values],
                [float(row["eval_rmse_mean"]) for row in values],
                yerr=[float(row["eval_rmse_std"]) for row in values],
                marker="o",
                capsize=3,
                label=model,
            )
        axis.set_xlabel("Calibration trajectories")
        axis.set_ylabel("Held-out one-step state RMSE")
        axis.set_xticks([0, 1, 2, 5])
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)
    return True


def _plot_with_pillow(path: Path, aggregates: list[dict[str, object]]) -> bool:
    """Dependency-light PNG fallback for experiment machines without matplotlib."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    width, height = 960, 620
    left, right, top, bottom = 110, 40, 50, 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [
        float(row["eval_rmse_mean"]) + float(row["eval_rmse_std"])
        for row in aggregates
    ]
    all_lows = [
        float(row["eval_rmse_mean"]) - float(row["eval_rmse_std"])
        for row in aggregates
    ]
    y_min = max(0.0, min(all_lows) * 0.95)
    y_max = max(all_values) * 1.05
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_pos(shot: int) -> float:
        return left + (shot / 5.0) * plot_width

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    draw.line((left, top, left, top + plot_height), fill="#222222", width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill="#222222",
        width=2,
    )
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_pos(value)
        draw.line((left, y, left + plot_width, y), fill="#e2e2e2", width=1)
        draw.text((10, y - 8), f"{value:.3f}", fill="#333333")
    for shot in (0, 1, 2, 5):
        x = x_pos(shot)
        draw.line((x, top + plot_height, x, top + plot_height + 6), fill="#222222")
        draw.text((x - 4, top + plot_height + 12), str(shot), fill="#333333")

    colors = {
        "topology_only": "#3b6fb6",
        "history_encoder": "#8e44ad",
        "parameter_matched": "#e67e22",
        "residual_only": "#7a7a7a",
        "monolithic_matched": "#2f8f5b",
        "dfwm": "#d1493f",
    }
    for model in (
        "topology_only",
        "history_encoder",
        "parameter_matched",
        "residual_only",
        "monolithic_matched",
        "dfwm",
    ):
        values = [row for row in aggregates if row["model"] == model]
        points = [
            (x_pos(int(row["shots"])), y_pos(float(row["eval_rmse_mean"])))
            for row in values
        ]
        draw.line(points, fill=colors[model], width=4)
        for row, (x, y) in zip(values, points):
            std = float(row["eval_rmse_std"])
            high = y_pos(float(row["eval_rmse_mean"]) + std)
            low = y_pos(float(row["eval_rmse_mean"]) - std)
            draw.line((x, high, x, low), fill=colors[model], width=2)
            draw.line((x - 5, high, x + 5, high), fill=colors[model], width=2)
            draw.line((x - 5, low, x + 5, low), fill=colors[model], width=2)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=colors[model])

    draw.text((width // 2 - 110, 15), "G1 held-out few-shot prediction", fill="#111111")
    draw.text(
        (width // 2 - 70, height - 34),
        "Calibration trajectories",
        fill="#222222",
    )
    draw.text((left + 10, top + 10), "State RMSE", fill="#222222")
    legend_x = width - 260
    for index, model in enumerate(
        (
            "topology_only",
            "history_encoder",
            "parameter_matched",
            "residual_only",
            "monolithic_matched",
            "dfwm",
        )
    ):
        y = top + 18 + index * 28
        draw.line((legend_x, y, legend_x + 28, y), fill=colors[model], width=4)
        draw.text((legend_x + 38, y - 7), model, fill="#222222")
    image.save(path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/g1_benchmark"))
    parser.add_argument("--seeds", type=parse_seeds, default=(7,))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--train-trajectories", type=int, default=2)
    parser.add_argument("--calibration-trajectories", type=int, default=5)
    parser.add_argument("--evaluation-trajectories", type=int, default=3)
    parser.add_argument("--trajectory-steps", type=int, default=30)
    parser.add_argument("--latent-steps", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--data-policy", choices=["random", "controller"], default="controller"
    )
    parser.add_argument(
        "--split", type=Path, default=None,
        help="Optional G1 split YAML path (default: config/splits/g1_5dof_v1.yaml)",
    )
    args = parser.parse_args()

    if args.calibration_trajectories < 5:
        parser.error("--calibration-trajectories must be at least 5")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    protocol = load_g1_protocol(args.split) if args.split else build_g1_protocol()
    target_split = load_target_split()
    calibration_targets = tuple(
        target.as_array() for target in target_split.calibration
    )
    evaluation_targets = tuple(
        target.as_array() for target in target_split.evaluation
    )
    joint_ranges = MujocoArmEnv().joint_ranges
    run_dir = args.out / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, float | int | str]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    started_at = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    collect = collect_controller_domains if args.data_policy == "controller" else collect_domains

    for seed in args.seeds:
        print(f"seed={seed} start", flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_data = collect(
            protocol.train,
            trajectories_per_domain=args.train_trajectories,
            steps=args.trajectory_steps,
            seed=seed * 10_000,
            targets=calibration_targets,
        )
        validation_data = collect(
            protocol.validation,
            trajectories_per_domain=args.train_trajectories,
            steps=args.trajectory_steps,
            seed=seed * 10_000 + 1,
            targets=calibration_targets,
        )
        print(f"seed={seed} data_collected", flush=True)
        models = train_mechanism_models(
            protocol.train,
            train_data,
            joint_ranges,
            epochs=args.epochs,
            device=device,
            validation_domains=protocol.validation,
            validation_trajectories=validation_data,
        )
        print(f"seed={seed} models_trained", flush=True)
        histories[str(seed)] = models.history

        for domain_index, domain in enumerate(protocol.test):
            print(f"seed={seed} domain={domain.domain_id} start", flush=True)
            calibration = collect(
                (domain,),
                trajectories_per_domain=args.calibration_trajectories,
                steps=args.trajectory_steps,
                seed=seed * 100_000 + domain_index * 1000,
                targets=calibration_targets,
            )
            evaluation = collect(
                (domain,),
                trajectories_per_domain=args.evaluation_trajectories,
                steps=args.trajectory_steps,
                seed=seed * 100_000 + domain_index * 1000 + 500,
                targets=evaluation_targets,
            )
            domain_rows = evaluate_test_domain(
                models,
                domain,
                calibration,
                evaluation,
                joint_ranges,
                shots=protocol.calibration_shots,
                latent_steps=args.latent_steps,
                device=device,
            )
            for row in domain_rows:
                row["seed"] = seed
                rows.append(row)
            print(f"seed={seed} domain={domain.domain_id} complete", flush=True)
            checkpoint = run_dir / f"seed_{seed}_checkpoint.csv"
            checkpoint_fields = sorted({key for item in rows for key in item})
            with checkpoint.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=checkpoint_fields)
                writer.writeheader()
                writer.writerows(rows)

    aggregates = aggregate_rows(rows)
    elapsed_seconds = time.perf_counter() - started_at
    write_csv(run_dir / "few_shot_results.csv", rows)
    write_csv(run_dir / "aggregate.csv", aggregates)
    plot_written = maybe_plot(run_dir / "few_shot_rmse.png", aggregates)
    summary = {
        "status": "g1_mechanism_benchmark_complete",
        "device": str(device),
        "seeds": list(args.seeds),
        "protocol": {
            "version": protocol.version,
            "sha256": protocol.sha256,
            "source": str(protocol.source_path),
            "train": [domain.domain_id for domain in protocol.train],
            "validation": [domain.domain_id for domain in protocol.validation],
            "test": [domain.domain_id for domain in protocol.test],
            "calibration_evaluation_disjoint": True,
            "validation_usage": "reserved; fixed-hyperparameter benchmark did not select on validation",
        },
        "target_split": {
            "version": target_split.version,
            "status": target_split.status,
            "sha256": target_split.sha256,
            "source": str(target_split.source_path),
            "calibration_ids": [
                target.target_id for target in target_split.calibration
            ],
            "evaluation_ids": [
                target.target_id for target in target_split.evaluation
            ],
        },
        "settings": {
            "epochs": args.epochs,
            "train_trajectories": args.train_trajectories,
            "trajectory_steps": args.trajectory_steps,
            "latent_steps": args.latent_steps,
            "data_policy": args.data_policy,
        },
        "compute": {
            "wall_clock_seconds": elapsed_seconds,
            "gpu_hours": elapsed_seconds / 3600.0 if device.type == "cuda" else 0.0,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "parameter_counts": {
            "topology_only": sum(
                parameter.numel()
                for module in (
                    models.topology_encoder,
                    models.topology_world_model,
                )
                for parameter in module.parameters()
            ),
            "residual_only": sum(
                parameter.numel()
                for parameter in models.residual_only_world_model.parameters()
            ),
            "monolithic_matched": sum(
                parameter.numel()
                for module in (
                    models.monolithic_encoder,
                    models.monolithic_world_model,
                    models.monolithic_projection,
                )
                for parameter in module.parameters()
            ),
            "dfwm": sum(
                parameter.numel()
                for module in (
                    models.dfwm_encoder,
                    models.dfwm_world_model,
                )
                for parameter in module.parameters()
            ),
            "deployment_trainable_parameters": {
                "topology_only": 0,
                "residual_only": 8,
                "monolithic_matched": 8,
                "dfwm": 8,
            },
        },
        "aggregate": aggregates,
        "plot_written": plot_written,
        "claim_scope": "simulation-only G1 prediction evidence after G0",
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "training_history.json").write_text(
        json.dumps(histories, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"run_dir={run_dir.resolve()}")


if __name__ == "__main__":
    main()
