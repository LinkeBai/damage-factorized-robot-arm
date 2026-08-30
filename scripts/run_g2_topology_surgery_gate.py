"""Gate A for a topology-surgery world model on held-out joint locks."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from robotarm.envs.mujoco_env import MujocoArmEnv
from robotarm.training.sim_protocol import load_g1_protocol
from robotarm.training.target_split import load_target_split
from robotarm.training.topology_surgery_gate import (
    evaluate_graph_surgery_model,
    evaluate_surgery_gate_model,
    surgery_gate_parameter_count,
    train_graph_surgery_model,
    train_gated_reaction_model,
    train_reduced_coordinate_graph_model,
    train_constraint_reaction_model,
    train_unconstrained_residual_model,
    train_surgery_gate_model,
)
from scripts.run_push_benchmark import PUSH_XML, collect_push_domains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/experiment/g2_topology_graph_gate_b_v1.yaml"),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed not in config["seeds"]:
        raise ValueError(f"seed {args.seed} not in frozen seed list")
    epochs = args.epochs or int(config["epochs"])
    steps = args.steps or int(config["steps"])
    output_dir = args.output_dir or Path("runs/g2_topology_surgery_gate") / f"seed{args.seed}_v1"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seed={args.seed} epochs={epochs}", flush=True)

    protocol = load_g1_protocol(Path(config["protocol"]))
    train_topologies = {domain.domain_id.split("__", 1)[0] for domain in protocol.train}
    if "D3" in train_topologies:
        raise RuntimeError("Gate integrity failure: D3 must be absent from training")
    targets = load_target_split(Path(config["targets"]))
    calibration = tuple(item.as_array() for item in targets.calibration)
    evaluation = tuple(item.as_array() for item in targets.evaluation)
    block_initial_xy = np.asarray(config["block_initial_xy"], dtype=float)
    joint_ranges = MujocoArmEnv(xml_path=PUSH_XML).joint_ranges

    print("[data] collecting shared corrected-protocol training trajectories", flush=True)
    train_trajectories = collect_push_domains(
        protocol.train,
        trajectories_per_domain=int(config["trajectories_per_train_domain"]),
        steps=steps,
        seed=args.seed * 10_000,
        targets=calibration,
        excitation="goal",
        block_initial_xy=block_initial_xy,
    )

    models, train_seconds = {}, {}
    for method in config["methods"]:
        print(f"[train] {method}", flush=True)
        started = time.perf_counter()
        if method == "unconstrained_residual":
            models[method] = train_unconstrained_residual_model(
                train_trajectories, base_epochs=epochs,
                residual_epochs=int(config.get("reaction_epochs", epochs)),
                device=device, seed=args.seed,
                hidden_dim=int(config.get("adapter_base_hidden_dim", 96)),
            )
        elif method == "gated_reaction":
            models[method] = train_gated_reaction_model(
                train_trajectories, base_epochs=epochs,
                reaction_epochs=int(config.get("reaction_epochs", epochs)),
                device=device, seed=args.seed,
                hidden_dim=int(config.get("adapter_base_hidden_dim", 128)),
                bottleneck_dim=int(config.get("reaction_bottleneck_dim", 16)),
                gate_logit_init=float(config.get("gate_logit_init", -4.0)),
            )
        elif method == "constraint_reaction":
            models[method] = train_constraint_reaction_model(
                train_trajectories, base_epochs=epochs,
                reaction_epochs=int(config.get("reaction_epochs", epochs)),
                device=device, seed=args.seed,
                hidden_dim=int(config.get("adapter_base_hidden_dim", 96)),
            )
        elif method == "reduced_coordinate_graph":
            models[method] = train_reduced_coordinate_graph_model(
                train_trajectories, epochs=epochs, device=device, seed=args.seed,
                hidden_dim=int(config.get("matched_hidden_dim", 128)),
            )
        elif method in ("graph_ordinary", "graph_ordinary_matched", "graph_matched_projected", "graph_topology_surgery"):
            models[method] = train_graph_surgery_model(
                train_trajectories, epochs=epochs, device=device, seed=args.seed,
                use_topology=method == "graph_topology_surgery",
                hidden_dim=(
                    int(config.get("matched_hidden_dim", 128))
                    if method in ("graph_ordinary_matched", "graph_matched_projected")
                    else 96
                ),
            )
        else:
            models[method] = train_surgery_gate_model(
                train_trajectories, joint_ranges, method=method, epochs=epochs,
                device=device, seed=args.seed,
            )
        train_seconds[method] = time.perf_counter() - started
        print(f"  done in {train_seconds[method]:.1f}s", flush=True)

    rows = []
    for index, domain in enumerate(protocol.test):
        print(f"[eval] {domain.domain_id}", flush=True)
        trajectories = collect_push_domains(
            (domain,),
            trajectories_per_domain=int(config["trajectories_per_test_domain"]),
            steps=steps,
            seed=args.seed * 100_000 + index * 1000 + 500,
            targets=evaluation,
            excitation="goal",
            block_initial_xy=block_initial_xy,
        )
        for method in config["methods"]:
            if method in ("constraint_reaction", "gated_reaction", "unconstrained_residual", "reduced_coordinate_graph"):
                metrics = evaluate_graph_surgery_model(
                    models[method], domain, trajectories, device=device,
                    horizon=int(config["rollout_horizon"]),
                    use_topology=method in ("constraint_reaction", "gated_reaction", "reduced_coordinate_graph"),
                )
            elif method in ("graph_ordinary", "graph_ordinary_matched", "graph_matched_projected", "graph_topology_surgery"):
                metrics = evaluate_graph_surgery_model(
                    models[method], domain, trajectories, device=device,
                    horizon=int(config["rollout_horizon"]),
                    use_topology=method in ("graph_topology_surgery", "graph_matched_projected"),
                )
            else:
                metrics = evaluate_surgery_gate_model(
                    models[method], domain, trajectories, joint_ranges,
                    device=device, horizon=int(config["rollout_horizon"]),
                )
            row = {"domain": domain.domain_id, "seed": args.seed, "method": method, **metrics}
            rows.append(row)
            print(
                f"  {method}: total={metrics['overall_rmse']:.4f} "
                f"free={metrics['free_arm_rmse']:.4f} object={metrics['object_rmse']:.4f} "
                f"violation={metrics['constraint_violation_rms']:.6f}",
                flush=True,
            )

    primary = config["primary_domain"]
    primary_rows = {row["method"]: row for row in rows if row["domain"] == primary}
    if "reduced_coordinate_graph" in primary_rows:
        base = primary_rows["graph_ordinary_matched"]
        reduced = primary_rows["reduced_coordinate_graph"]
        object_regression = 100.0 * (reduced["object_rmse"] - base["object_rmse"]) / max(base["object_rmse"], 1e-12)
        free_regression = 100.0 * (reduced["free_arm_rmse"] - base["free_arm_rmse"]) / max(base["free_arm_rmse"], 1e-12)
        gate = config["gate"]
        passed = (
            reduced["constraint_violation_rms"] <= float(gate["maximum_constraint_violation_rms"])
            and object_regression <= float(gate["maximum_object_rmse_regression_pct"])
            and free_regression <= float(gate["maximum_free_arm_rmse_regression_pct"])
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "config_version": config["version"], "seed": args.seed, "device": str(device),
            "epochs": epochs, "steps": steps, "primary_domain": primary,
            "parameters": {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
            "train_seconds": train_seconds,
            "constraint_violation_rms": reduced["constraint_violation_rms"],
            "object_regression_pct": object_regression,
            "free_regression_pct": free_regression,
            "gate_passed": passed, "rows": rows,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"[gate-i] violation={reduced['constraint_violation_rms']:.8f} "
            f"object_regression={object_regression:+.2f}% free_regression={free_regression:+.2f}% "
            f"decision={'PASS' if passed else 'NO-GO'}", flush=True,
        )
        return
    if "gated_reaction" in primary_rows:
        base = primary_rows["graph_ordinary_matched"]
        gated = primary_rows["gated_reaction"]
        object_regression = 100.0 * (gated["object_rmse"] - base["object_rmse"]) / max(base["object_rmse"], 1e-12)
        free_regression = 100.0 * (gated["free_arm_rmse"] - base["free_arm_rmse"]) / max(base["free_arm_rmse"], 1e-12)
        gate = config["gate"]
        passed = (
            gated["constraint_violation_rms"] <= float(gate["maximum_constraint_violation_rms"])
            and object_regression <= float(gate["maximum_object_rmse_regression_pct"])
            and free_regression <= float(gate["maximum_free_arm_rmse_regression_pct"])
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "config_version": config["version"], "seed": args.seed, "device": str(device),
            "epochs": epochs, "steps": steps, "primary_domain": primary,
            "parameters": {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
            "trainable_parameters": {name: sum(p.numel() for p in model.parameters() if p.requires_grad) for name, model in models.items()},
            "train_seconds": train_seconds,
            "constraint_violation_rms": gated["constraint_violation_rms"],
            "object_regression_pct": object_regression,
            "free_regression_pct": free_regression,
            "gate_passed": passed, "rows": rows,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"[gate-h] violation={gated['constraint_violation_rms']:.8f} "
            f"object_regression={object_regression:+.2f}% free_regression={free_regression:+.2f}% "
            f"decision={'PASS' if passed else 'NO-GO'}", flush=True,
        )
        return
    if "graph_matched_projected" in primary_rows:
        base = primary_rows["graph_ordinary_matched"]
        projected = primary_rows["graph_matched_projected"]
        violation_reduction = 100.0 * (
            base["constraint_violation_rms"] - projected["constraint_violation_rms"]
        ) / max(base["constraint_violation_rms"], 1e-12)
        object_regression = 100.0 * (
            projected["object_rmse"] - base["object_rmse"]
        ) / max(base["object_rmse"], 1e-12)
        free_regression = 100.0 * (
            projected["free_arm_rmse"] - base["free_arm_rmse"]
        ) / max(base["free_arm_rmse"], 1e-12)
        gate = config["gate"]
        passed = (
            violation_reduction >= float(gate["minimum_violation_reduction_pct"])
            and object_regression <= float(gate["maximum_object_rmse_regression_pct"])
            and free_regression <= float(gate["maximum_free_arm_rmse_regression_pct"])
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "config_version": config["version"], "seed": args.seed, "device": str(device),
            "epochs": epochs, "steps": steps, "primary_domain": primary,
            "parameters": {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
            "train_seconds": train_seconds, "violation_reduction_pct": violation_reduction,
            "object_regression_pct": object_regression, "free_regression_pct": free_regression,
            "gate_passed": passed, "rows": rows,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"[gate] violation_reduction={violation_reduction:+.2f}% "
            f"object_regression={object_regression:+.2f}% free_regression={free_regression:+.2f}% "
            f"decision={'PASS' if passed else 'NO-GO'}", flush=True,
        )
        return
    ordinary = (
        primary_rows["graph_ordinary"]
        if "graph_ordinary" in primary_rows
        else primary_rows["ordinary"]
    )
    surgery = primary_rows.get(
        "constraint_reaction", primary_rows.get("graph_topology_surgery")
    )
    if surgery is None:
        raise KeyError("no topology-aware graph method in primary results")
    object_improvement = 100.0 * (
        ordinary["object_rmse"] - surgery["object_rmse"]
    ) / ordinary["object_rmse"]
    free_improvement = 100.0 * (
        ordinary["free_arm_rmse"] - surgery["free_arm_rmse"]
    ) / ordinary["free_arm_rmse"]
    gate = config["gate"]
    passed = (
        object_improvement >= float(gate["minimum_object_improvement_pct"])
        and free_improvement >= float(gate["minimum_free_arm_improvement_pct"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "config_version": config["version"], "seed": args.seed, "device": str(device),
        "epochs": epochs, "steps": steps, "train_topologies": sorted(train_topologies),
        "parameters": {
            name: (sum(p.numel() for p in model.parameters())
                   if name in ("graph_ordinary", "graph_ordinary_matched", "graph_matched_projected", "graph_topology_surgery", "constraint_reaction", "unconstrained_residual")
                   else surgery_gate_parameter_count(model))
            for name, model in models.items()
        },
        "train_seconds": train_seconds, "primary_domain": primary,
        "object_improvement_pct": object_improvement,
        "free_arm_improvement_pct": free_improvement,
        "gate_passed": passed, "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[gate] object={object_improvement:+.2f}% free={free_improvement:+.2f}% "
        f"decision={'PASS' if passed else 'NO-GO'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
