"""Small preregistered cross-arm held-out-lock prediction Gate.

One shared graph parameter set is trained jointly on full 5-DoF GenkiArm and
full 7-DoF Panda transitions.  For each arm the middle lock is held out.  A
flat padded MLP receives exactly the same state/action/lock/geometry inputs and
is parameter matched.  This Gate covers robot free-joint transition only; it
does not establish Push/Grasp or object/contact prediction.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from robotarm.envs.damage import DamageConfig
from robotarm.envs.variable_mujoco_env import VariableMujocoArmEnv
from robotarm.models.variable_dof_ipwm import SerialChainSpec, VariableDofInterventionCore


ROOT = Path(__file__).resolve().parent.parent
MAX_DOF = 7


def robot_definition(name: str):
    if name == "genkiarm":
        joints = tuple(f"j{i}" for i in range(1, 6))
        return {
            "xml": ROOT / "sim/assets/genkiarm_push.xml", "joints": joints,
            "actuators": tuple(f"m{i}" for i in range(1, 6)),
            "object_body": "block", "object_geom": "block_geom", "home": None,
            "train_locks": (1, 3), "test_lock": 2,
        }
    joints = tuple(f"joint{i}" for i in range(1, 8))
    return {
        "xml": ROOT / "sim/assets/panda_push_grasp.xml", "joints": joints,
        "actuators": tuple(f"actuator{i}" for i in range(1, 8)),
        "object_body": "task_cube", "object_geom": "cube_geom", "home": "task_home",
        "train_locks": (1, 5), "test_lock": 3,
    }


def collect(seed: int, *, episodes: int, steps: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, np.ndarray | int | str]] = []
    for robot in ("genkiarm", "panda"):
        cfg = robot_definition(robot)
        env = VariableMujocoArmEnv(
            cfg["xml"], joint_names=cfg["joints"], actuator_names=cfg["actuators"],
            object_body=cfg["object_body"], object_geom=cfg["object_geom"],
            home_keyframe=cfg["home"],
        )
        spec = SerialChainSpec.from_mjcf(cfg["xml"], cfg["joints"], name=robot)
        home_state, _, _ = env.reset()
        conditions = [("intact", None), *[("train", x) for x in cfg["train_locks"]],
                      ("test", cfg["test_lock"])]
        for split, lock in conditions:
            if lock is None:
                damage = DamageConfig.intact(env.dof)
            else:
                joint_id = env.joint_ids[lock]
                low, high = env.model.jnt_range[joint_id]
                offset = 0.18 if (lock % 2 == 0) else -0.18
                angle = float(np.clip(home_state[lock, 0] + offset, low + 0.03, high - 0.03))
                damage = DamageConfig.lock_single(lock, angle, dof=env.dof)
            for episode in range(episodes):
                before = env.reset(damage)[0]
                action = np.zeros(env.dof)
                for _ in range(steps):
                    excitation = rng.uniform(-0.55, 0.55, env.dof)
                    action = np.clip(0.65 * action + 0.35 * excitation, -0.7, 0.7)
                    after = env.step(action)[0]
                    rows.append({
                        "robot": robot, "split": split, "episode": episode,
                        "dof": env.dof, "state": before.copy(),
                        "action": env.last_applied_action.copy(), "target": after.copy(),
                        "mask": damage.joint_mask.copy(), "angle": damage.lock_angle.copy(),
                        "axes": spec.axes.copy(), "origins": spec.origins.copy(),
                    })
                    before = after
    keys = ("state", "action", "target", "mask", "angle", "axes", "origins")
    result: dict[str, list] = {key: [] for key in keys}
    result.update({"robot": [], "split": [], "episode": [], "dof": []})
    for row in rows:
        dof = int(row["dof"])
        for key in ("state", "target"):
            value = np.zeros((MAX_DOF, 2), dtype=np.float32)
            value[:dof] = row[key]
            result[key].append(value)
        for key in ("action", "mask", "angle"):
            value = np.zeros(MAX_DOF, dtype=np.float32)
            value[:dof] = row[key]
            result[key].append(value)
        for key in ("axes", "origins"):
            value = np.zeros((MAX_DOF, 3), dtype=np.float32)
            value[:dof] = row[key]
            result[key].append(value)
        for key in ("robot", "split", "episode", "dof"):
            result[key].append(row[key])
    return {key: np.asarray(value) for key, value in result.items()}


def flat_features(data: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.arange(MAX_DOF)[None] < data["dof"][:, None]
    node = np.concatenate([
        data["state"], data["action"][..., None], data["mask"][..., None],
        data["angle"][..., None], data["axes"], data["origins"], valid[..., None],
    ], axis=-1)
    return node.reshape(len(valid), -1)


class FlatMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
            nn.SiLU(), nn.Linear(hidden, MAX_DOF * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, MAX_DOF, 2)


def count(model: nn.Module) -> int:
    return sum(x.numel() for x in model.parameters())


def metric(actual: np.ndarray, predicted: np.ndarray, data, select: np.ndarray):
    result = {}
    for robot in ("genkiarm", "panda"):
        rows = select & (data["robot"] == robot)
        valid = np.arange(MAX_DOF)[None] < data["dof"][rows, None]
        squared = (actual[rows] - predicted[rows]) ** 2
        result[robot] = float(np.sqrt(squared[valid].mean()))
    rows = np.flatnonzero(select)
    values = []
    for row in rows:
        values.append((actual[row, : data["dof"][row]] - predicted[row, : data["dof"][row]]) ** 2)
    result["pooled"] = float(np.sqrt(np.concatenate([x.reshape(-1) for x in values]).mean()))
    return result


def run(seed: int, *, episodes: int, steps: int, epochs: int, batch_size: int) -> dict[str, object]:
    torch.manual_seed(seed)
    data = collect(seed * 1009, episodes=episodes, steps=steps)
    episode_cut = max(1, int(episodes * 0.7))
    train = (data["split"] != "test") & (data["episode"] < episode_cut)
    validation = (data["split"] != "test") & (data["episode"] >= episode_cut)
    test = (data["split"] == "test") & (data["episode"] >= episode_cut)
    valid = np.arange(MAX_DOF)[None] < data["dof"][:, None]
    tensors = {key: torch.as_tensor(data[key], dtype=torch.float32) for key in
               ("state", "action", "target", "mask", "angle", "axes", "origins")}
    valid_t = torch.as_tensor(valid)
    xflat = torch.as_tensor(flat_features(data), dtype=torch.float32)

    structured = VariableDofInterventionCore(hidden_dim=64)
    # h=160 keeps parameter counts within five percent of the graph core.
    flat = FlatMLP(xflat.shape[1], hidden=160)
    relative_parameters = abs(count(structured) - count(flat)) / count(structured)
    if relative_parameters > 0.05:
        raise RuntimeError("baseline parameter match exceeds five percent")

    def graph_predict(model, indices=None):
        choose = slice(None) if indices is None else indices
        return model(
            tensors["state"][choose], tensors["action"][choose],
            tensors["mask"][choose], tensors["angle"][choose],
            valid_t[choose], tensors["axes"][choose], tensors["origins"][choose],
        )

    def flat_predict(model, indices=None):
        choose = slice(None) if indices is None else indices
        delta = model(xflat[choose])
        projected = tensors["state"][choose].clone()
        locked = tensors["mask"][choose].bool()
        projected[..., 0] = torch.where(
            locked, tensors["angle"][choose], projected[..., 0]
        )
        projected[..., 1] = torch.where(
            locked, torch.zeros_like(projected[..., 1]), projected[..., 1]
        )
        projected = projected + delta * (~locked).unsqueeze(-1)
        projected[..., 0] = torch.where(
            locked, tensors["angle"][choose], projected[..., 0]
        )
        projected[..., 1] = torch.where(locked, torch.zeros_like(projected[..., 1]), projected[..., 1])
        return projected * valid_t[choose].unsqueeze(-1)

    def fit(model, predictor):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        best, best_loss, stale = None, float("inf"), 0
        train_indices = np.flatnonzero(train)
        validation_indices = torch.as_tensor(np.flatnonzero(validation), dtype=torch.long)
        generator = np.random.default_rng(seed + count(model))
        for epoch in range(epochs):
            chosen = torch.as_tensor(
                generator.choice(train_indices, size=min(batch_size, len(train_indices)), replace=False),
                dtype=torch.long,
            )
            optimizer.zero_grad()
            prediction = predictor(model, chosen)
            mask = valid_t[chosen, :, None]
            loss = ((prediction - tensors["target"][chosen]) ** 2)[mask.expand_as(prediction)].mean()
            loss.backward(); optimizer.step()
            if epoch % 10 != 0:
                continue
            with torch.no_grad():
                prediction = predictor(model, validation_indices)
                mask = valid_t[validation_indices, :, None]
                val = float(((prediction - tensors["target"][validation_indices]) ** 2)[mask.expand_as(prediction)].mean())
            if val < best_loss - 1e-8:
                best_loss, stale = val, 0
                best = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= 20:
                break
        model.load_state_dict(best)
        return model

    structured = fit(structured, graph_predict)
    flat = fit(flat, flat_predict)
    with torch.no_grad():
        graph_np = graph_predict(structured).numpy()
        flat_np = flat_predict(flat).numpy()
    actual = data["target"]
    inertial = data["state"].copy()
    inertial[..., 0] += 0.005 * inertial[..., 1]
    locked = data["mask"].astype(bool)
    inertial[..., 0] = np.where(locked, data["angle"], inertial[..., 0])
    inertial[..., 1] = np.where(locked, 0.0, inertial[..., 1])
    graph_metric = metric(actual, graph_np, data, test)
    flat_metric = metric(actual, flat_np, data, test)
    inertial_metric = metric(actual, inertial, data, test)
    return {
        "version": "cross_arm_prediction_gate_v1", "seed": seed,
        "episodes_per_condition": episodes, "steps": steps, "epochs_max": epochs,
        "batch_size": batch_size,
        "split": {"train_rows": int(train.sum()), "validation_rows": int(validation.sum()),
                  "test_rows": int(test.sum()), "heldout_locks": {"genkiarm": "j3", "panda": "joint4"}},
        "parameter_counts": {"structured": count(structured), "flat": count(flat),
                             "relative_difference": relative_parameters},
        "analytic_inertial": inertial_metric, "flat_unstructured": flat_metric,
        "shared_structured": graph_metric,
        "structured_relative_pooled_improvement": (
            flat_metric["pooled"] - graph_metric["pooled"]
        ) / flat_metric["pooled"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.seed, episodes=args.episodes, steps=args.steps,
        epochs=args.epochs, batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
