"""Closed-form information diagnostic for cross-arm active-contact histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_fault_hybrid_counterfactual_gate import action_metrics, response_metrics


HISTORY_KEYS = ("probe_joint_state", "probe_joint_delta", "probe_action",
                "probe_object_pose", "probe_object_twist", "probe_object_delta",
                "probe_contact")


def standardize(train_value, value):
    mean, std = train_value.mean(0, keepdims=True), train_value.std(0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (value - mean) / std, mean, std


def ridge_fit_predict(x, y, train, test, alpha=1.0):
    x_all, _, _ = standardize(x[train], x)
    y_all, ym, ys = standardize(y[train], y)
    design = np.concatenate([x_all, np.ones((len(x_all), 1))], axis=1)
    xt, yt = design[train], y_all[train]
    gram = xt.T @ xt; regularizer = np.eye(gram.shape[0]) * alpha
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(gram + regularizer, xt.T @ yt)
    return (design[test] @ weights) * ys + ym


def current_features(data):
    valid = np.arange(7)[None] < data["dof"][:, None]
    return np.concatenate([
      data["state"].reshape(len(valid), -1), data["action"], data["mask"], data["angle"],
      data["axes"].reshape(len(valid), -1), data["origins"].reshape(len(valid), -1),
      valid.astype(np.float32), data["object_pose"], data["object_twist"],
      data["ee_object_relative"], data["ee_action_delta"], data["ee_projected_action"]], 1).astype(np.float64)


def history_features(data):
    return np.concatenate([data[key].reshape(len(data[key]), -1) for key in HISTORY_KEYS], 1).astype(np.float64)


def evaluate(dataset: Path, seed: int):
    with np.load(dataset) as source:
        data = {key: np.asarray(source[key]) for key in source.files}
    robots, profiles = data["robot"].astype(str), data["profile"].astype(str)
    rng = np.random.default_rng(seed); train_ids = {}
    for robot in ("genkiarm", "panda"):
        ids = rng.permutation(np.unique(data["prefix_id"][robots == robot]))
        train_ids[robot] = set(map(int, ids[:int(0.7*len(ids))]))
    prefix_train = np.asarray([int(p) in train_ids[r] for p,r in zip(data["prefix_id"],robots)])
    middle = np.where(robots=="genkiarm",2,3); heldout_lock = data["lock_index"]==middle
    train = prefix_train & ~heldout_lock & (profiles!="heldout_mixed")
    test = ~prefix_train & heldout_lock & (profiles=="heldout_mixed")
    current, history = current_features(data), history_features(data)
    permutation = rng.permutation(len(history)); permuted = history[permutation]
    target = data["locked_object_step"].astype(np.float64)
    physics = data["physics_values"].astype(np.float64)
    feature_sets = {
      "current":current,
      "ordered_history":np.concatenate([current,history],1),
      "permuted_history":np.concatenate([current,permuted],1),
      "oracle_physics":np.concatenate([current,physics],1),
    }
    selected_data = {key:value[test] for key,value in data.items()}
    selected_robots = robots[test]; actual = target[test]
    methods={}
    for name, features in feature_sets.items():
        prediction=ridge_fit_predict(features,target,train,test)
        methods[name]={"prediction":response_metrics(actual,prediction,selected_robots,np.ones(len(actual),bool)),
                       "action":action_metrics(actual,prediction,selected_data,np.ones(len(actual),bool))}
    # Information upper bounds only: separate decoders are allowed to know the
    # robot identity. They diagnose whether failure comes from absent history
    # signal or from the shared raw-coordinate representation and are never a
    # deployable candidate or a Gate baseline.
    for label_name, features in (("current_robot_specific_oracle", current),
                                 ("history_robot_specific_oracle", np.concatenate([current, history], 1))):
        prediction = np.zeros_like(actual)
        test_global = np.flatnonzero(test)
        for robot in ("genkiarm", "panda"):
            robot_train = train & (robots == robot)
            robot_test = test & (robots == robot)
            robot_prediction = ridge_fit_predict(features, target, robot_train, robot_test)
            prediction[selected_robots == robot] = robot_prediction
        methods[label_name] = {
          "prediction": response_metrics(actual, prediction, selected_robots, np.ones(len(actual), bool)),
          "action": action_metrics(actual, prediction, selected_data, np.ones(len(actual), bool)),
          "uses_robot_identity": True,
          "evidential_role": "information_upper_bound_only"}
    physics_prediction=ridge_fit_predict(history,physics,train,test)
    physics_constant=np.broadcast_to(physics[train].mean(0),physics_prediction.shape)
    physics_true=physics[test]
    methods["history_physics_decode"]={
      "mae":np.mean(np.abs(physics_prediction-physics_true),axis=0).tolist(),
      "constant_mae":np.mean(np.abs(physics_constant-physics_true),axis=0).tolist()}
    base, hist, shuffled = methods["current"],methods["ordered_history"],methods["permuted_history"]
    terms={
      "pooled_rmse_improvement":(base["prediction"]["pooled"]-hist["prediction"]["pooled"])/base["prediction"]["pooled"],
      "spearman_improvement":hist["action"]["mean_spearman"]-base["action"]["mean_spearman"],
      "lower_regret":hist["action"]["normalized_top1_regret"]<base["action"]["normalized_top1_regret"],
      "both_robots_rmse_improve":all(hist["prediction"][r]<base["prediction"][r] for r in ("genkiarm","panda")),
      "ordered_better_than_permuted_rmse":hist["prediction"]["pooled"]<shuffled["prediction"]["pooled"],
      "ordered_better_than_permuted_spearman":hist["action"]["mean_spearman"]>shuffled["action"]["mean_spearman"]}
    return {"version":"active_probe_identifiability_v1","seed":seed,
      "split":{"train":int(train.sum()),"test":int(test.sum()),"heldout_profile":"heldout_mixed",
               "heldout_locks":{"genkiarm":"j3","panda":"joint4"},"grouped_by_prefix":True},
      "methods":methods,"gate_terms":terms}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); result=evaluate(args.dataset,args.seed); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
