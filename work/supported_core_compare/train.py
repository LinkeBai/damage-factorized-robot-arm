"""Matched supported-reset adaptation fits; test data never enters training.

The fit seeds change adaptation initialization/minibatch order. All nine fits
share pretraining seed 27 and are not independent pretraining repetitions.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]
import numpy as np
import torch

from work.supported_core_compare.common import ROOT, OUT, sha, read, write, load_protocol
from scripts.ipwm_scale_train import build, batch_rollout, validation

METHODS = ("ipwm", "carrier", "global")
SEEDS = (7, 17, 27)
EXPECTED = dict(seeds=list(SEEDS), methods=list(METHODS), epochs=10,
                batch_size=512, learning_rate=.0003, gradient_clip=5.,
                horizon_schedule=[10] * 5 + [25] * 5)
IDENTITY_TEXT = ("Matched adaptation repetitions sharing pretrained carrier seed27; "
                 "not independent pretraining or new real-robot deployment evidence.")


def tensor_hash(values):
    """Stable value hash independent of torch serialization container metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_manifest():
    paths = [
        "scripts/ipwm_scale_train.py", "scripts/audit_ipwm_targeted_advantage.py",
        "config/experiment/icra_primary_d2d4_decision_development_3seed_v1.yaml",
        "runs/icra_primary_decision_full_w10_128eval_strict_v2/seed27/model.pt",
        "runs/icra_primary_global_matched_w10_128eval_strict_v2/seed27/model.pt",
        "runs/g2_bt_dpwm_meta_train_z32/seed27_v1/baseline_model.pt",
    ]
    paths += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "src/robotarm/models").glob("*.py"))]
    return {path: sha(ROOT / path) for path in paths}


def audit_initialization(device="cpu"):
    """Read and construct all source models without training or simulation."""
    models = {method: build(method, SEEDS[0], torch.device(device)) for method in METHODS}
    states = {method: model.state_dict() for method, model in models.items()}
    common = set.intersection(*(set(state) for state in states.values()))
    common = sorted(name for name in common if not name.startswith(("geometric_object_head.", "global_residual_head.")))
    hashes = {method: tensor_hash({name: state[name] for name in common}) for method, state in states.items()}
    if len(set(hashes.values())) != 1:
        raise ValueError("Common initial carrier/object weights differ between methods")
    records = {}
    for method, model in models.items():
        trainable = {name: int(value.numel()) for name, value in model.named_parameters() if value.requires_grad}
        if not trainable or any(not (name.startswith("object_") or
                (method == "ipwm" and name.startswith("geometric_object_head.")) or
                (method == "global" and name.startswith("global_residual_head."))) for name in trainable):
            raise ValueError("Unexpected trainable parameter set")
        records[method] = dict(trainable_parameters=sum(trainable.values()), trainable_names=trainable,
                               common_initial_weights_sha256=hashes[method])
    return dict(passed=True, identity=IDENTITY_TEXT, common_pretraining_seed=27,
                source_hashes=source_manifest(), common_parameter_names=common, methods=records)


def capture_rng(rng):
    return dict(numpy=rng.bit_generator.state, torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(saved, rng):
    rng.bit_generator.state = saved["numpy"]
    torch.set_rng_state(saved["torch_cpu"].cpu())
    if saved["torch_cuda"]:
        if not torch.cuda.is_available() or len(saved["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed since resume")
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["torch_cuda"]])


def save_torch(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    for attempt in range(10):
        try:
            temp.replace(path)
            return
        except PermissionError:
            # Windows indexing/antivirus can briefly hold a just-written file.
            if attempt == 9:
                raise
            time.sleep(min(.02 * 2 ** attempt, .25))


def dataset_files(split):
    if split not in ("pool", "validation", "test"):
        raise ValueError("Unknown dataset split")
    return sorted((OUT / "data" / split).glob("shard-*/data.npz"))


def load_dataset(split, device, *, limit=None, paths=None):
    """Load in stable shard order, using one preallocated CPU buffer per tensor."""
    paths = list(paths) if paths is not None else dataset_files(split)
    if not paths:
        raise FileNotFoundError(f"No {split} shards under {OUT / 'data'}")
    counts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            counts.append(len(arrays["locked_joint"]))
    count = min(sum(counts), limit) if limit is not None else sum(counts)
    states = np.empty((count, 51, 14), dtype=np.float32)
    actions = np.empty((count, 5, 5), dtype=np.float32)
    locks = np.empty(count, dtype=np.int64)
    ids = []
    cursor = 0
    for path, size in zip(paths, counts):
        take = min(size, count - cursor)
        if take <= 0:
            break
        with np.load(path, allow_pickle=False) as arrays:
            states[cursor:cursor + take] = arrays["states"][:take]
            actions[cursor:cursor + take] = arrays["segment_actions"][:take]
            locks[cursor:cursor + take] = arrays["locked_joint"][:take]
            ids.extend(arrays["reset_id"][:take].tolist())
        cursor += take
    if not np.isfinite(states).all() or not np.isfinite(actions).all() or np.any((locks < 0) | (locks > 4)):
        raise ValueError("Invalid training tensor")
    return (torch.from_numpy(states).to(device), torch.from_numpy(actions).to(device),
            torch.from_numpy(locks).to(device), ids)


def training_spec(protocol):
    spec = protocol.get("training", protocol)
    for key, expected in EXPECTED.items():
        if spec.get(key) != expected:
            raise ValueError(f"Frozen training setting differs: {key}")
    return spec


def audited_training_files(audit):
    """Recheck the audit's exact pool/validation artifacts without opening test data."""
    if not audit.get("passed") or not audit.get("cross_split_and_development_disjoint"):
        raise ValueError("Formal training requires a passed disjoint full-data audit")
    files = {}
    for split, count in (("pool", 50000), ("validation", 2000)):
        report = audit["splits"][split]
        if not report.get("passed") or report["count"] != count:
            raise ValueError(f"Audited sample budget differs: {split}")
        expected_npz_paths = []
        for shard in report["shards"]:
            for name in ("npz", "records", "manifest"):
                path = Path(shard[f"{name}_path"]).resolve()
                if not path.is_relative_to((OUT / "data" / split).resolve()):
                    raise ValueError("Audited shard path leaves the declared split")
                if sha(path) != shard[f"{name}_sha256"]:
                    raise ValueError(f"Data changed after audit: {path}")
                relative_path = path.relative_to(OUT.resolve()).as_posix()
                if audit["files"].get(relative_path) != shard[f"{name}_sha256"]:
                    raise ValueError("Audit file index disagrees with shard provenance")
                files[relative_path] = shard[f"{name}_sha256"]
                if name == "npz":
                    expected_npz_paths.append(path)
        if expected_npz_paths != [path.resolve() for path in dataset_files(split)]:
            raise ValueError(f"Loaded shard list differs from frozen audit: {split}")
    return files


def formal_identity():
    protocol = load_protocol()
    training_spec(protocol)
    protocol_sha = sha(OUT / "protocol.json")
    if read(OUT / "protocol-frozen.json")["protocol_sha256"] != protocol_sha:
        raise ValueError("Frozen protocol changed")
    sources = protocol["source_sha256"]
    for path, expected in sources.items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    for path, current in source_manifest().items():
        if sources.get(path) != current:
            raise ValueError(f"Training dependency absent from frozen sources: {path}")
    implementation_path = OUT / "implementation-frozen.json"
    implementation = read(implementation_path)
    if implementation["protocol_sha256"] != protocol_sha:
        raise ValueError("Implementation belongs to a different protocol")
    for path, expected in implementation["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen implementation changed: {path}")
    if implementation["source_sha256"].get(Path(__file__).relative_to(ROOT).as_posix()) != sha(__file__):
        raise ValueError("Training implementation was not frozen")
    physics_path = OUT / "physics-gate.json"
    audit_path = OUT / "data-audit.json"
    physics = read(physics_path)
    audit = read(audit_path)
    if not physics.get("passed"):
        raise ValueError("Formal training requires passed physics gate")
    if not audit.get("passed"):
        raise ValueError("Formal training requires passed data audit")
    for gate in (physics, audit):
        if gate.get("protocol_sha256", protocol_sha) != protocol_sha:
            raise ValueError("Gate belongs to a different protocol")
    training_files = audited_training_files(audit)
    return dict(protocol_sha256=protocol_sha,
                source_hashes=sources, train_script_sha256=sha(__file__),
                implementation_sha256=sha(implementation_path),
                data_audit_sha256=sha(audit_path), physics_gate_sha256=sha(physics_path),
                train_shard_hashes=training_files)


def verify_complete(folder, identity=None):
    result = read(folder / "complete.json")
    if not result or result.get("model_sha256") != sha(folder / "model.pt"):
        raise ValueError(f"Invalid completed model: {folder}")
    if identity is not None and result.get("identity") != identity:
        raise ValueError(f"Completed model provenance differs: {folder}")
    if identity is not None and "protocol_sha256" in identity:
        if (result.get("smoke") is not False or result.get("total_updates") != 980 or
                result.get("actual_training_unique_trajectories") != 50000 or
                result.get("examples_per_epoch") != 50000 or len(result.get("history", [])) != 10 or
                result.get("method") != folder.parent.name or folder.name != f"seed{result.get('seed')}"):
            raise ValueError(f"Completed model does not satisfy frozen training budget: {folder}")
        if any(row.get("examples") != 50000 or row.get("unique_examples") != 50000 for row in result["history"]):
            raise ValueError("Completed model has incomplete epoch coverage")
        if result["method"] == "global" and not result.get("global_robot_first_gradient_norm", 0) > 0:
            raise ValueError("Global robot output gradient audit missing")
        if result["method"] == "global" and not result.get("global_robot_direct_supervision_gradient_norm", 0) > 0:
            raise ValueError("Global direct robot-supervision gradient audit missing")
    return result


def load_selected(method, seed, device):
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("Unknown method/seed")
    folder = OUT / "training" / method / f"seed{seed}"
    complete = verify_complete(folder, formal_identity())
    model = build(method, seed, torch.device(device))
    model.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True))
    return model.eval(), complete["model_sha256"]


def fit(method, seed, data, val, spec, identity, *, destination=None, max_updates=None,
        initialization_audit=None):
    folder = destination or OUT / "training" / method / f"seed{seed}"
    if (folder / "complete.json").exists():
        return verify_complete(folder, identity)
    wall_started = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    device = data[0].device
    model = build(method, seed, device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=spec["learning_rate"])
    rng = np.random.default_rng(seed)
    history = []
    seen = set()
    completed_epochs = 0
    gradient = None
    direct_gradient = None
    total_updates = 0
    prior_wall = 0.
    prior_core = 0.
    common_hash = None
    trainable = {name: int(p.numel()) for name, p in model.named_parameters() if p.requires_grad}
    if initialization_audit is not None:
        expected = initialization_audit["methods"][method]
        common_hash = tensor_hash({name: model.state_dict()[name] for name in initialization_audit["common_parameter_names"]})
        if trainable != expected["trainable_names"] or common_hash != expected["common_initial_weights_sha256"]:
            raise ValueError("Fit initialization does not match frozen initialization audit")
    if (folder / "resume.pt").exists():
        saved = torch.load(folder / "resume.pt", map_location=device, weights_only=False)
        if saved["identity"] != identity or saved["method"] != method or saved["seed"] != seed:
            raise ValueError("Resume provenance differs")
        if saved["selected_model_sha256"] != sha(folder / "model.pt"):
            # A crash may happen between updating the selected file and saving
            # the epoch resume. Restore the committed selection, then replay.
            if "selected_model" not in saved:
                raise ValueError("Selected checkpoint changed since resume")
            save_torch(folder / "model.pt", saved["selected_model"])
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        completed_epochs = saved["epoch"]
        seen = set(saved["seen"])
        best, best_epoch = saved["best"], saved["best_epoch"]
        initial_validation = saved["initial_validation_xy_mse"]
        gradient = saved["global_robot_first_gradient_norm"]
        direct_gradient = saved["global_robot_direct_supervision_gradient_norm"]
        total_updates = saved["total_updates"]
        prior_wall, prior_core = saved["wall_seconds"], saved["core_seconds"]
        restore_rng(saved["rng"], rng)
    else:
        best, best_epoch = validation(model, val), 0
        initial_validation = best
        if not math.isfinite(best):
            raise ValueError("Nonfinite initial validation")
        save_torch(folder / "model.pt", model.state_dict())
    core_started = time.perf_counter()
    s, u, locks, ids = data
    for epoch, horizon in enumerate(spec["horizon_schedule"], 1):
        if epoch <= completed_epochs:
            continue
        order = rng.permutation(len(s))
        total = 0.
        count = 0
        epoch_seen = []
        for start in range(0, len(order), spec["batch_size"]):
            selected = order[start:start + spec["batch_size"]]
            epoch_seen.extend(selected.tolist())
            seen.update(selected.tolist())
            ix = torch.tensor(selected, device=device)
            starts = torch.tensor(rng.integers(0, 51 - horizon, len(ix)), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, prediction = batch_rollout(model, s[ix], u[ix], locks[ix], starts, horizon)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            if direct_gradient is None and method == "global":
                robot_scales = prediction.new_tensor([.1] * 5 + [1.] * 5)
                robot_loss = ((prediction[:, :10] - s[ix, starts + horizon, :10]) / robot_scales).square().mean()
                robot_grad = torch.autograd.grad(robot_loss, model.global_residual_head[-1].weight, retain_graph=True)[0]
                direct_gradient = float(robot_grad[:10].norm())
                if not math.isfinite(direct_gradient) or direct_gradient <= 0:
                    raise ValueError("Robot-only supervised loss supplies no global robot-output gradient")
            loss.backward()
            if gradient is None and method == "global":
                gradient = float(model.global_residual_head[-1].weight.grad[:10].norm())
                if not math.isfinite(gradient) or gradient <= 0:
                    raise ValueError("Global robot residual lacks direct supervision gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec["gradient_clip"])
            optimizer.step()
            total += loss.item() * len(ix)
            count += len(ix)
            total_updates += 1
            if max_updates is not None and total_updates >= max_updates:
                break
        if max_updates is None and (len(epoch_seen) != len(s) or len(set(epoch_seen)) != len(s)):
            raise ValueError("Epoch did not consume each trajectory exactly once")
        score = validation(model, val)
        if not math.isfinite(score):
            raise ValueError("Nonfinite validation score")
        if score < best:
            best, best_epoch = score, epoch
            save_torch(folder / "model.pt", model.state_dict())
        history.append(dict(epoch=epoch, horizon=horizon, train_loss=total / count,
                            validation_xy_mse=score, examples=count, unique_examples=len(set(epoch_seen))))
        wall_seconds = prior_wall + time.perf_counter() - wall_started
        core_seconds = prior_core + time.perf_counter() - core_started
        save_torch(folder / "resume.pt", dict(method=method, seed=seed, identity=identity,
            model=model.state_dict(), optimizer=optimizer.state_dict(), rng=capture_rng(rng),
            selected_model=torch.load(folder / "model.pt", map_location="cpu", weights_only=True),
            history=history, epoch=epoch, best=best, best_epoch=best_epoch, seen=sorted(seen),
            initial_validation_xy_mse=initial_validation,
            selected_model_sha256=sha(folder / "model.pt"), total_updates=total_updates,
            global_robot_first_gradient_norm=gradient, global_robot_direct_supervision_gradient_norm=direct_gradient,
            wall_seconds=wall_seconds, core_seconds=core_seconds))
        write(folder / "progress.json", dict(method=method, seed=seed, history=history,
              best_epoch=best_epoch, wall_seconds=wall_seconds, core_seconds=core_seconds))
        print(json.dumps(dict(method=method, seed=seed, epoch=epoch, validation_xy_mse=score,
                              wall_seconds=round(wall_seconds, 2))), flush=True)
        if max_updates is not None and total_updates >= max_updates:
            break
    if max_updates is None and (len(history) != spec["epochs"] or len(seen) != len(s)):
        raise ValueError("Incomplete training cannot be finalized")
    result = dict(method=method, seed=seed, identity=identity, repetition_identity=IDENTITY_TEXT,
        history=history, best_epoch=best_epoch, best_validation_xy_mse=best,
        initial_validation_xy_mse=initial_validation, common_initial_weights_sha256=common_hash,
        actual_training_unique_trajectories=len(seen), examples_per_epoch=len(ids), total_updates=total_updates,
        global_robot_first_gradient_norm=gradient, trainable_parameters=sum(trainable.values()), trainable_names=trainable,
        global_robot_direct_supervision_gradient_norm=direct_gradient,
        model_sha256=sha(folder / "model.pt"), wall_seconds=prior_wall + time.perf_counter() - wall_started,
        core_seconds=prior_core + time.perf_counter() - core_started, smoke=max_updates is not None)
    write(folder / "complete.json", result)
    del model, optimizer
    gc.collect()
    return result


def freeze_completed(identity):
    models = {}
    for seed in SEEDS:
        for method in METHODS:
            folder = OUT / "training" / method / f"seed{seed}"
            if not (folder / "complete.json").exists():
                return None
            models[f"{method}/seed{seed}"] = verify_complete(folder, identity)["model_sha256"]
    result = dict(passed=True, protocol_sha256=identity["protocol_sha256"], methods=list(METHODS), seeds=list(SEEDS), models=models,
                  identity=identity, repetition_identity=IDENTITY_TEXT)
    path = OUT / "training-complete.json"
    if path.exists():
        if read(path) != result:
            raise ValueError("Frozen model registry changed")
    else:
        write(path, result)
    return result


def freeze_selection():
    result = freeze_completed(formal_identity())
    if result is None:
        raise ValueError("All nine model selections must complete before freezing")
    return result


def verify_selection_freeze():
    path = OUT / "training-complete.json"
    if not path.exists():
        raise ValueError("All nine model selections must be frozen before test access")
    existing = read(path)
    result = freeze_completed(formal_identity())
    if result is None or result != existing:
        raise ValueError("Frozen model selection provenance differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and not args.all and (args.method is None or args.seed is None):
        parser.error("Use --all or both --method and --seed")
    if args.all and (args.method is not None or args.seed is not None):
        parser.error("--all cannot be combined with --method/--seed")
    worker_started = time.perf_counter()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("The frozen training protocol requires CUDA")
    device = torch.device("cuda")
    if args.smoke:
        spec = EXPECTED.copy()
        identity = dict(smoke=True, source_hashes=source_manifest(), train_script_sha256=sha(__file__))
        # Smoke uses development-only data; formal dataset audit is deliberately not required.
        development_path = OUT / load_protocol().get("development_directory", "data-development") / "data.npz"
        data = load_dataset("pool", device, limit=1024, paths=[development_path])
        val = load_dataset("validation", device, limit=128, paths=[development_path])
        init = audit_initialization("cpu")
        seed = args.seed or SEEDS[0]
        smoke_root = OUT / "smoke" / "training" / f"{time.time_ns()}"
        write(smoke_root / "initialization-audit.json", init)
        for method in ([args.method] if args.method else METHODS):
            destination = smoke_root / method / f"seed{seed}"
            fit(method, seed, data, val, spec, identity, destination=destination, max_updates=2,
                initialization_audit=init)
        write(smoke_root / "worker-timing.json", dict(full_wall_seconds=time.perf_counter() - worker_started,
              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device), smoke=True))
        return
    protocol = load_protocol()
    spec = training_spec(protocol)
    identity = formal_identity()
    init = audit_initialization("cpu")
    init_path = OUT / "training-initialization-audit.json"
    if init_path.exists():
        if read(init_path) != init:
            raise ValueError("Frozen initialization audit changed")
    else:
        write(init_path, init)
    data = load_dataset("pool", device)
    val = load_dataset("validation", device)
    if len(data[0]) != 50000 or len(val[0]) != 2000:
        raise ValueError("Frozen sample budget is 50000 pool and 2000 validation trajectories")
    shared_setup_seconds = time.perf_counter() - worker_started
    jobs = [(method, seed) for seed in SEEDS for method in METHODS] if args.all else [(args.method, args.seed)]
    for method, seed in jobs:
        fit(method, seed, data, val, spec, identity, initialization_audit=init)
    freeze_completed(identity)
    write(OUT / "training-worker-timing.json", dict(jobs=[dict(method=m, seed=s) for m, s in jobs],
          shared_setup_seconds=shared_setup_seconds, full_worker_wall_seconds=time.perf_counter() - worker_started,
          identity=identity, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))


if __name__ == "__main__":
    main()
