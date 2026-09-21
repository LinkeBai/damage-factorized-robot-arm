"""Critical provenance and resume checks for the isolated comparison trainer."""
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch

from work.supported_core_compare import train


def test_rng_round_trip_restores_numpy_torch_and_cuda():
    rng = np.random.default_rng(913)
    torch.manual_seed(91)
    saved = train.capture_rng(rng)
    expected_numpy = rng.integers(0, 10000, 20)
    expected_cpu = torch.rand(20)
    expected_cuda = torch.rand(20, device="cuda") if torch.cuda.is_available() else None
    train.restore_rng(saved, rng)
    np.testing.assert_array_equal(rng.integers(0, 10000, 20), expected_numpy)
    assert torch.equal(torch.rand(20), expected_cpu)
    if expected_cuda is not None:
        assert torch.equal(torch.rand(20, device="cuda"), expected_cuda)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.object_head = torch.nn.Linear(2, 2)


def toy_build(method, seed, device):
    torch.manual_seed(seed)
    return Toy().to(device)


def toy_loss(model, s, u, locks, starts, horizon):
    prediction = model.object_head(s[:, 0, :2])
    # Consume torch RNG as well as the trainer's NumPy RNG to catch incomplete resumes.
    prediction = prediction + torch.rand_like(prediction) * .01
    target = s[torch.arange(len(s)), starts + 1, :2]
    return (prediction - target).square().mean(), prediction


def toy_validation(model, data):
    s = data[0]
    with torch.no_grad():
        return float((model.object_head(s[:, 0, :2]) - s[:, 1, :2]).square().mean())


@pytest.fixture
def toy_setup(monkeypatch):
    monkeypatch.setattr(train, "build", toy_build)
    monkeypatch.setattr(train, "batch_rollout", toy_loss)
    monkeypatch.setattr(train, "validation", toy_validation)
    rng = np.random.default_rng(3)
    data = (torch.tensor(rng.normal(size=(7, 51, 14)), dtype=torch.float32),
            torch.zeros((7, 5, 5)), torch.zeros(7, dtype=torch.long), [str(i) for i in range(7)])
    spec = dict(train.EXPECTED, epochs=3, batch_size=3, horizon_schedule=[1, 1, 1])
    return data, spec


def test_epoch_boundary_resume_matches_uninterrupted_fit(tmp_path, monkeypatch, toy_setup):
    data, spec = toy_setup
    identity = {"test": "fixed"}
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    expected = train.fit("carrier", 7, data, data, spec, identity, destination=full)
    original_write = train.write

    def crash_after_resume(path, value):
        if Path(path).name == "progress.json":
            raise RuntimeError("simulated process failure after committed epoch")
        original_write(path, value)

    monkeypatch.setattr(train, "write", crash_after_resume)
    with pytest.raises(RuntimeError, match="simulated process failure"):
        train.fit("carrier", 7, data, data, spec, identity, destination=resumed)
    monkeypatch.setattr(train, "write", original_write)
    actual = train.fit("carrier", 7, data, data, spec, identity, destination=resumed)
    assert expected["history"] == actual["history"]
    assert expected["best_epoch"] == actual["best_epoch"]
    assert expected["total_updates"] == actual["total_updates"] == 9
    for folder in (full, resumed):
        latest = torch.load(folder / "resume.pt", weights_only=False)
        if folder == full:
            expected_state = copy.deepcopy(latest)
        else:
            assert train.tensor_hash(latest["model"]) == train.tensor_hash(expected_state["model"])
            assert torch.equal(latest["rng"]["torch_cpu"], expected_state["rng"]["torch_cpu"])
    assert train.tensor_hash(torch.load(full / "model.pt", weights_only=True)) == train.tensor_hash(
        torch.load(resumed / "model.pt", weights_only=True))


def test_completed_fit_is_immutable_and_provenance_checked(tmp_path, monkeypatch, toy_setup):
    data, spec = toy_setup
    folder = tmp_path / "fit"
    identity = {"test": "fixed"}
    train.fit("carrier", 7, data, data, spec, identity, destination=folder)
    original_hashes = {name: train.sha(folder / name) for name in ("model.pt", "complete.json", "resume.pt")}
    monkeypatch.setattr(train, "build", lambda *_: pytest.fail("Completed fit must not initialize again"))
    train.fit("carrier", 7, data, data, spec, identity, destination=folder)
    with pytest.raises(ValueError, match="provenance differs"):
        train.fit("carrier", 7, data, data, spec, {"test": "changed"}, destination=folder)
    assert original_hashes == {name: train.sha(folder / name) for name in original_hashes}


def test_loader_preserves_shard_order_and_does_not_load_test(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "OUT", tmp_path)
    for split in ("pool", "test"):
        for index in (1, 0):
            folder = tmp_path / "data" / split / f"shard-{index:06d}"
            folder.mkdir(parents=True)
            (folder / "data.npz").write_bytes(b"must not open") if split == "test" else np.savez(
                folder / "data.npz", states=np.full((2, 51, 14), index, dtype=np.float32),
                segment_actions=np.zeros((2, 5, 5), dtype=np.float32), locked_joint=np.zeros(2, dtype=np.int8),
                reset_id=np.asarray([f"{index}-0", f"{index}-1"]))
    s, u, locks, ids = train.load_dataset("pool", torch.device("cpu"))
    assert ids == ["0-0", "0-1", "1-0", "1-1"]
    assert s.dtype == u.dtype == torch.float32
    assert locks.dtype == torch.int64
    assert s[:, 0, 0].tolist() == [0., 0., 1., 1.]


def test_source_initialization_shared_weights_and_trainable_budgets():
    audit = train.audit_initialization("cpu")
    assert audit["passed"]
    assert {m: x["trainable_parameters"] for m, x in audit["methods"].items()} == {
        "ipwm": 20000, "carrier": 19724, "global": 20008}
    assert len({x["common_initial_weights_sha256"] for x in audit["methods"].values()}) == 1


def test_audit_rejects_modified_training_data_and_ignores_test_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "OUT", tmp_path)
    report = {"passed": True, "cross_split_and_development_disjoint": True, "splits": {}, "files": {}}
    for split, count in (("pool", 50000), ("validation", 2000)):
        folder = tmp_path / "data" / split / "shard-000000-000250"
        folder.mkdir(parents=True)
        shard = {}
        for key, filename in (("npz", "data.npz"), ("records", "records.json"), ("manifest", "manifest.json")):
            path = folder / filename
            path.write_bytes(b"fixed audited bytes")
            shard[f"{key}_path"] = str(path)
            shard[f"{key}_sha256"] = train.sha(path)
            report["files"][path.relative_to(tmp_path).as_posix()] = train.sha(path)
        report["splits"][split] = {"passed": True, "count": count, "shards": [shard]}
    # No test files are present: training provenance reads only its audit metadata.
    assert len(train.audited_training_files(report)) == 6
    Path(report["splits"]["pool"]["shards"][0]["npz_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Data changed after audit"):
        train.audited_training_files(report)
