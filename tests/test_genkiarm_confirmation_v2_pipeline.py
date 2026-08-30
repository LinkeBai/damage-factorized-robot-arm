import json
from pathlib import Path

import yaml

from scripts.run_genkiarm_confirmation_v2 import SEEDS, STAGES, _command, _write_execution_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_all_frozen_configs_use_the_preregistered_seed_set_and_paths_exist():
    contract = yaml.safe_load((ROOT / "config/experiment/icra_2027_genkiarm_confirmation_v2.yaml").read_text())
    assert tuple(contract["training"]["fresh_seeds"]) == SEEDS
    for key in ("base_model_config", "zero_topology_config", "contact_residual_config",
                "physical_context_config", "context_encoder_config", "evaluation_config"):
        path = ROOT / contract["training"][key]
        assert path.is_file(), key
        config = yaml.safe_load(path.read_text())
        assert tuple(config["seeds"]) == SEEDS


def test_pipeline_commands_keep_xml_and_disjoint_query_seed():
    for stage in STAGES:
        command = _command(stage, 107)
        assert "--xml" in command
        assert any(value.replace("\\", "/") == "sim/assets/genkiarm_push.xml" for value in command)
    evaluation = _command("evaluate", 107)
    assert evaluation[evaluation.index("--query-seed-base") + 1] == "1107"


def test_execution_manifest_records_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_execution_manifest(107)
    payload = json.loads((tmp_path / "runs/g2_ipwm_genkiarm_confirmation_v2/seed107_v1/execution_manifest.json").read_text())
    assert payload["seed"] == 107 and payload["query_seed_base"] == 1107
    assert payload["xml"].replace("\\", "/").endswith("sim/assets/genkiarm_push.xml")
