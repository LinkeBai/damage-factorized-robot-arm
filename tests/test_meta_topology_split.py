from pathlib import Path

from robotarm.training.sim_protocol import load_g1_protocol


def test_meta_topology_split_brackets_d3_without_leaking_it():
    protocol = load_g1_protocol(Path(
        "config/splits/g2_topology_meta_train_d1_d5_v1.yaml"
    ))
    train_topologies = {domain.topology for domain in protocol.train}
    validation_topologies = {domain.topology for domain in protocol.validation}
    test_topologies = {domain.topology for domain in protocol.test}
    assert {"D1", "D2", "D4", "D5"} <= train_topologies
    assert "D3" not in train_topologies
    assert "D3" not in validation_topologies
    assert "D3" in test_topologies
