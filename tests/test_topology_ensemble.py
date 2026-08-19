import numpy as np
import torch

from robotarm.models.topology_encoder import TopologyEncoder
from robotarm.models.world_model import WorldModel, WorldModelConfig
from robotarm.training.sim_data import SimTrajectory
from robotarm.training.sim_protocol import build_g1_protocol
from robotarm.training.topology_ensemble import TopologyMember, evaluate_topology_ensemble


def test_identical_ensemble_has_zero_disagreement():
    protocol = build_g1_protocol()
    domain = protocol.test[0]
    model = WorldModel(WorldModelConfig(state_dim=10, context_dim=64))
    encoder = TopologyEncoder()
    member = TopologyMember(encoder=encoder, world_model=model)
    trajectory = SimTrajectory(
        domain_id=domain.domain_id,
        states=torch.zeros(4, 10),
        actions=torch.zeros(3, 5),
        applied_actions=torch.zeros(3, 5),
    )
    metrics = evaluate_topology_ensemble(
        [member, member], domain, [trajectory],
        np.tile(np.array([[-1.0, 1.0]]), (5, 1)),
        device=torch.device("cpu"), horizon=3,
    )
    assert metrics["mean_uncertainty"] == 0.0
    assert metrics["uncertainty_error_spearman"] == 0.0
