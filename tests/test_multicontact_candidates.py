import torch

from robotarm.models.hybrid_contact_impulse import HybridContactImpulseModel


def test_two_contact_candidates_have_orthonormal_planar_frames() -> None:
    model = HybridContactImpulseModel()
    state = torch.zeros(3, 14)
    state[:, 10:12] = torch.tensor([0.24, 0.10])
    gap, normal, tangent = model.candidate_contact_frames(state, state[:, :5])
    assert gap.shape == (3, 2)
    assert normal.shape == tangent.shape == (3, 2, 2)
    assert torch.allclose(torch.linalg.vector_norm(normal, dim=-1), torch.ones(3, 2))
    assert torch.allclose((normal * tangent).sum(-1), torch.zeros(3, 2), atol=1e-6)


def test_capsule_box_candidates_have_orthonormal_planar_frames() -> None:
    model = HybridContactImpulseModel()
    state = torch.zeros(3, 14)
    state[:, 10:12] = torch.tensor([0.24, 0.10])
    gap, normal, tangent = model.candidate_box_contact_frames(state, state[:, :5])
    assert gap.shape == (3, 2)
    assert torch.allclose(torch.linalg.vector_norm(normal, dim=-1), torch.ones(3, 2))
    assert torch.allclose((normal * tangent).sum(-1), torch.zeros(3, 2), atol=1e-6)
