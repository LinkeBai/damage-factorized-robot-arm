import torch

from robotarm.models.contact_action_ranker import ContactActionRanker, pairwise_ranking_loss


def test_contact_action_ranker_shape():
    model = ContactActionRanker(12, (8, 4))
    assert model(torch.zeros(5, 12)).shape == (5,)


def test_pairwise_loss_prefers_matching_order():
    costs = torch.tensor([0.0, 1.0, 2.0])
    good = pairwise_ranking_loss(torch.tensor([0.0, 1.0, 2.0]), costs)
    bad = pairwise_ranking_loss(torch.tensor([2.0, 1.0, 0.0]), costs)
    assert good < bad


def test_pairwise_loss_handles_equal_costs():
    scores = torch.tensor([0.2, -0.1], requires_grad=True)
    loss = pairwise_ranking_loss(scores, torch.ones(2))
    loss.backward()
    assert loss.item() == 0.0
    assert scores.grad is not None
