import torch

from robotarm.training.sim_data import SimTrajectory


def test_contact_mask_is_optional_and_transition_aligned() -> None:
    trajectory = SimTrajectory(
        domain_id="intact__nominal",
        states=torch.zeros(4, 14),
        actions=torch.zeros(3, 5),
        applied_actions=torch.zeros(3, 5),
        contact_mask=torch.tensor([False, True, False]),
        contact_impulses=torch.zeros(3, 2),
        table_impulses=torch.zeros(3, 2),
        contact_records=[[], [], []],
    )
    assert trajectory.contact_mask is not None
    assert trajectory.contact_mask.dtype == torch.bool
    assert trajectory.contact_mask.shape[0] == trajectory.actions.shape[0]
    assert trajectory.contact_impulses is not None
    assert trajectory.contact_impulses.shape == (3, 2)
    assert trajectory.table_impulses is not None
    assert trajectory.table_impulses.shape == (3, 2)
    assert trajectory.contact_records is not None
    assert len(trajectory.contact_records) == 3
