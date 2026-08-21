"""Frozen product-space fusion of predictive and structural dynamics experts."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from robotarm.training.topology_ensemble import TopologyMember


@dataclass
class DualExpertHidden:
    predictive: list[torch.Tensor | None]
    structural: torch.Tensor | None


@dataclass
class DualExpertOutput:
    mean: torch.Tensor
    member_means: torch.Tensor
    epistemic_uncertainty: torch.Tensor
    cross_expert_discrepancy: torch.Tensor
    structural_prediction: torch.Tensor


class DualExpertWorldModel(nn.Module):
    """Use FT dynamics for joints and an ordinary ensemble for object state.

    All predictive members receive the same fused state at every step. This is
    deliberate: their object prediction is conditioned on the joint trajectory
    that will actually be returned by the fused model.
    """

    def __init__(
        self,
        predictive_experts: list[TopologyMember],
        structural_expert: nn.Module,
        *,
        joint_state_dim: int = 10,
    ) -> None:
        super().__init__()
        if not predictive_experts:
            raise ValueError("at least one predictive expert is required")
        self.predictive_encoders = nn.ModuleList(
            [member.encoder for member in predictive_experts]
        )
        self.predictive_models = nn.ModuleList(
            [member.world_model for member in predictive_experts]
        )
        self.structural_expert = structural_expert
        self.joint_state_dim = joint_state_dim
        self.requires_grad_(False)

    def initial_hidden(self) -> DualExpertHidden:
        return DualExpertHidden(
            predictive=[None] * len(self.predictive_models), structural=None
        )

    @torch.no_grad()
    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        contexts: list[torch.Tensor],
        mask: torch.Tensor,
        lock_angle: torch.Tensor,
        hidden: DualExpertHidden | None = None,
    ) -> tuple[DualExpertOutput, DualExpertHidden]:
        if len(contexts) != len(self.predictive_models):
            raise ValueError("one context tensor is required per predictive expert")
        hidden = hidden or self.initial_hidden()
        structural, structural_hidden = self.structural_expert.step(
            state, action, mask, lock_angle, hidden.structural
        )
        member_means, predictive_hidden = [], []
        for model, context, member_hidden in zip(
            self.predictive_models, contexts, hidden.predictive
        ):
            prediction, next_hidden = model.step(
                state, action, context, member_hidden
            )
            member_means.append(prediction["mean"])
            predictive_hidden.append(next_hidden)
        stacked = torch.stack(member_means)
        predictive_mean = stacked.mean(dim=0)
        fused = predictive_mean.clone()
        fused[:, : self.joint_state_dim] = structural[:, : self.joint_state_dim]
        epistemic = stacked.var(dim=0, unbiased=False).mean(dim=-1).sqrt()
        cross = (
            predictive_mean[:, : self.joint_state_dim]
            - structural[:, : self.joint_state_dim]
        ).pow(2).mean(dim=-1).sqrt()
        return (
            DualExpertOutput(
                mean=fused,
                member_means=stacked,
                epistemic_uncertainty=epistemic,
                cross_expert_discrepancy=cross,
                structural_prediction=structural,
            ),
            DualExpertHidden(
                predictive=predictive_hidden, structural=structural_hidden
            ),
        )
