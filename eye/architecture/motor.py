"""Motor module for the eye."""

import torch
from torch import nn
from jigsaw.piece import Piece

from eye.architecture.names import (
    FOCUS_NEXT,
    FOCUS_POINT,
    MOTOR_LOSS,
    STATE,
)


class MotorModule(Piece):
    def __init__(self, dims: tuple[int, int], num_filters: int, noise_std: float = 0.0):
        """Initialize the eye motor network.

        Args:
            dims: The dimensions of the input image.
            num_filters: The number of filters that are input to the motor.
            noise_std: Standard deviation of Gaussian noise added to focus predictions.
        """
        super().__init__(piece_type="module")
        self.dims = dims
        self.maxs = torch.Tensor(dims) - 1
        self.zeros = torch.zeros_like(self.maxs)
        self.refocus = nn.Linear(in_features=num_filters, out_features=2, bias=True)
        self.loss = nn.MSELoss(reduction="none")
        self.layer_norm = nn.LayerNorm(num_filters)
        self.noise_std = noise_std

    def inputs(self) -> tuple[str, ...]:
        return (STATE, FOCUS_POINT)

    def outputs(self) -> tuple[str, ...]:
        return (FOCUS_NEXT, MOTOR_LOSS)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the new focus given the input and the current focus.

        Args:
            state: (B, F) input tensor
            current_focus: (B, 2) focus tensor

        Returns:
            (B, 2) new focus tensor
            (B, 1) motor loss
        """
        state = inputs[STATE]
        focus = inputs[FOCUS_POINT]
        batch_size = state.shape[0]
        assert len(state.shape) == 2
        assert len(focus.shape) == 2
        assert batch_size == focus.shape[0]
        refocus = self.refocus(self.layer_norm(state))

        # Add Gaussian noise to refocus prediction during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(refocus) * self.noise_std
            refocus = refocus + noise

        new_focus = focus + refocus
        assert new_focus.shape == (batch_size, 2)

        # Compute boundary violation loss using MSE between unclamped and clamped focus
        maxs = self.maxs.to(new_focus.device)
        zeros = self.zeros.to(new_focus.device)
        clamped_focus = torch.clamp(new_focus, min=zeros, max=maxs)
        boundary_loss = self.loss(new_focus, clamped_focus).sum(dim=1, keepdim=True)
        assert boundary_loss.shape == (batch_size, 1)

        return {FOCUS_NEXT: new_focus, MOTOR_LOSS: boundary_loss}
