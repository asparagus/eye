"""Motor module for the eye."""

import torch
from torch import nn
from jigsaw.piece import Piece

from eye.architecture.names import (
    EMBEDDING,
    FOCUS_NEXT,
    FOCUS_POINT,
    MOTOR_LOSS,
)


class MotorModule(Piece):
    def __init__(self, dims: tuple[int, int], num_filters: int):
        """Initialize the eye motor network.

        Args:
            dims: The dimensions of the input image.
            num_filters: The number of filters that are input to the motor.
        """
        super().__init__(piece_type="module")
        self.dims = dims
        self.maxs = torch.Tensor(dims) - 1
        self.zeros = torch.zeros_like(self.maxs)
        self.refocus = nn.Linear(in_features=num_filters, out_features=2, bias=True)
        self.loss = nn.MSELoss(reduction="none")

    def inputs(self) -> tuple[str, ...]:
        return (EMBEDDING, FOCUS_POINT)

    def outputs(self) -> tuple[str, ...]:
        return (FOCUS_NEXT, MOTOR_LOSS)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the new focus given the input and the current focus.

        Args:
            embedding_input: (B, F) input tensor
            current_focus: (B, 2) focus tensor

        Returns:
            (B, 2) new focus tensor
            (B, 1) motor loss
        """
        embedding = inputs[EMBEDDING]
        focus = inputs[FOCUS_POINT]
        assert len(embedding.shape) == 2
        assert len(focus.shape) == 2
        assert embedding.shape[0] == focus.shape[0]
        refocus = self.refocus(embedding)
        new_focus = focus + refocus
        assert new_focus.shape == (embedding.shape[0], 2)

        # Compute boundary violation loss using MSE between unclamped and clamped focus
        maxs = self.maxs.to(new_focus.device)
        zeros = self.zeros.to(new_focus.device)
        clamped_focus = torch.clamp(new_focus, min=zeros, max=maxs)
        boundary_loss = self.loss(new_focus, clamped_focus).sum(dim=1, keepdim=True)
        assert boundary_loss.shape == (embedding.shape[0], 1)

        return {FOCUS_NEXT: new_focus, MOTOR_LOSS: boundary_loss}
