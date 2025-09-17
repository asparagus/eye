"""Retina module for the eye."""

import math
import torch
from jigsaw.piece import Piece

from eye.architecture.names import (
    FOCUS_POINT,
    IMAGE_INPUT,
    RETINA_OUTPUT,
)


# Retina constants
STD_RADIUS_DIVISOR = 6


class RetinaModule(Piece):
    """Simplified model of the retina.

    Will attend to a focal point within the image, given as a position to its
    forward function (focus).
    """

    def __init__(
        self,
        dims: tuple[int, int],
        num_filters: int,
        fovea_radius: float,
        frozen: bool = True,
    ):
        """Initialize the retina module.

        Args:
            dims: The dimensions of the input image.
            num_filters: The number of filters to use in the convolutional layer.
            fovea_radius: The radius of the fovea which relates to the kernel size
              and the gaussian mask function.
            frozen: Whether the filters are frozen (and do not learn)
        """
        super().__init__(piece_type="module")
        self.std = fovea_radius / STD_RADIUS_DIVISOR
        self.fovea_radius = int(math.ceil(fovea_radius))
        self.base_coords = self.coords(dims=dims)
        kernel_size = 1 + 2 * self.fovea_radius
        self.conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            bias=False,
        )
        # Freeze the weights of the convolutional layer
        if frozen:
            for param in self.conv.parameters():
                param.requires_grad = False

        kernel_mask = self.gaussian_mask(
            coords=self.coords(dims=(kernel_size, kernel_size)),
            focus=torch.Tensor([[1 + self.fovea_radius, 1 + self.fovea_radius]]),
            std=self.std,
        )
        self.normalization_constant = kernel_mask.sum()
        self.layer_norm = torch.nn.LayerNorm(num_filters)

    @classmethod
    def coords(cls, dims: tuple[int, ...]) -> torch.Tensor:
        """Compute the coordinates of the image."""
        ones = torch.ones(*dims)
        x = torch.cumsum(ones, dim=0) - 1
        y = torch.cumsum(ones, dim=1) - 1
        return torch.cat([x.unsqueeze(2), y.unsqueeze(2)], dim=2)

    @classmethod
    def gaussian_mask(
        cls,
        coords: torch.Tensor,
        focus: torch.Tensor,
        std: float,
    ) -> torch.Tensor:
        """Compute a gaussian mask for a focus on a particular position.

        Args:
            coords: (H, W, 2) coordinates for each cell in a grid
            focus: (B, 2) coordinates for the focus point
            std: standard deviation of the gaussian mask

        Returns:
            (B, H, W) mask tensor
        """
        assert len(coords.shape) == 3, f"Wrong coords shape: {coords.shape}"
        assert coords.shape[-1] == 2, f"Wrong coords shape: {coords.shape}"
        assert len(focus.shape) == 2, f"Wrong focus shape: {focus.shape}"
        assert focus.shape[-1] == 2, f"Wrong focus shape: {focus.shape}"
        norm_const = torch.sqrt(
            torch.tensor(
                2 * torch.pi * std**2, dtype=torch.float32, device=focus.device
            )
        )
        batched_coords = (
            coords.unsqueeze(0).repeat(focus.shape[0], 1, 1, 1).to(focus.device)
        )
        focus_with_dimensions = focus.reshape((focus.shape[0], 1, 1, focus.shape[-1]))
        mask = (
            torch.exp(
                -torch.sum((batched_coords - focus_with_dimensions) ** 2, dim=-1)
                / (2 * std**2)
            )
            / norm_const
        )
        assert len(mask.shape) == 3, f"Wrong mask shape: {mask.shape}"
        assert mask.shape[0] == focus.shape[0], (
            f"Wrong batch dimension: {mask.shape[0]}"
        )
        assert mask.shape[1:] == coords.shape[:-1], (
            f"Wrong mask dimensions {mask.shape[1:]} don't match {coords.shape[:-1]}"
        )
        return mask

    def filter_output(
        self,
        image: torch.Tensor,
        focus: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the sensory input given the current focus.

        Args:
        image: (B, H, W) input tensor
        focus: (B, 2) focus tensor

        Returns:
        (B, F) sensory input tensor with a number of filters
        """
        assert len(image.shape) == 3, f"Wrong image shape: {image.shape}"
        assert len(focus.shape) == 2, f"Wrong focus shape: {focus.shape}"
        assert image.shape[0] == focus.shape[0], (
            f"Mismatch in batch size: {image.shape[0]} v/s {focus.shape[0]}"
        )
        mask = self.gaussian_mask(
            coords=self.base_coords,
            focus=focus,
            std=self.std,
        )
        image_with_filters = image.unsqueeze(1)
        raw_filter_output = self.conv(image_with_filters)
        out_sum = torch.sum(
            torch.sum(raw_filter_output * mask.unsqueeze(1), dim=-1), dim=-1
        )
        out = out_sum / self.normalization_constant
        assert len(out.shape) == 2, f"Wrong output shape: {out.shape}"
        assert out.shape[0] == image.shape[0], (
            f"Wrong output batch size: {out.shape[0]} v/s {image.shape[0]}"
        )
        assert out.shape[1] == self.conv.out_channels, (
            f"Wrong output dimensions: {out.shape[1]}"
        )
        return out

    def inputs(self) -> tuple[str, ...]:
        return (IMAGE_INPUT, FOCUS_POINT)

    def outputs(self) -> tuple[str, ...]:
        return (RETINA_OUTPUT,)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the sensory input given the current focus.

        Args:
            image: (B, H, W) input tensor
            focus: (B, 2) focus tensor

        Returns:
            (B, F) sensory input tensor with a number of filters
        """
        image = inputs[IMAGE_INPUT]
        focus = inputs[FOCUS_POINT]
        return {
            RETINA_OUTPUT: self.layer_norm(self.filter_output(image=image, focus=focus))
        }
