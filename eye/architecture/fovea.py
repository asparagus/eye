"""Fovea module for focused, high-resolution vision."""

import torch
from jigsaw.piece import Piece

from eye.architecture.names import (
    ATTENTION,
    FOVEAL_OUTPUT,
    IMAGE_INPUT,
)


class FoveaModule(Piece):
    """Simplified model of the fovea.

    Implements focused, high-resolution vision by attending to a focal point
    within the image using attention and convolutional filters.
    """

    def __init__(
        self,
        dims: tuple[int, int],
        num_filters: int,
        kernel_size: int,
        frozen: bool = True,
    ):
        """Initialize the fovea module.

        Args:
            dims: The dimensions of the input image.
            num_filters: The number of filters to use in the convolutional layer.
            kernel_size: Size of the filters in the convolutional layer.
            frozen: Whether the filters are frozen (and do not learn)
        """
        super().__init__(piece_type="module")
        self.dims = dims
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

        # Pre-compute coordinate vectors for efficiency
        self.x_coords = torch.arange(dims[1], dtype=torch.float32)  # Width
        self.y_coords = torch.arange(dims[0], dtype=torch.float32)  # Height

        self.layer_norm = torch.nn.LayerNorm(num_filters)

    def filter_output(
        self,
        image: torch.Tensor,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the sensory input given the current attention.

        Args:
        image: (B, H, W) input tensor
        attention: (B, H, W) attention tensor

        Returns:
        (B, F) sensory input tensor with a number of filters
        """
        assert len(image.shape) == 3, f"Wrong image shape: {image.shape}"
        assert len(attention.shape) == 3, f"Wrong attention shape: {attention.shape}"
        assert image.shape[0] == attention.shape[0], (
            f"Mismatch in batch size: {image.shape[0]} v/s {attention.shape[0]}"
        )
        image_with_filters = image.unsqueeze(1)
        raw_filter_output = self.conv(image_with_filters)
        # Apply mask to each filter channel and sum spatially
        masked_output = raw_filter_output * attention.unsqueeze(1)  # (B, F, H, W)
        out = torch.sum(masked_output, dim=(2, 3))  # Sum over H, W dimensions -> (B, F)
        assert len(out.shape) == 2, f"Wrong output shape: {out.shape}"
        assert out.shape[0] == image.shape[0], (
            f"Wrong output batch size: {out.shape[0]} v/s {image.shape[0]}"
        )
        assert out.shape[1] == self.conv.out_channels, (
            f"Wrong output dimensions: {out.shape[1]}"
        )
        return out

    def inputs(self) -> tuple[str, ...]:
        return (IMAGE_INPUT, ATTENTION)

    def outputs(self) -> tuple[str, ...]:
        return (FOVEAL_OUTPUT,)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the sensory input given the current attention.

        Args:
            image: (B, H, W) input tensor
            attention: (B, H, W) attention tensor

        Returns:
            (B, F) sensory input tensor with a number of filters
        """
        image = inputs[IMAGE_INPUT]
        attention = inputs[ATTENTION]
        return {
            FOVEAL_OUTPUT: self.layer_norm(
                self.filter_output(image=image, attention=attention)
            )
        }
