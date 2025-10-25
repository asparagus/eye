"""Peripheral module for broader, lower-resolution vision."""

import torch
from jigsaw.piece import Piece

from eye.architecture.names import (
    ATTENTION,
    DOWNSAMPLED_IMAGE,
    PERIPHERAL_OUTPUT,
)


class PeripheralModule(Piece):
    """Focus-aware peripheral vision with multi-scale pooling.

    First applies pooling to reduce resolution, then extracts features
    using broader gaussian attention centered on the focus point.
    Provides contextual information at lower resolution than foveal vision.
    """

    def __init__(
        self,
        dims: tuple[int, int],
        kernel_size: int,
        num_filters: int,
        num_downsamples: int,
    ):
        """Initialize the peripheral module.

        Args:
            dims: The dimensions of the input image.
            kernel_size: The size of the kernels for feature extraction.
            num_filters: The number of filters to use for feature extraction.
            num_downsamples: The number of downsamplings to perform.
        """
        super().__init__(piece_type="module")
        self.dims = dims
        self.num_filters = num_filters
        self.num_downsamples = num_downsamples

        self.downsampler = torch.nn.AvgPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2, ceil_mode=False
        )

        # Feature extraction on pooled representation
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    dilation=1,
                    bias=False,
                )
                for _ in range(num_downsamples)
            ]
        )

        self.layer_norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(num_filters) for _ in range(num_downsamples)]
        )

        self.merge = torch.nn.Linear(num_filters * num_downsamples, num_filters)
        self.merge_norm = torch.nn.LayerNorm(num_filters)

    def downsample(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Downsample the given image."""
        assert len(image.shape) == 3, f"Wrong image shape: {image.shape}"
        assert image.shape[1:] == self.dims, f"Wrong image shape: {image.shape}"
        downsampled: list[torch.Tensor] = []
        current = image.unsqueeze(1)
        for _ in range(self.num_downsamples):
            current = self.downsampler(current)
            assert current.shape == (image.shape[0], 1, *self.dims), (
                f"Wrong downsampled dimensions: {current.shape}"
            )
            downsampled.append(current)
        return downsampled

    def inputs(self) -> tuple[str, ...]:
        return (DOWNSAMPLED_IMAGE, ATTENTION)

    def outputs(self) -> tuple[str, ...]:
        return (PERIPHERAL_OUTPUT,)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute attention-aware peripheral vision with pooling and feature extraction.

        Args:
            downsampled: list of D (B, H, W) input tensors
            attention: (B, H, W) attention tensor

        Returns:
            (B, F x D) peripheral sensory input tensor
        """
        downsampled = inputs[DOWNSAMPLED_IMAGE]
        attention = inputs[ATTENTION]

        assert len(downsampled) == self.num_downsamples, (
            f"Unexpected number of downsamples {len(downsampled)}"
        )
        assert len(attention.shape) == 3, f"Wrong attention shape: {attention.shape}"

        attention = attention.unsqueeze(1)

        outs: list[torch.Tensor] = []
        for spl, conv, norm in zip(downsampled, self.convs, self.layer_norms):
            filter_out = conv(spl)  # [batch, channels, H, W]
            weighted_output = filter_out * attention  # [batch, channels, H, W]
            summed_output = weighted_output.sum(dim=(-2, -1))  # [batch, channels]
            outs.append(norm(summed_output))

        merged = self.merge(torch.cat(outs, dim=-1))

        return {PERIPHERAL_OUTPUT: self.merge_norm(merged)}
