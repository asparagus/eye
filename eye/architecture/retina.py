"""Retina module for the eye - now delegates to fovea for backward compatibility."""

import torch
from torch import nn
from jigsaw import Composite, Piece

from eye.architecture.fovea import FoveaModule
from eye.architecture.names import (
    FOVEAL_OUTPUT,
    PERIPHERAL_OUTPUT,
    RETINA_OUTPUT,
)
from eye.architecture.peripheral import PeripheralModule


class RetinaAggregator(Piece):
    def __init__(self, num_filters: int):
        super().__init__(piece_type="module")
        self.foveal = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.LayerNorm(num_filters),
        )
        self.peripheral = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.LayerNorm(num_filters),
        )
        self.combination = nn.Linear(2 * num_filters, num_filters)

    def inputs(self) -> tuple[str, ...]:
        return (FOVEAL_OUTPUT, PERIPHERAL_OUTPUT)

    def outputs(self) -> tuple[str, ...]:
        return (RETINA_OUTPUT,)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        foveal_output = self.foveal(inputs[FOVEAL_OUTPUT])
        peripheral_output = self.peripheral(inputs[PERIPHERAL_OUTPUT])
        combined = torch.cat([foveal_output, peripheral_output], dim=-1)
        return {RETINA_OUTPUT: self.combination(combined)}


class RetinaModule(Composite):
    """Simplified model of the retina.

    Will attend to a focal point within the image, given as a position to its
    forward function (focus).
    """

    def __init__(
        self,
        dims: tuple[int, int],
        num_filters: int,
        kernel_size: int,
        num_downsamples: int,
        frozen: bool = True,
    ):
        """Initialize the retina module.

        Args:
            dims: The dimensions of the input image.
            num_filters: The number of filters to use in the convolutional layer.
            kernel_size: Size of the convolutional layer kernel.
            num_downsamples: Number of downsamples used by peripheral vision.
            frozen: Whether the filters are frozen (and do not learn)
        """
        fovea = FoveaModule(
            dims=dims,
            num_filters=num_filters,
            kernel_size=kernel_size,
            frozen=frozen,
        )
        peripheral = PeripheralModule(
            dims=dims,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_downsamples=num_downsamples,
        )
        agg = RetinaAggregator(num_filters=num_filters)
        super().__init__(components=[fovea, peripheral, agg])
        self.peripheral = peripheral
