"""Module for the eye model."""

import logging
from typing import Literal

import torch
import torch.nn as nn
from jigsaw import Composite, Piece

from eye.architecture.names import (
    ATTENTION,
    ATTENTION_NEXT,
    ATTENTION_HISTORY,
    DOWNSAMPLED_IMAGE,
    IMAGE_INPUT,
    PERIPHERAL_OUTPUT,
    RETINA_OUTPUT,
    STATE_CELL,
    STATE,
    STATE_HISTORY,
)
from eye.architecture.motor import MotorModule
from eye.architecture.retina import RetinaModule


logger = logging.getLogger(__name__)


class RNNStateModule(Piece):
    """The module that produces the next eye state."""

    def __init__(self, num_filters: int, embedding_dimension: int):
        """Initialize the state module.

        Args:
            num_filters: Number of filters for the embedding networks.
            embedding_dimension: Dimension for the embedding networks.
        """
        super().__init__(piece_type="module")
        self.retina_embed = nn.Sequential(
            nn.Linear(num_filters, embedding_dimension),
            nn.ReLU(),
            nn.LayerNorm(embedding_dimension),
        )
        self.state_network = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.ReLU(),
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.ReLU(),
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.ReLU(),
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, embedding_dimension),
        )
        self.num_filters = num_filters
        self.embedding_dimension = embedding_dimension

    def inputs(self) -> tuple[str, ...]:
        return (RETINA_OUTPUT, PERIPHERAL_OUTPUT, STATE)

    def outputs(self) -> tuple[str, ...]:
        return (STATE,)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the new states given the input and the current states.

        Args:
            retina_output: (B, F) retina output tensor
            state: (B, F) previous state tensor

        Returns:
            (B, F) new state tensor
        """
        state = inputs[STATE]
        retina_output = inputs[RETINA_OUTPUT]
        batch_size = state.shape[0]
        assert state.shape == (batch_size, self.embedding_dimension)
        assert retina_output.shape == (batch_size, self.num_filters)
        retina_embedding = self.retina_embed(retina_output)
        new_state = self.state_network(retina_embedding)
        assert new_state.shape == (batch_size, self.embedding_dimension)
        return {STATE: new_state}


class LSTMStateModule(Piece):
    """The module that produces the next eye state."""

    def __init__(self, num_filters: int, embedding_dimension: int):
        """Initialize the state module.

        Args:
            num_filters: Number of filters for the embedding networks.
            embedding_dimension: Dimension for the embedding networks.
        """
        super().__init__(piece_type="module")
        self.retina_embedding_network = nn.Linear(num_filters, embedding_dimension)
        self.retina_norm = nn.LayerNorm(embedding_dimension)
        self.lstm = nn.LSTMCell(embedding_dimension, embedding_dimension)
        self.num_filters = num_filters
        self.embedding_dimension = embedding_dimension

    def inputs(self) -> tuple[str, ...]:
        return (RETINA_OUTPUT, STATE, STATE_CELL)

    def outputs(self) -> tuple[str, ...]:
        return (STATE, STATE_CELL)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the new LSTM states given the input and the current states.

        Args:
            retina_output: (B, F) retina output tensor
            state: (B, F) previous state tensor
            cell_state: (B, F) previous cell state tensor

        Returns:
            (B, F) new state tensor
            (B, F) new cell state tensor
        """
        state = inputs[STATE]
        cell_state = inputs[STATE_CELL]
        retina_output = inputs[RETINA_OUTPUT]
        batch_size = state.shape[0]
        assert state.shape == (batch_size, self.embedding_dimension)
        assert cell_state.shape == (batch_size, self.embedding_dimension)
        assert retina_output.shape == (batch_size, self.num_filters)
        retina_embedding = self.retina_norm(
            torch.relu(self.retina_embedding_network(retina_output))
        )
        new_state, new_cell = self.lstm(retina_embedding, (state, cell_state))
        assert new_state.shape == (batch_size, self.embedding_dimension)
        assert new_cell.shape == (batch_size, self.embedding_dimension)
        return {STATE: new_state, STATE_CELL: new_cell}


class SaccadeModule(Composite):
    """Module that performs saccades around the image.

    This module contains the retina and motor modules.
    It moves the focus point around.
    """

    def __init__(
        self,
        dims: tuple[int, int],
        num_filters: int,
        embedding_dimension: int,
        kernel_size: int,
        num_downsamples: int,
        state_module: Piece,
        retina_frozen: bool = True,
    ):
        retina = RetinaModule(
            dims=dims,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_downsamples=num_downsamples,
            frozen=retina_frozen,
        )
        motor = MotorModule(dims=dims, embedding_dimension=embedding_dimension)
        super().__init__(components=[retina, state_module, motor])
        self.retina = retina
        self.motor = motor


class EyeModule(Piece):
    """Module that mimics the eye."""

    def __init__(
        self,
        dims: tuple[int, int],
        num_filters: int,
        embedding_dimension: int,
        iterations: int,
        kernel_size: int,
        num_downsamples: int,
        recurrent_module: Literal["rnn", "lstm"],
        retina_frozen: bool = True,
    ):
        """Initialize the eye module."""
        super().__init__(piece_type="composite")
        self.num_filters = num_filters
        self.embedding_dimension = embedding_dimension
        state_module = (
            LSTMStateModule(
                num_filters=num_filters,
                embedding_dimension=embedding_dimension,
            )
            if recurrent_module == "lstm"
            else RNNStateModule(
                num_filters=num_filters,
                embedding_dimension=embedding_dimension,
            )
        )
        self.saccade = SaccadeModule(
            dims=dims,
            num_filters=num_filters,
            embedding_dimension=embedding_dimension,
            kernel_size=kernel_size,
            num_downsamples=num_downsamples,
            retina_frozen=retina_frozen,
            state_module=state_module,
        )
        self.iterations = iterations
        self.recurrent_module = recurrent_module

        # Learnable initial states
        self.initial_state = nn.Parameter(torch.zeros(embedding_dimension))
        if recurrent_module == "lstm":
            self.initial_cell_state = nn.Parameter(torch.zeros(embedding_dimension))

        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.initial_state.unsqueeze(0))
        if recurrent_module == "lstm":
            nn.init.xavier_uniform_(self.initial_cell_state.unsqueeze(0))

    def inputs(self) -> tuple[str, ...]:
        return (IMAGE_INPUT, ATTENTION)

    def outputs(self) -> tuple[str, ...]:
        return (STATE, ATTENTION_HISTORY)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the new attention given the input and the current attention.

        Args:
            image: (B, H, W) input image tensor
            current_attention: (B, H, W) attention mask tensor

        Returns:
            (B, F) last state values
            (B, I, F) state history
            (B, I, H, W) attention history
            (B, 1) accumulated motor loss
        """
        image = inputs[IMAGE_INPUT]
        current_attention = inputs[ATTENTION]
        batch_dimension = image.shape[0]
        assert len(image.shape) == 3
        assert len(current_attention.shape) == 3
        assert batch_dimension == current_attention.shape[0]

        state_history = torch.empty(
            (batch_dimension, self.iterations, self.embedding_dimension),
            device=image.device,
            dtype=image.dtype,
        )
        # Attention history stores the attention used for each iteration (not the output of the last iteration)
        attention_history = torch.empty(
            (
                batch_dimension,
                self.iterations,
                current_attention.shape[1],
                current_attention.shape[2],
            ),
            device=image.device,
            dtype=image.dtype,
        )
        # Store initial attention as first element
        attention_history[:, 0, :, :] = current_attention

        # Initialize states using learnable parameters
        state = self.initial_state.unsqueeze(0).expand(batch_dimension, -1).contiguous()
        cell_state = (
            self.initial_cell_state.unsqueeze(0)
            .expand(batch_dimension, -1)
            .contiguous()
            if self.recurrent_module == "lstm"
            else None
        )

        downsampled = self.saccade.retina.peripheral.downsample(image)
        for i in range(self.iterations):
            current_inputs = {
                IMAGE_INPUT: image,
                DOWNSAMPLED_IMAGE: downsampled,
                ATTENTION: current_attention,
                STATE: state,
                STATE_CELL: cell_state,
            }
            outputs = self.saccade(inputs=current_inputs)
            current_attention = outputs[ATTENTION_NEXT]
            state = outputs[STATE]
            cell_state = (
                outputs[STATE_CELL] if self.recurrent_module == "lstm" else None
            )
            state_history[:, i, :] = state
            # Only store attention for iterations 1 through iterations-1 (skip the last unused attention)
            if i < self.iterations - 1:
                attention_history[:, i + 1, :, :] = current_attention
        assert state_history.shape == (
            batch_dimension,
            self.iterations,
            self.embedding_dimension,
        )
        assert attention_history.shape == (
            batch_dimension,
            self.iterations,
            current_attention.shape[1],
            current_attention.shape[2],
        )
        return {
            STATE: state,
            STATE_HISTORY: state_history,
            ATTENTION_HISTORY: attention_history,
        }
