"""Module for the eye model."""

import torch
import torch.nn as nn
from jigsaw import Composite, Piece

from eye.architecture.names import (
    FOCUS_HISTORY,
    FOCUS_NEXT,
    FOCUS_POINT,
    IMAGE_INPUT,
    MOTOR_LOSS,
    RETINA_OUTPUT,
    STATE_CELL,
    STATE,
    STATE_HISTORY,
)
from eye.architecture.motor import MotorModule
from eye.architecture.retina import RetinaModule


class StateModule(Piece):
    """The module that produces the next eye state."""

    def __init__(self, num_filters: int):
        """Initialize the state module.

        Args:
            num_filters: Number of filters for the embedding networks.
        """
        super().__init__(piece_type="module")
        self.focus_embedding_network = nn.Linear(2, num_filters)
        self.retina_embedding_network = nn.Linear(num_filters, num_filters)
        self.focus_norm = nn.LayerNorm(num_filters)
        self.retina_norm = nn.LayerNorm(num_filters)
        self.lstm = nn.LSTMCell(num_filters, num_filters)
        self.num_filters = num_filters

    def inputs(self) -> tuple[str, ...]:
        return (RETINA_OUTPUT, STATE, STATE_CELL, FOCUS_POINT)

    def outputs(self) -> tuple[str, ...]:
        return (STATE, STATE_CELL)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the new LSTM states given the input and the current states.

        Args:
            retina_output: (B, F) retina output tensor
            state: (B, F) previous hidden state tensor
            cell_state: (B, F) previous cell state tensor
            focus: (B, 2) current focus

        Returns:
            (B, F) new state tensor
            (B, F) new cell state tensor
        """
        state = inputs[STATE]
        cell_state = inputs[STATE_CELL]
        retina_output = inputs[RETINA_OUTPUT]
        focus = inputs[FOCUS_POINT]
        batch_size = state.shape[0]
        assert state.shape == (batch_size, self.num_filters)
        assert cell_state.shape == (batch_size, self.num_filters)
        assert retina_output.shape == (batch_size, self.num_filters)
        assert focus.shape == (batch_size, 2)
        focus_embedding = self.focus_norm(
            torch.relu(self.focus_embedding_network(focus))
        )
        retina_embedding = self.retina_norm(
            torch.relu(self.retina_embedding_network(retina_output))
        )
        new_state, new_cell = self.lstm(
            retina_embedding + focus_embedding, (state, cell_state)
        )
        assert new_state.shape == (batch_size, self.num_filters)
        assert new_cell.shape == (batch_size, self.num_filters)
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
        fovea_radius: float,
        retina_frozen: bool = True,
        motor_noise_std: float = 0.0,
    ):
        retina = RetinaModule(
            dims=dims,
            num_filters=num_filters,
            fovea_radius=fovea_radius,
            frozen=retina_frozen,
        )
        motor = MotorModule(
            dims=dims, num_filters=num_filters, noise_std=motor_noise_std
        )
        state_module = StateModule(num_filters=num_filters)
        super().__init__(components=[retina, state_module, motor])


class EyeModule(Piece):
    """Module that mimics the eye."""

    def __init__(
        self,
        dims: tuple[int, int],
        num_filters: int,
        iterations: int,
        fovea_radius: float,
        retina_frozen: bool = True,
        motor_noise_std: float = 0.0,
    ):
        """Initialize the eye module."""
        super().__init__(piece_type="composite")
        self.num_filters = num_filters
        self.saccade = SaccadeModule(
            dims=dims,
            num_filters=num_filters,
            fovea_radius=fovea_radius,
            retina_frozen=retina_frozen,
            motor_noise_std=motor_noise_std,
        )
        self.iterations = iterations

    def inputs(self) -> tuple[str, ...]:
        return (IMAGE_INPUT, FOCUS_POINT)

    def outputs(self) -> tuple[str, ...]:
        return (STATE, FOCUS_HISTORY, MOTOR_LOSS)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the new focus given the input and the current focus.

        Args:
            image: (B, H, W) input image tensor
            current_focus: (B, 2) focus tensor

        Returns:
            (B, F) last state values
            (B, I, F) state history
            (B, I, 2) focus history
            (B, 1) accumulated motor loss
        """
        image = inputs[IMAGE_INPUT]
        current_focus = inputs[FOCUS_POINT]
        batch_dimension = image.shape[0]
        assert len(image.shape) == 3
        assert len(current_focus.shape) == 2
        assert batch_dimension == current_focus.shape[0]
        focus_history = [current_focus]
        state_history: list[torch.Tensor] = []
        loss_history: list[torch.Tensor] = []

        # Initialize LSTM states to zeros
        state = torch.zeros(
            size=(batch_dimension, self.num_filters),
            device=image.device,
        )
        cell_state = torch.zeros(
            size=(batch_dimension, self.num_filters),
            device=image.device,
        )

        for _ in range(self.iterations):
            current_inputs = {
                IMAGE_INPUT: image,
                FOCUS_POINT: current_focus,
                STATE: state,
                STATE_CELL: cell_state,
            }
            outputs = self.saccade(inputs=current_inputs)
            current_focus = outputs[FOCUS_NEXT]
            state = outputs[STATE]
            cell_state = outputs[STATE_CELL]
            motor_loss = outputs[MOTOR_LOSS]
            state_history.append(state)
            focus_history.append(current_focus)
            loss_history.append(motor_loss)
        stacked_state_history = torch.stack(state_history, dim=1)
        stacked_focus_history = torch.stack(focus_history[:-1], dim=1)
        aggregated_motor_loss = torch.sum(
            torch.stack(loss_history, dim=0), dim=0, keepdim=False
        )
        assert stacked_state_history.shape == (
            batch_dimension,
            self.iterations,
            self.num_filters,
        )
        assert stacked_focus_history.shape == (batch_dimension, self.iterations, 2)
        assert aggregated_motor_loss.shape == (batch_dimension, 1)
        return {
            STATE: state,
            STATE_HISTORY: stacked_state_history,
            FOCUS_HISTORY: stacked_focus_history,
            MOTOR_LOSS: aggregated_motor_loss,
        }
