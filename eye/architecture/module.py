"""Module for the eye model."""

import torch
import torch.nn as nn
from jigsaw import Composite, Piece

from eye.architecture.names import (
    EMBEDDING,
    EMBEDDING_HISTORY,
    FOCUS_HISTORY,
    FOCUS_NEXT,
    FOCUS_POINT,
    IMAGE_INPUT,
    MOTOR_LOSS,
    RETINA_OUTPUT,
)
from eye.architecture.motor import MotorModule
from eye.architecture.retina import RetinaModule


class EmbeddingModule(Piece):
    """The module that produces the next eye embedding."""

    def __init__(self, num_filters: int):
        super().__init__(piece_type="module")
        self.focus_embedding_network = nn.Linear(2, num_filters)
        self.retina_embedding_network = nn.Linear(num_filters, num_filters)
        self.hidden_network = nn.Linear(num_filters, num_filters)
        self.embedding_network = nn.Linear(num_filters, num_filters)

    def inputs(self) -> tuple[str, ...]:
        return (RETINA_OUTPUT, EMBEDDING, FOCUS_POINT)

    def outputs(self) -> tuple[str, ...]:
        return (EMBEDDING,)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the new focus given the input and the current focus.

        Args:
            retina_output: (B, F) retina output tensor
            embedding_input: (B, F) previous embedding tensor
            focus: (B, 2) current focus

        Returns:
            (B, F) new embedding tensor
        """
        embedding = inputs[EMBEDDING]
        retina_output = inputs[RETINA_OUTPUT]
        focus = inputs[FOCUS_POINT]
        focus_embedding = torch.relu(self.focus_embedding_network(focus))
        retina_embedding = torch.relu(self.retina_embedding_network(retina_output))
        hidden_embedding = torch.relu(
            self.hidden_network(focus_embedding + retina_embedding)
        )
        embedding = torch.relu(embedding + self.embedding_network(hidden_embedding))
        return {EMBEDDING: embedding}


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
    ):
        retina = RetinaModule(
            dims=dims,
            num_filters=num_filters,
            fovea_radius=fovea_radius,
            frozen=retina_frozen,
        )
        motor = MotorModule(dims=dims, num_filters=num_filters)
        embedding_module = EmbeddingModule(num_filters=num_filters)
        super().__init__(components=[retina, embedding_module, motor])


class EyeModule(Piece):
    """Module that mimics the eye."""

    def __init__(
        self,
        dims: tuple[int, int],
        num_filters: int,
        iterations: int,
        fovea_radius: float,
        retina_frozen: bool = True,
    ):
        """Initialize the eye module."""
        super().__init__(piece_type="composite")
        self.num_filters = num_filters
        self.saccade = SaccadeModule(
            dims=dims,
            num_filters=num_filters,
            fovea_radius=fovea_radius,
            retina_frozen=retina_frozen,
        )
        self.iterations = iterations

    def inputs(self) -> tuple[str, ...]:
        return (IMAGE_INPUT, FOCUS_POINT)

    def outputs(self) -> tuple[str, ...]:
        return (EMBEDDING, FOCUS_HISTORY, MOTOR_LOSS)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the new focus given the input and the current focus.

        Args:
            embedding_input: (B, F) input tensor
            current_focus: (B, 2) focus tensor

        Returns:
            (B, F) last embedding values
            (B, I, F) embedding history
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
        embedding_history: list[torch.Tensor] = []
        loss_history: list[torch.Tensor] = []
        embedding = torch.zeros(
            size=(image.shape[0], self.num_filters),
            device=image.device,
        )
        for _ in range(self.iterations):
            current_inputs = {
                IMAGE_INPUT: image,
                FOCUS_POINT: current_focus,
                EMBEDDING: embedding,
            }
            outputs = self.saccade(inputs=current_inputs)
            current_focus = outputs[FOCUS_NEXT]
            embedding = outputs[EMBEDDING]
            motor_loss = outputs[MOTOR_LOSS]
            embedding_history.append(embedding)
            focus_history.append(current_focus)
            loss_history.append(motor_loss)
        stacked_embedding_history = torch.stack(embedding_history, dim=1)
        stacked_focus_history = torch.stack(focus_history[:-1], dim=1)
        aggregated_motor_loss = torch.sum(
            torch.stack(loss_history, dim=0), dim=0, keepdim=False
        )
        assert stacked_embedding_history.shape == (
            batch_dimension,
            self.iterations,
            self.num_filters,
        )
        assert stacked_focus_history.shape == (batch_dimension, self.iterations, 2)
        assert aggregated_motor_loss.shape == (batch_dimension, 1)
        return {
            EMBEDDING: embedding,
            EMBEDDING_HISTORY: stacked_embedding_history,
            FOCUS_HISTORY: stacked_focus_history,
            MOTOR_LOSS: aggregated_motor_loss,
        }
