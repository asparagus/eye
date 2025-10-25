"""Motor module for the eye."""

import torch
from torch import nn
from jigsaw.piece import Piece

from eye.architecture.names import (
    ATTENTION,
    ATTENTION_NEXT,
    STATE,
)


class MotorModule(Piece):
    def __init__(self, dims: tuple[int, int], embedding_dimension: int):
        """Initialize the eye motor network.

        Args:
            dims: The dimensions of the input image.
            embedding_dimension: The dimension of the input to the motor.
        """
        super().__init__(piece_type="module")
        self.attention_predictor = nn.Linear(
            in_features=embedding_dimension, out_features=dims[0] * dims[1], bias=True
        )
        self.layer_norm = nn.LayerNorm(embedding_dimension)

        # Create coordinate grids for generating attention masks
        h_coords = torch.arange(dims[0]).float().unsqueeze(1).expand(dims)
        w_coords = torch.arange(dims[1]).float().unsqueeze(0).expand(dims)
        self.register_buffer("h_coords", h_coords)
        self.register_buffer("w_coords", w_coords)

    def inputs(self) -> tuple[str, ...]:
        return (STATE, ATTENTION)

    def outputs(self) -> tuple[str, ...]:
        return (ATTENTION_NEXT,)

    def compute_attention_center(self, attention: torch.Tensor) -> torch.Tensor:
        """Compute the weighted centroid of the attention mask.

        Args:
            attention: (B, H, W) attention mask

        Returns:
            (B, 2) attention center coordinates [h, w]
        """
        # Normalize attention to sum to 1
        attention_sum = attention.sum(dim=(-2, -1), keepdim=True)
        attention_norm = attention / (attention_sum + 1e-8)

        # Compute weighted average coordinates
        h_center = (attention_norm * self.h_coords).sum(dim=(-2, -1))
        w_center = (attention_norm * self.w_coords).sum(dim=(-2, -1))

        return torch.stack([h_center, w_center], dim=-1)

    def translate_attention(
        self, attention: torch.Tensor, translation: torch.Tensor
    ) -> torch.Tensor:
        """Translate attention masks by given offsets using grid_sample.

        Args:
            attention: (B, H, W) attention masks
            translation: (B, 2) translation offsets [dh, dw]

        Returns:
            (B, H, W) translated attention masks
        """
        import torch.nn.functional as F

        batch_size, H, W = attention.shape
        device = attention.device

        # Create normalized grid coordinates [-1, 1]
        h_coords = torch.linspace(-1, 1, H, device=device).unsqueeze(1).expand(H, W)
        w_coords = torch.linspace(-1, 1, W, device=device).unsqueeze(0).expand(H, W)

        # Create grid for all batch items
        grid = (
            torch.stack([w_coords, h_coords], dim=-1)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .clone()
        )

        # Convert translation to normalized coordinates and apply
        translation_norm = translation.clone()
        translation_norm[:, 0] = translation_norm[:, 0] * 2.0 / (H - 1)  # dh
        translation_norm[:, 1] = translation_norm[:, 1] * 2.0 / (W - 1)  # dw

        # Apply translation (subtract because grid_sample does inverse mapping)
        grid[:, :, :, 0] -= (
            translation_norm[:, 1].unsqueeze(-1).unsqueeze(-1)
        )  # w offset
        grid[:, :, :, 1] -= (
            translation_norm[:, 0].unsqueeze(-1).unsqueeze(-1)
        )  # h offset

        # Add channel dimension for grid_sample
        attention_4d = attention.unsqueeze(1)  # (B, 1, H, W)

        # Sample using bilinear interpolation
        translated_attention_4d = F.grid_sample(
            attention_4d,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # Remove channel dimension
        translated_attention = translated_attention_4d.squeeze(1)  # (B, H, W)

        return translated_attention

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute the new attention given the input and the current attention.

        Args:
            state: (B, F) input tensor
            current_attention: (B, H, W) attention tensor

        Returns:
            (B, H, W) new attention tensor
        """
        state = inputs[STATE]
        current_attention = inputs[ATTENTION]
        batch_size, H, W = current_attention.shape

        assert len(state.shape) == 2, f"Expected state shape (B, F), got {state.shape}"
        assert len(current_attention.shape) == 3, (
            f"Expected attention shape (B, H, W), got {current_attention.shape}"
        )
        assert batch_size == state.shape[0]

        # Step 1: Predict attention logits from state
        attention_logits = self.attention_predictor(self.layer_norm(state))  # (B, H*W)

        # Step 2: Apply softmax to get attention probabilities
        attention_probs = torch.softmax(attention_logits, dim=-1)  # (B, H*W)

        # Step 3: Reshape to attention mask
        new_attention = attention_probs.view(batch_size, H, W)  # (B, H, W)

        # Step 4: Compute current attention center
        current_center = self.compute_attention_center(current_attention)

        # Step 5: Translate the new mask relative to current attention center
        # This centers the new attention based on where the current attention is focused
        image_center = (
            torch.tensor([H // 2, W // 2], device=current_attention.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        translation = current_center - image_center

        # Apply translation to move the new attention relative to current focus
        new_attention = self.translate_attention(new_attention, translation)

        # Step 6: Renormalize (handles probability mass that falls outside boundaries)
        new_attention = new_attention / (
            new_attention.sum(dim=(-2, -1), keepdim=True) + 1e-8
        )

        assert new_attention.shape == current_attention.shape

        return {ATTENTION_NEXT: new_attention}
