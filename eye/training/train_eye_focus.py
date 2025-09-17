import logging
import pathlib

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from eye import CONFIG
from eye.architecture.module import EyeModule
from eye.architecture.names import (
    FOCUS_POINT,
    FOCUS_HISTORY,
    IMAGE_INPUT,
    MOTOR_LOSS,
    STATE,
    STATE_HISTORY,
)


logger = logging.getLogger("eye.training.train_eye_focus")


FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Training constants
DEFAULT_BATCH_SIZE = 1024
DEFAULT_NUM_WORKERS = 6  # Increased for better data loading performance


def get_device() -> torch.device:
    """Get the appropriate device for PyTorch operations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def get_transforms() -> transforms.Compose:
    """Get the standard transforms for Fashion-MNIST."""
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )


def load_datasets(
    transform: transforms.Compose,
) -> tuple[torchvision.datasets.FashionMNIST, torchvision.datasets.FashionMNIST]:
    """Load Fashion-MNIST train and test datasets."""
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def plot_focus_visualization(
    ax: plt.Axes,
    img: np.ndarray,
    focus_points: np.ndarray,
    radius: float,
    test_label: int,
    prediction: int,
) -> None:
    """Plot focus trajectory visualization on a single axis."""
    ax.imshow(img, cmap="gray", extent=[0, 28, 28, 0])
    ax.set_title(
        f"True: {FASHION_MNIST_CLASSES[test_label]}\nPred: {FASHION_MNIST_CLASSES[prediction]}",
        fontsize=10,
    )

    # Plot trajectory path
    ax.plot(focus_points[:, 1], focus_points[:, 0], "r-", linewidth=2, alpha=0.7)

    # Plot focus points with different colors for different iterations
    colors = plt.cm.viridis(np.linspace(0, 1, len(focus_points)))
    for j, (focus_point, color) in enumerate(zip(focus_points, colors)):
        # Draw fovea circle
        circle = patches.Circle(
            (focus_point[1], focus_point[0]),
            radius,
            fill=False,
            edgecolor=color,
            linewidth=2,
            alpha=0.8,
        )
        ax.add_patch(circle)

        # Mark focus point
        ax.plot(
            focus_point[1],
            focus_point[0],
            "o",
            color=color,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1,
        )

        # Label iterations
        if j < len(focus_points) - 1:  # Don't label the last point to avoid clutter
            ax.text(
                focus_point[1] + 1,
                focus_point[0] - 1,
                str(j),
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    ax.set_xlim(0, 28)
    ax.set_ylim(28, 0)
    ax.set_aspect("equal")


def plot_focus_trajectory_over_time(
    ax: plt.Axes,
    focus_points: np.ndarray,
    classification_probs: np.ndarray,
    true_label: int,
) -> None:
    """Plot focus position over iterations."""
    ax.plot(
        range(len(focus_points)),
        focus_points[:, 0],
        "r-",
        marker="o",
        label="Y (row)",
    )
    ax.plot(
        range(len(focus_points)),
        focus_points[:, 1],
        "b-",
        marker="s",
        label="X (col)",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Focus Position")
    ax.set_title("Focus Movement Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, len(focus_points) - 0.9)
    ax.set_ylim(0, 28)


def plot_classification_over_time(
    ax: plt.Axes, classification_probs: np.ndarray, true_label: int
) -> None:
    """Plot classification probability for true class over iterations."""
    true_class_probs = classification_probs[:, true_label]
    predicted_classes = np.argmax(classification_probs, axis=1)

    # Plot probability of true class
    ax.plot(
        range(len(true_class_probs)),
        true_class_probs,
        "g-",
        marker="o",
        linewidth=2,
        label=f"P({FASHION_MNIST_CLASSES[true_label]})",
    )

    # Mark correct predictions
    correct_predictions = predicted_classes == true_label
    ax.scatter(
        np.where(correct_predictions)[0],
        true_class_probs[correct_predictions],
        color="green",
        s=50,
        marker="^",
        alpha=0.8,
        label="Correct",
    )

    # Mark incorrect predictions
    incorrect_predictions = predicted_classes != true_label
    ax.scatter(
        np.where(incorrect_predictions)[0],
        true_class_probs[incorrect_predictions],
        color="red",
        s=50,
        marker="v",
        alpha=0.8,
        label="Incorrect",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Probability")
    ax.set_title("Classification Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, len(true_class_probs) - 0.9)
    ax.set_ylim(0, 1)


class TrackingEyeFashionMNISTNet(LightningModule):
    """Modified network that tracks eye focus movements using PyTorch Lightning."""

    def __init__(
        self,
        num_filters: int,
        iterations: int,
        fovea_radius: float,
        learning_rate: float,
        retina_frozen: bool = True,
        motor_noise_std: float = 0.0,
        dims: tuple[int, int] = (28, 28),
        num_classes: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.eye = EyeModule(
            dims=dims,
            num_filters=num_filters,
            iterations=iterations,
            fovea_radius=fovea_radius,
            retina_frozen=retina_frozen,
            motor_noise_std=motor_noise_std,
        )
        self.iterations = iterations
        self.fovea_radius = fovea_radius
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_filters),
            nn.Linear(num_filters, num_filters, bias=True),
            nn.ReLU(),
            nn.LayerNorm(num_filters),
            nn.Linear(num_filters, num_filters, bias=True),
            nn.ReLU(),
            nn.LayerNorm(num_filters),
            nn.Linear(num_filters, num_classes, bias=True),
        )
        self.criterion = nn.CrossEntropyLoss()

    def get_training_progress(self) -> float:
        """Get training progress as a value between 0 and 1.

        Returns:
            Progress value where 0 is start of training, 1 is end of training.
        """
        if not self.training or not hasattr(self, "trainer") or self.trainer is None:
            return 1.0  # Default to full randomness during inference

        current_epoch = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs

        if max_epochs is None or max_epochs <= 0:
            return 1.0

        return min(current_epoch / max_epochs, 1.0)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        batch_size = x.shape[0]
        # Curriculum learning: start from center, gradually increase randomness
        if self.training:
            progress = self.get_training_progress()
        else:
            progress = 0.0
        center = torch.tensor([13.5, 13.5], device=x.device).expand(batch_size, 2)

        if progress < 1.0:
            # Random offset that grows from 0 to full range over training
            max_offset = (
                13.5 * progress
            )  # Start at center (0 offset), grow to full range
            random_offset = (
                (torch.rand(batch_size, 2, device=x.device) - 0.5) * 2 * max_offset
            )
            initial_focus = center + random_offset
            # Clamp to valid range [0, 27]
            initial_focus = torch.clamp(initial_focus, 0.0, 27.0)
        else:
            # Full random after training is complete
            initial_focus = torch.rand(batch_size, 2, device=x.device) * 27.0

        results = self.eye.forward({IMAGE_INPUT: x, FOCUS_POINT: initial_focus})
        state = results[STATE]
        state_history = results[STATE_HISTORY]
        focus_trajectory = results[FOCUS_HISTORY]
        motor_loss = results[MOTOR_LOSS]
        output = self.classifier(state)
        output_history = self.classifier(state_history)

        return output, focus_trajectory, output_history, motor_loss

    def batch_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = batch
        outputs, _, output_history, motor_loss = self.forward(images)
        batch_size = images.shape[0]
        assert output_history.shape == (batch_size, self.iterations, 10)

        # Train all time steps to predict the correct class
        output_history_flat = output_history.view(-1, 10)
        labels_expanded = labels.repeat_interleave(self.iterations)
        loss = self.criterion(output_history_flat, labels_expanded) + motor_loss.mean()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean()
        return loss, accuracy

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, accuracy = self.batch_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)

        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Log gradient histograms to TensorBoard."""
        if self.trainer.logger:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.logger.experiment.add_histogram(
                        f"gradients/{name}", param.grad, self.global_step
                    )

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, accuracy = self.batch_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(
        self,
    ) -> dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler]:
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.95)
        return {
            "optimizer": optim,
            "lr_scheduler": scheduler,
        }


def visualize_eye_focus_movements(
    model: TrackingEyeFashionMNISTNet, output_path: str, num_examples: int = 16
):
    """Create visualizations of eye focus movements for a given model.

    Args:
        model: Pre-loaded TrackingEyeFashionMNISTNet model
        output_path: Path where the visualization will be saved
        num_examples: Number of examples to visualize (default: 16)
    """
    device = next(model.parameters()).device
    transform = get_transforms()

    # Load test dataset for visualization
    _, test_dataset = load_datasets(transform)

    half_examples = num_examples // 2

    # First half: consistent examples (same indices each time for tracking progress)
    consistent_indices = list(range(half_examples))
    consistent_loader = DataLoader(
        torch.utils.data.Subset(test_dataset, consistent_indices),
        batch_size=half_examples,
        shuffle=False,
    )
    consistent_images, consistent_labels = next(iter(consistent_loader))

    # Second half: find incorrect predictions from a larger sample
    large_batch_size = min(200, len(test_dataset) - half_examples)
    remaining_indices = list(range(half_examples, half_examples + large_batch_size))
    remaining_loader = DataLoader(
        torch.utils.data.Subset(test_dataset, remaining_indices),
        batch_size=large_batch_size,
        shuffle=False,
    )
    remaining_images, remaining_labels = next(iter(remaining_loader))

    # Move to device
    consistent_images = consistent_images.to(device)
    consistent_labels = consistent_labels.to(device)
    remaining_images = remaining_images.to(device)
    remaining_labels = remaining_labels.to(device)

    model.eval()
    with torch.no_grad():
        # Get predictions for remaining examples to find incorrect ones
        remaining_outputs, remaining_trajectories, _, _ = model(remaining_images)
        remaining_predictions = torch.max(remaining_outputs, 1)[1]

        # Find incorrect predictions
        incorrect_mask = remaining_predictions != remaining_labels
        incorrect_indices = torch.where(incorrect_mask)[0]

        # Select incorrect examples for second half
        remaining_slots = num_examples - half_examples
        if len(incorrect_indices) >= remaining_slots:
            selected_incorrect = incorrect_indices[:remaining_slots]
        else:
            # If not enough incorrect, just use what we have
            selected_incorrect = incorrect_indices

        # Combine consistent and incorrect examples
        all_images = torch.cat(
            [consistent_images, remaining_images[selected_incorrect]]
        )
        all_labels = torch.cat(
            [consistent_labels, remaining_labels[selected_incorrect]]
        )

        # Get final predictions and trajectories for visualization
        all_outputs, all_trajectories, all_output_history, _ = model(all_images)
        all_predictions = torch.max(all_outputs, 1)[1]

        # Convert output history to probabilities for classification over time
        all_classification_history = torch.softmax(all_output_history, dim=-1)

    # Create visualization: 3 rows - focus plots, classification plots, error examples
    _fig, axes = plt.subplots(3, half_examples, figsize=(4 * half_examples, 12))
    if half_examples == 1:
        axes = axes.reshape(3, 1)

    # Top row: Stable/consistent examples
    for i in range(half_examples):
        # Original image
        img = all_images[i].cpu().numpy()
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = img.squeeze(0)

        # Denormalize image for display
        img = img * 0.5 + 0.5

        # Plot focus trajectory
        trajectory = all_trajectories[i].cpu().numpy()

        plot_focus_visualization(
            axes[0, i],
            img,
            trajectory,
            model.fovea_radius,
            all_labels[i],
            all_predictions[i],
        )

        # Second row: Classification over time for consistent examples
        classification_probs = all_classification_history[i].cpu().numpy()
        plot_classification_over_time(
            axes[1, i],
            classification_probs,
            all_labels[i].item(),
        )

    # Third row: Error examples
    num_errors = len(selected_incorrect)
    for i in range(half_examples):
        if i < num_errors:
            error_idx = half_examples + i
            # Original image
            img = all_images[error_idx].cpu().numpy()
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img.squeeze(0)

            # Denormalize image for display
            img = img * 0.5 + 0.5

            # Plot focus trajectory
            trajectory = all_trajectories[error_idx].cpu().numpy()

            plot_focus_visualization(
                axes[2, i],
                img,
                trajectory,
                model.fovea_radius,
                all_labels[error_idx],
                all_predictions[error_idx],
            )
        else:
            # Hide axis if no error example available
            axes[2, i].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save visualization
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    consistent_correct = (
        (all_predictions[:half_examples] == all_labels[:half_examples]).sum().item()
    )
    incorrect_count = (
        len(selected_incorrect)
        if len(incorrect_indices) >= remaining_slots
        else len(incorrect_indices)
    )
    logger.info(
        f"Visualization saved as {output_path} ({consistent_correct}/{half_examples} consistent examples correct, {incorrect_count} incorrect examples shown)"
    )


class VisualizationCallback(Callback):
    """Custom callback to create visualizations during training."""

    def __init__(
        self,
        dirpath: str,
        filename: str,
        every_n_epochs: int = 2,
        num_examples: int = 16,
    ):
        self.dirpath = dirpath
        self.filename = filename
        self.every_n_epochs = every_n_epochs
        self.num_examples = num_examples

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: TrackingEyeFashionMNISTNet
    ) -> None:
        current_epoch = trainer.current_epoch

        # Create visualization at specified intervals or at the end
        is_last = (current_epoch + 1) == trainer.max_epochs
        if current_epoch % self.every_n_epochs == 0 or is_last:
            logger.info(f"Creating visualization for epoch {current_epoch}...")

            # Format filename with epoch
            formatted_filename = self.filename.format(epoch=current_epoch)
            output_path = pathlib.Path(self.dirpath, formatted_filename).with_suffix(
                ".png"
            )

            visualize_eye_focus_movements(
                pl_module, output_path, num_examples=self.num_examples
            )


def train(
    max_epochs: int,
    num_filters: int,
    iterations: int,
    fovea_radius: float,
    learning_rate: float,
    retina_frozen: bool,
    motor_noise_std: float = 0.0,
):
    transform = get_transforms()
    train_dataset, test_dataset = load_datasets(transform)

    # Split test dataset for validation
    val_size = len(test_dataset) // 2
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(
        test_dataset, [val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4,  # Prefetch more batches per worker
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Create tracking model
    model = TrackingEyeFashionMNISTNet(
        num_filters=num_filters,
        iterations=iterations,
        fovea_radius=fovea_radius,
        learning_rate=learning_rate,
        retina_frozen=retina_frozen,
        motor_noise_std=motor_noise_std,
    )

    # Compile model for PyTorch 2.0 speedup (requires CUDA capability >= 7.0)
    try:
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()
            if device_capability[0] >= 7:  # Major version >= 7
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled with PyTorch 2.0 for additional speedup")
            else:
                logger.info(
                    f"GPU CUDA capability {device_capability[0]}.{device_capability[1]} < 7.0, skipping compilation"
                )
        else:
            logger.info("No CUDA available, skipping compilation")
    except Exception as e:
        logger.warning(f"Could not compile model: {e}")
        logger.info("Continuing without compilation")

    checkpoint_callback = ModelCheckpoint(
        dirpath="saved/",
        filename="eye_model_epoch_{epoch}",
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1,
    )

    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")

    visualization_callback = VisualizationCallback(
        dirpath="visualizations/", filename="eye_model_epoch_{epoch}", every_n_epochs=2
    )

    tb_logger = TensorBoardLogger(
        CONFIG.TENSORBOARD_LOGS, name=CONFIG.TENSORBOARD_PROJECT_NAME
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, visualization_callback, learning_rate_monitor],
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        log_every_n_steps=50,
        logger=[tb_logger],
        precision="16-mixed",  # Mixed precision for ~2x speedup
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    logger.info("Training completed!")


if __name__ == "__main__":
    train(
        max_epochs=50,
        num_filters=96,  # Reduced from 128 for faster training
        iterations=12,  # Reduced from 15 for faster iterations
        fovea_radius=3.0,
        learning_rate=8e-6,  # Slightly higher since we have fewer params
        retina_frozen=True,
        motor_noise_std=0.5,
    )
