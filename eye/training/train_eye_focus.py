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
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from eye import CONFIG
from eye.architecture.module import EyeModule
from eye.architecture.names import (
    EMBEDDING,
    EMBEDDING_HISTORY,
    FOCUS_POINT,
    FOCUS_HISTORY,
    IMAGE_INPUT,
    MOTOR_LOSS,
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


def get_device():
    """Get the appropriate device for PyTorch operations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def get_transforms():
    """Get the standard transforms for Fashion-MNIST."""
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )


def load_datasets(transform):
    """Load Fashion-MNIST train and test datasets."""
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def plot_focus_visualization(ax, img, focus_points, radius, test_label, prediction):
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


def plot_focus_trajectory_over_time(ax, focus_points):
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


class TrackingEyeFashionMNISTNet(LightningModule):
    """Modified network that tracks eye focus movements using PyTorch Lightning."""

    def __init__(
        self,
        num_filters: int,
        iterations: int,
        fovea_radius: float,
        learning_rate: float,
        retina_frozen: bool = True,
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
        )
        self.fovea_radius = fovea_radius
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        batch_size = x.shape[0]
        initial_focus = torch.tensor([[13.5, 13.5]], device=x.device).repeat(
            batch_size, 1
        )

        results = self.eye.forward({IMAGE_INPUT: x, FOCUS_POINT: initial_focus})
        eye_features = results[EMBEDDING]
        embedding_history = results[EMBEDDING_HISTORY]
        focus_trajectory = results[FOCUS_HISTORY]
        motor_loss = results[MOTOR_LOSS]
        output = self.classifier(eye_features)

        return output, focus_trajectory, embedding_history, motor_loss

    def batch_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = batch
        outputs, _, _, motor_loss = self.forward(images)
        loss = self.criterion(outputs, labels) + motor_loss.mean()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean()
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.batch_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.batch_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def visualize_eye_focus_movements(
    model: TrackingEyeFashionMNISTNet, output_path: str, num_examples: int = 16
):
    """Create visualizations of eye focus movements for a given model.

    Args:
        model: Pre-loaded TrackingEyeFashionMNISTNet model
        output_path: Path where the visualization will be saved
        num_examples: Number of examples to visualize (default: 16)
        title: Title for the visualization (default: "Eye Focus Movements")
    """
    device = next(model.parameters()).device
    transform = get_transforms()

    # Load test dataset for visualization
    _, test_dataset = load_datasets(transform)
    test_loader = DataLoader(test_dataset, batch_size=num_examples, shuffle=False)

    # Get a batch of test images
    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    model.eval()
    with torch.no_grad():
        outputs, focus_trajectories, _embedding_history, _motor_loss = model(
            test_images
        )
        predictions = torch.max(outputs, 1)[1]

    # Create visualization
    _fig, axes = plt.subplots(2, num_examples, figsize=(4 * num_examples, 8))
    if num_examples == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_examples):
        # Original image
        img = test_images[i].cpu().numpy()
        if len(img.shape) == 3 and img.shape[0] == 1:
            img = img.squeeze(0)

        # Denormalize image for display
        img = img * 0.5 + 0.5

        # Plot focus trajectory
        trajectory = focus_trajectories[i].cpu().numpy()

        # Top row: Original image with focus trajectory
        plot_focus_visualization(
            axes[0, i],
            img,
            trajectory,
            model.fovea_radius,
            test_labels[i],
            predictions[i],
        )

        # Bottom row: Focus trajectory over iterations
        plot_focus_trajectory_over_time(axes[1, i], trajectory)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save visualization
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Visualization saved as {output_path}")


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

    def on_train_epoch_end(self, trainer, pl_module):
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
        train_dataset, batch_size=256, shuffle=True, num_workers=7
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=7)

    # Create tracking model
    model = TrackingEyeFashionMNISTNet(
        num_filters=num_filters,
        iterations=iterations,
        fovea_radius=fovea_radius,
        learning_rate=learning_rate,
        retina_frozen=retina_frozen,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="saved/",
        filename="eye_model_epoch_{epoch}",
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1,
    )

    visualization_callback = VisualizationCallback(
        dirpath="visualizations/", filename="eye_model_epoch_{epoch}", every_n_epochs=2
    )

    # Create trainer
    mlf_logger = MLFlowLogger(
        experiment_name=CONFIG.MLFLOW_EXPERIMENT_NAME,
        tracking_uri=CONFIG.MLFLOW_TRACKING_URI,
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, visualization_callback],
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        log_every_n_steps=50,
        logger=mlf_logger,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    logger.info("Training completed!")


if __name__ == "__main__":
    train(
        max_epochs=50,
        num_filters=128,
        iterations=15,
        fovea_radius=3.0,
        learning_rate=0.0001,
        retina_frozen=True,
    )
