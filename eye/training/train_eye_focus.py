import logging
from typing import Literal

import optuna
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import random_split, Subset
from torchvision.datasets import FashionMNIST
from lightning.pytorch import (
    LightningModule,
    LightningDataModule,
    Trainer,
)
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from eye.architecture.module import EyeModule
from eye.architecture.names import (
    ATTENTION,
    ATTENTION_HISTORY,
    IMAGE_INPUT,
    STATE,
    STATE_HISTORY,
)


logger = logging.getLogger("eye.training.train_eye_focus")


DEFAULT_NUM_WORKERS = 7


def get_device() -> torch.device:
    """Get the appropriate device for PyTorch operations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


class Classifier(nn.Module):
    def __init__(self, embedding_dimension: int, num_classes: int, hidden_layers: int):
        super().__init__()
        self.internal = nn.Sequential(
            *(
                layer
                for _ in range(hidden_layers)
                for layer in (
                    nn.LayerNorm(embedding_dimension),
                    nn.Linear(embedding_dimension, embedding_dimension, bias=True),
                    nn.ReLU(),
                )
            ),
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, num_classes, bias=True),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.internal(embedding)


class TrackingEyeFashionMNISTNet(LightningModule):
    """Modified network that tracks eye attention movements using PyTorch Lightning."""

    def __init__(
        self,
        num_filters: int,
        embedding_dimension: int,
        iterations: int,
        kernel_size: int,
        num_downsamples: int,
        learning_rate: float,
        learning_rate_decay: float,
        enable_curriculum_learning: bool,
        retina_frozen: bool,
        classifier_hidden_layers: int,
        optimizer_beta_1: float,
        optimizer_beta_2: float,
        optimizer_eps: float,
        recurrent_module: Literal["rnn", "lstm"],
        dims: tuple[int, int] = (28, 28),
        num_classes: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dims", "num_classes"])

        self.eye = EyeModule(
            dims=dims,
            num_filters=num_filters,
            embedding_dimension=embedding_dimension,
            iterations=iterations,
            kernel_size=kernel_size,
            num_downsamples=num_downsamples,
            recurrent_module=recurrent_module,
            retina_frozen=retina_frozen,
        )
        self.iterations = iterations
        self.kernel_size = kernel_size
        self.classifier = Classifier(
            embedding_dimension=embedding_dimension,
            num_classes=num_classes,
            hidden_layers=classifier_hidden_layers,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.enable_curriculum_learning = enable_curriculum_learning
        self.num_classes = num_classes
        self.optimizer_beta_1 = optimizer_beta_1
        self.optimizer_beta_2 = optimizer_beta_2
        self.optimizer_eps = optimizer_eps

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(x.shape) == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        batch_size = x.shape[0]
        height, width = x.shape[1], x.shape[2]

        # Create attention mask with 1.0 at center, 0.0 elsewhere
        initial_attention = torch.zeros(batch_size, height, width, device=x.device)
        center_h, center_w = height // 2, width // 2

        if self.enable_curriculum_learning:
            # Curriculum learning: start from center, gradually increase randomness
            if self.training:
                progress = self.get_training_progress()
            else:
                progress = 0.0

            if progress < 1.0:
                # Random offset that grows from 0 to full range over training
                max_offset_h = int(center_h * progress)
                max_offset_w = int(center_w * progress)

                for i in range(batch_size):
                    offset_h = torch.randint(
                        -max_offset_h, max_offset_h + 1, (1,), device=x.device
                    ).item()
                    offset_w = torch.randint(
                        -max_offset_w, max_offset_w + 1, (1,), device=x.device
                    ).item()

                    attention_h = torch.clamp(
                        torch.tensor(center_h + offset_h), 0, height - 1
                    ).item()
                    attention_w = torch.clamp(
                        torch.tensor(center_w + offset_w), 0, width - 1
                    ).item()

                    initial_attention[i, attention_h, attention_w] = 1.0
            else:
                # Full random after training is complete
                for i in range(batch_size):
                    random_h = torch.randint(0, height, (1,), device=x.device).item()
                    random_w = torch.randint(0, width, (1,), device=x.device).item()
                    initial_attention[i, random_h, random_w] = 1.0
        else:
            # Fixed center initialization without curriculum learning
            initial_attention[:, center_h, center_w] = 1.0

        results = self.eye.forward({IMAGE_INPUT: x, ATTENTION: initial_attention})
        state = results[STATE]
        state_history = results[STATE_HISTORY]
        attention_trajectory = results[ATTENTION_HISTORY]
        output = self.classifier(state)
        output_history = self.classifier(state_history)

        return output, attention_trajectory, output_history

    def batch_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = batch
        outputs, _, output_history = self.forward(images)
        batch_size = images.shape[0]
        assert output_history.shape == (batch_size, self.iterations, self.num_classes)

        output_history_flat = output_history.view(-1, self.num_classes)
        labels_expanded = labels.repeat_interleave(self.iterations)
        loss = self.criterion(output_history_flat, labels_expanded)

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

    # def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
    #     """Log gradient histograms to TensorBoard."""
    #     if self.trainer.logger:
    #         for name, param in self.named_parameters():
    #             if param.grad is not None and param.grad.size(dim=0) >= 2:
    #                 self.logger.experiment.add_histogram(
    #                     f"gradients/{name}", param.grad, self.global_step
    #                 )

    # def on_train_start(self):
    #     self.logger.log_hyperparams(self.hparams, {"hp_metric": 0})

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, accuracy = self.batch_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        self.log("hp_metric", accuracy)

        return loss

    def configure_optimizers(
        self,
    ) -> dict[str, torch.optim.Optimizer | torch.optim.lr_scheduler.LRScheduler]:
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.optimizer_beta_1, self.optimizer_beta_2),
            eps=self.optimizer_eps,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optim, gamma=self.learning_rate_decay
        )
        return {
            "optimizer": optim,
            "lr_scheduler": scheduler,
        }


class EarlyStopCallback(Callback):
    """Callback to prune unpromising trials during training using Optuna."""

    def __init__(self, trial: optuna.Trial):
        """Initialize the early stop callback.

        Args:
            trial: Optuna trial object for pruning decisions.
        """
        self.trial = trial

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: TrackingEyeFashionMNISTNet
    ) -> None:
        """Check if trial should be pruned after validation.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: The model being trained.

        Raises:
            optuna.TrialPruned: If the trial should be pruned based on validation accuracy.
        """
        current_epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        if "val_acc" in metrics:
            val_accuracy = metrics["val_acc"].item()

            self.trial.report(val_accuracy, current_epoch)

            if self.trial.should_prune():
                raise optuna.TrialPruned()


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object for suggesting hyperparameters.

    Returns:
        Validation accuracy as a float between 0 and 1.
    """
    num_filters = trial.suggest_categorical(
        "num_filters", [32, 64, 128]
    )  # , 256, 512])
    embedding_dimension = trial.suggest_categorical(
        "embedding_dimension", [32, 64]
    )  # , 128, 256, 512])
    kernel_size = trial.suggest_int("kernel_size", low=3, high=7, step=2)
    num_downsamples = trial.suggest_int("num_downsamples", low=1, high=3)
    learning_rate = trial.suggest_float("learning_rate", low=1e-8, high=1e-3, log=True)
    learning_rate_decay = trial.suggest_float(
        "learning_rate_decay", low=0.5, high=0.99999, log=True
    )
    enable_curriculum_learning = trial.suggest_categorical(
        name="enable_curriculum_learning", choices=[True, False]
    )
    classifier_hidden_layers = trial.suggest_int(
        "classifier_hidden_layers", low=1, high=5
    )
    optimizer_beta_1 = trial.suggest_float(
        "optimizer_beta_1", low=0.8, high=0.99999, log=True
    )
    optimizer_beta_2 = trial.suggest_float(
        "optimizer_beta_2", low=0.9, high=0.9999999, log=True
    )
    optimizer_eps = trial.suggest_float("optimizer_eps", low=1e-10, high=1e-6, log=True)
    return train(
        trial=trial,
        max_epochs=10,
        num_filters=num_filters,
        embedding_dimension=embedding_dimension,
        iterations=15,
        kernel_size=kernel_size,
        num_downsamples=num_downsamples,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        retina_frozen=True,
        enable_curriculum_learning=enable_curriculum_learning,
        recurrent_module="rnn",
        classifier_hidden_layers=classifier_hidden_layers,
        optimizer_beta_1=optimizer_beta_1,
        optimizer_beta_2=optimizer_beta_2,
        optimizer_eps=optimizer_eps,
    )


def load_datasets(
    transform: transforms.Compose,
) -> tuple[FashionMNIST, FashionMNIST]:
    """Load Fashion-MNIST train and test datasets."""
    train_dataset = FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


class MNISTDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for Fashion-MNIST dataset."""

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_fraction: float = 1.0,
    ):
        """Initialize the data module.

        Args:
            data_dir: Directory to store/load the dataset.
            batch_size: Batch size for dataloaders.
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            persistent_workers: Whether to keep workers alive between epochs.
            train_fraction: Fraction of training data to use (0.0 to 1.0).
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_fraction = train_fraction
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def prepare_data(self) -> None:
        """Download Fashion-MNIST dataset if not already present."""
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        """Set up datasets for different stages.

        Args:
            stage: Stage of training ('fit', 'test', or 'predict').
        """
        if stage == "fit":
            mnist = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            if self.train_fraction < 1.0:
                mnist, _ = random_split(
                    mnist,
                    [self.train_fraction, 1 - self.train_fraction],
                    generator=torch.Generator().manual_seed(42),
                )
            train, val = random_split(
                mnist, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )
            self.train: Subset[FashionMNIST] = train
            self.val: Subset[FashionMNIST] = val

        if stage == "test":
            self.test = FashionMNIST(
                self.data_dir, train=False, transform=self.transform
            )

        if stage == "predict":
            self.predict = FashionMNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def dataloader(
        self,
        data: FashionMNIST | Subset[FashionMNIST],
        shuffle: bool = False,
    ) -> DataLoader:
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
        )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with shuffling."""
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
            if self.num_workers > 0
            else False,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return self.dataloader(self.val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return self.dataloader(self.test, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader."""
        return self.dataloader(self.predict, shuffle=False)


datamodule = MNISTDataModule(train_fraction=0.1)


def train(
    trial: optuna.Trial,
    max_epochs: int,
    num_filters: int,
    embedding_dimension: int,
    iterations: int,
    kernel_size: int,
    num_downsamples: int,
    learning_rate: float,
    learning_rate_decay: float,
    classifier_hidden_layers: int,
    optimizer_beta_1: float,
    optimizer_beta_2: float,
    optimizer_eps: float,
    retina_frozen: bool,
    enable_curriculum_learning: bool,
    recurrent_module: Literal["rnn", "lstm"],
) -> float:
    """Train the eye model and return validation accuracy.

    Args:
        trial: Optuna trial object for pruning.
        max_epochs: Maximum number of training epochs.
        num_filters: Number of filters in retina convolutions.
        embedding_dimension: Dimension of the state embedding.
        iterations: Number of attention iterations.
        kernel_size: Size of convolutional kernels.
        num_downsamples: Number of downsampling layers in retina.
        learning_rate: Initial learning rate.
        learning_rate_decay: Learning rate decay factor per epoch.
        classifier_hidden_layers: Number of hidden layers in classifier.
        optimizer_beta_1: Adam optimizer beta1 parameter.
        optimizer_beta_2: Adam optimizer beta2 parameter.
        optimizer_eps: Adam optimizer epsilon parameter.
        retina_frozen: Whether to freeze retina weights during training.
        enable_curriculum_learning: Whether to use curriculum learning for attention initialization.
        recurrent_module: Type of recurrent module to use ('rnn' or 'lstm').

    Returns:
        Validation accuracy as a float between 0 and 1.
    """
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = TrackingEyeFashionMNISTNet(
        num_filters=num_filters,
        embedding_dimension=embedding_dimension,
        iterations=iterations,
        kernel_size=kernel_size,
        num_downsamples=num_downsamples,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        retina_frozen=retina_frozen,
        enable_curriculum_learning=enable_curriculum_learning,
        recurrent_module=recurrent_module,
        classifier_hidden_layers=classifier_hidden_layers,
        optimizer_beta_1=optimizer_beta_1,
        optimizer_beta_2=optimizer_beta_2,
        optimizer_eps=optimizer_eps,
    )

    early_stop = EarlyStopCallback(trial=trial)
    callbacks: list[Callback] = [early_stop]

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        log_every_n_steps=50,
        enable_model_summary=False,
        enable_checkpointing=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model, train_loader, val_loader)

    # Get validation accuracy from the last logged metric
    validation_accuracy = float(trainer.callback_metrics.get("val_acc", 0.0))
    logger.info(f"Validation accuracy: {validation_accuracy:.4f}")

    return validation_accuracy


if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        direction="maximize",
    )
    study.optimize(objective, n_trials=100, n_jobs=2)
