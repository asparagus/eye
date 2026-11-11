"""Run or resume an Optuna sweep for the eye model."""

import argparse
import logging
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

import optuna

from eye.training import train_eye_focus
from eye.training.train_eye_focus import objective

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default="sqlite:///db.sqlite3")
    parser.add_argument("--study-name", default="eye")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--max-hours", type=float, default=11.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backup", type=Path, default=None)
    return parser.parse_args()


def backup_sqlite(source: Path, destination: Path) -> None:
    """Copy a live SQLite database without leaving a partial destination file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path("/tmp") / f"{destination.name}.tmp"
    with sqlite3.connect(source) as src, sqlite3.connect(temporary) as dst:
        src.backup(dst)
    shutil.copyfile(temporary, destination)
    temporary.unlink()


def main() -> None:
    """Create or resume the study and run trials until a limit is reached."""
    args = parse_args()

    if args.n_jobs != 1:
        logger.warning("n_jobs > 1 shares a single GPU and will contend for it")

    if args.num_workers is not None:
        train_eye_focus.datamodule.num_workers = args.num_workers

    db_path: Path | None = None
    if args.backup is not None:
        if not args.storage.startswith("sqlite:///"):
            raise SystemExit("--backup requires sqlite storage")
        db_path = Path(args.storage.removeprefix("sqlite:///"))

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    callbacks: list[Callable[[optuna.Study, optuna.trial.FrozenTrial], None]] = []
    if args.backup is not None and db_path is not None:

        def backup_callback(
            study: optuna.Study, trial: optuna.trial.FrozenTrial
        ) -> None:
            backup_sqlite(db_path, args.backup)

        callbacks.append(backup_callback)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.max_hours * 3600,
        n_jobs=args.n_jobs,
        catch=(Exception,),
        callbacks=callbacks,
    )

    try:
        print(f"Best value: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
    except ValueError:
        print("No completed trials yet.")


if __name__ == "__main__":
    main()
