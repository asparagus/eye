# eye

## About

This project explores a simplified model of the eye and trains it on the Fashion MNIST dataset.

The module consists of three components under the `eye/architecture` directory.
- [module.py](eye/architecture/module.py)
- [retina.py](eye/architecture/retina.py)
- [motor.py](eye/architecture/motor.py)


## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python package management.

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run scripts using uv:
   ```bash
   uv run python -m ...
   ```

## Training Fashion-MNIST

To train the neural network on Fashion-MNIST dataset:

```bash
bin/train
```

This will download the dataset, train a simple network, and save the model.

## Results
We can see the model learns to move the eye around, and gets most answers right
![results-fashion-mnist](public/results.png)
