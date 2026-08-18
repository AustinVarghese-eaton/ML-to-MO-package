import numpy as np
import pytest

from surrogategen.train import WeightBundle


def make_bundle(n_in: int = 3, n_out: int = 2, seed: int = 0) -> WeightBundle:
    rng = np.random.default_rng(seed)
    dims = [(128, n_in), (128, 128), (64, 128), (n_out, 64)]
    layers = []
    for rows, cols in dims:
        W = rng.standard_normal((rows, cols)).tolist()
        b = rng.standard_normal(rows).tolist()
        layers.append((W, b))
    return WeightBundle(
        layers=layers,
        x_mean=rng.standard_normal(n_in).tolist(),
        x_scale=(rng.random(n_in) + 0.5).tolist(),
        y_mean=rng.standard_normal(n_out).tolist(),
        y_scale=(rng.random(n_out) + 0.5).tolist(),
        input_columns=[f"in{i}" for i in range(n_in)],
        output_columns=[f"out{i}" for i in range(n_out)],
        backend="test",
        epochs_run=1,
        final_val_loss=0.0,
    )


@pytest.fixture
def bundle():
    return make_bundle()


def make_bundle_arch(hidden=(128, 128, 64), n_in: int = 3, n_out: int = 2,
                     seed: int = 0) -> WeightBundle:
    """Build a bundle with an arbitrary hidden-layer configuration."""
    rng = np.random.default_rng(seed)
    sizes = list(hidden) + [n_out]
    prev = n_in
    layers = []
    for rows in sizes:
        W = rng.standard_normal((rows, prev)).tolist()
        b = rng.standard_normal(rows).tolist()
        layers.append((W, b))
        prev = rows
    return WeightBundle(
        layers=layers,
        x_mean=rng.standard_normal(n_in).tolist(),
        x_scale=(rng.random(n_in) + 0.5).tolist(),
        y_mean=rng.standard_normal(n_out).tolist(),
        y_scale=(rng.random(n_out) + 0.5).tolist(),
        input_columns=[f"in{i}" for i in range(n_in)],
        output_columns=[f"out{i}" for i in range(n_out)],
        backend="test",
        epochs_run=1,
        final_val_loss=0.0,
    )
