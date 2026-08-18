"""Training with a TensorFlow primary path and a scikit-learn fallback.

Both paths emit a common :class:`WeightBundle` so the Modelica exporter never needs
to know which framework produced the weights. Weight matrices are stored already
transposed for Modelica (``W_mo[i, j]`` with ``y[i] = b[i] + sum_j W[i, j] * x[j]``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from surrogategen.config import TrainingParams
from surrogategen.data import PreparedData

SEED = 42
HIDDEN = (128, 128, 64)  # default architecture; overridden by params.hidden_layers


@dataclass
class WeightBundle:
    """Framework-agnostic container of everything the exporter needs."""

    layers: list[tuple[list[list[float]], list[float]]]  # [(W_mo, b), ...] per Dense layer
    x_mean: list[float]
    x_scale: list[float]
    y_mean: list[float]
    y_scale: list[float]
    input_columns: list[str]
    output_columns: list[str]
    backend: str
    epochs_run: int
    final_val_loss: float
    y_log_mask: list[bool] | None = None  # per-output: expm1 applied on True entries

    @property
    def n_in(self) -> int:
        return len(self.x_mean)

    @property
    def n_out(self) -> int:
        return len(self.y_mean)

    def predict(self, u: np.ndarray) -> np.ndarray:
        """Run the exact exported math in NumPy (original units in and out).

        Mirrors the Modelica ``SurrogateMLP`` so ``predictions.json`` is derived from
        the same constants that are written into the package.
        """
        u = np.atleast_2d(np.asarray(u, dtype=float))
        x_mean = np.asarray(self.x_mean)
        x_scale = np.asarray(self.x_scale)
        eps = 1e-12
        denom = np.where(np.abs(x_scale) > eps, x_scale, 1.0)
        h = (u - x_mean) / denom
        n_layers = len(self.layers)
        for idx, (W, b) in enumerate(self.layers):
            W = np.asarray(W)
            b = np.asarray(b)
            h = h @ W.T + b
            if idx < n_layers - 1:  # relu on hidden layers, linear on output
                h = np.maximum(h, 0.0)
        y = h * np.asarray(self.y_scale) + np.asarray(self.y_mean)
        if self.y_log_mask is not None and any(self.y_log_mask):
            mask = np.asarray(self.y_log_mask, dtype=bool)
            y = y.copy()
            y[:, mask] = np.expm1(y[:, mask])
        return y


def _assert_finite(bundle: WeightBundle) -> None:
    for name, arr in (
        ("x_mean", bundle.x_mean),
        ("x_scale", bundle.x_scale),
        ("y_mean", bundle.y_mean),
        ("y_scale", bundle.y_scale),
    ):
        a = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"Non-finite value found in scaler '{name}'.")
    for i, (W, b) in enumerate(bundle.layers, start=1):
        if not np.all(np.isfinite(np.asarray(W, dtype=float))):
            raise ValueError(f"Non-finite value found in weight matrix W{i}.")
        if not np.all(np.isfinite(np.asarray(b, dtype=float))):
            raise ValueError(f"Non-finite value found in bias b{i}.")
        if len(W) != len(b):
            raise ValueError(
                f"Layer {i}: weight rows ({len(W)}) != bias length ({len(b)})."
            )


def _train_tensorflow(data: PreparedData, params: TrainingParams) -> WeightBundle:
    import tensorflow as tf  # local import so the module loads without TF installed

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    hidden = tuple(params.hidden_layers)
    reg = tf.keras.regularizers.l2(params.l2) if params.l2 > 0 else None
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(data.n_in,))]
        + [
            tf.keras.layers.Dense(u, activation="relu", kernel_regularizer=reg)
            for u in hidden
        ]
        + [tf.keras.layers.Dense(data.n_out, activation="linear", kernel_regularizer=reg)]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss="mse",
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=params.patience, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0
    )
    batch_size = min(params.batch_size, max(1, len(data.X_train) // 10))
    history = model.fit(
        data.X_train,
        data.Y_train,
        validation_data=(data.X_val, data.Y_val),
        batch_size=batch_size,
        epochs=params.epochs,
        callbacks=[early, reduce_lr],
        verbose=0,
    )

    layers: list[tuple[list[list[float]], list[float]]] = []
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        W_keras, b = weights  # shapes (in, out), (out,)
        W_mo = np.asarray(W_keras).T.tolist()
        layers.append((W_mo, np.asarray(b).tolist()))

    epochs_run = len(history.history["loss"])
    final_val_loss = float(history.history["val_loss"][-1])
    print(f"[train] backend=tensorflow epochs_run={epochs_run} "
          f"final_val_loss={final_val_loss:.6g}")

    return WeightBundle(
        layers=layers,
        x_mean=data.x_mean,
        x_scale=data.x_scale,
        y_mean=data.y_mean,
        y_scale=data.y_scale,
        input_columns=data.input_columns,
        output_columns=data.output_columns,
        backend="tensorflow",
        epochs_run=epochs_run,
        final_val_loss=final_val_loss,
        y_log_mask=data.y_log_mask,
    )


def _train_sklearn(data: PreparedData, params: TrainingParams) -> WeightBundle:
    from sklearn.neural_network import MLPRegressor

    model = MLPRegressor(
        hidden_layer_sizes=tuple(params.hidden_layers),
        activation="relu",
        solver="adam",
        alpha=params.l2,
        learning_rate_init=params.learning_rate,
        batch_size=min(params.batch_size, max(1, len(data.X_train) // 10)),
        max_iter=params.epochs,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=params.patience,
        random_state=SEED,
    )
    model.fit(data.X_train, data.Y_train)

    layers: list[tuple[list[list[float]], list[float]]] = []
    for W_sk, b in zip(model.coefs_, model.intercepts_):
        # sklearn coefs_ have shape (in, out) like Keras kernels.
        W_mo = np.asarray(W_sk).T.tolist()
        layers.append((W_mo, np.asarray(b).tolist()))

    val_pred = model.predict(data.X_val)
    final_val_loss = float(np.mean((val_pred - data.Y_val) ** 2))
    epochs_run = int(getattr(model, "n_iter_", params.epochs))
    print(f"[train] backend=sklearn epochs_run={epochs_run} "
          f"final_val_loss={final_val_loss:.6g}")

    return WeightBundle(
        layers=layers,
        x_mean=data.x_mean,
        x_scale=data.x_scale,
        y_mean=data.y_mean,
        y_scale=data.y_scale,
        input_columns=data.input_columns,
        output_columns=data.output_columns,
        backend="sklearn",
        epochs_run=epochs_run,
        final_val_loss=final_val_loss,
        y_log_mask=data.y_log_mask,
    )


def train(data: PreparedData, params: TrainingParams) -> WeightBundle:
    """Train the surrogate, preferring TensorFlow and falling back to scikit-learn."""
    try:
        bundle = _train_tensorflow(data, params)
    except Exception as exc:  # noqa: BLE001 - fallback is intentional for CI robustness
        print(f"[train] TensorFlow path failed ({exc!r}); falling back to scikit-learn.")
        bundle = _train_sklearn(data, params)
    _assert_finite(bundle)
    return bundle
