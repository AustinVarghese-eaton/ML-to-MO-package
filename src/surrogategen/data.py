"""Data preparation pipeline (mirrors prompt STEP 2-A exactly)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from surrogategen.config import SurrogateConfig

RANDOM_STATE = 42


@dataclass
class PreparedData:
    """Everything downstream stages need from data prep."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    Y_train: np.ndarray
    Y_val: np.ndarray
    Y_test: np.ndarray
    x_mean: list[float]
    x_scale: list[float]
    y_mean: list[float]
    y_scale: list[float]
    u_test: list[float]  # median of input columns, original units
    sample_inputs: list[list[float]]  # a few raw input rows for parity checks
    input_columns: list[str]
    output_columns: list[str]

    @property
    def n_in(self) -> int:
        return len(self.input_columns)

    @property
    def n_out(self) -> int:
        return len(self.output_columns)


def _load_frame(cfg: SurrogateConfig) -> pd.DataFrame:
    path = cfg.dataset_path()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset type '{suffix}': {path}")


def prepare(cfg: SurrogateConfig, n_samples: int = 5) -> PreparedData:
    """Load, clean, split, and scale the dataset per the prompt spec."""
    df = _load_frame(cfg)

    required = list(cfg.inputs) + list(cfg.outputs)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataset: {missing}")

    df = df[required].copy()
    df = df.drop_duplicates()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)

    if len(df) < 10:
        raise ValueError(
            f"Too few usable rows after cleaning ({len(df)}); need at least 10."
        )

    X = df[cfg.inputs].to_numpy(dtype=float)
    Y = df[cfg.outputs].to_numpy(dtype=float)

    # 70/15/15 split via two calls with fixed random_state.
    X_train, X_tmp, Y_train, Y_tmp = train_test_split(
        X, Y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_tmp, Y_tmp, test_size=0.50, random_state=RANDOM_STATE
    )

    print(
        f"[data] rows={len(df)} train={len(X_train)} "
        f"val={len(X_val)} test={len(X_test)}"
    )

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(Y_train)

    X_train_s = x_scaler.transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)
    Y_train_s = y_scaler.transform(Y_train)
    Y_val_s = y_scaler.transform(Y_val)
    Y_test_s = y_scaler.transform(Y_test)

    u_test = np.median(X, axis=0).tolist()
    # Representative raw input rows (from the test split) for numeric parity checks.
    take = min(n_samples, len(X_test))
    sample_inputs = X_test[:take].tolist()

    return PreparedData(
        X_train=X_train_s,
        X_val=X_val_s,
        X_test=X_test_s,
        Y_train=Y_train_s,
        Y_val=Y_val_s,
        Y_test=Y_test_s,
        x_mean=x_scaler.mean_.tolist(),
        x_scale=x_scaler.scale_.tolist(),
        y_mean=y_scaler.mean_.tolist(),
        y_scale=y_scaler.scale_.tolist(),
        u_test=u_test,
        sample_inputs=sample_inputs,
        input_columns=list(cfg.inputs),
        output_columns=list(cfg.outputs),
    )
