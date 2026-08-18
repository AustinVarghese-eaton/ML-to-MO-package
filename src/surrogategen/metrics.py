"""Accuracy metrics for the trained surrogate on the held-out test set.

Metrics are computed in original (physical) units so the numbers are directly
comparable to the dataset. Used at build time to print a per-output report and
to write ``<PackageName>.metrics.json`` beside the parity file.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _per_column(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_abs_err = float(np.max(abs_err)) if abs_err.size else 0.0

    # MAPE only over non-zero targets to avoid division blow-ups.
    denom = np.abs(y_true)
    nonzero = denom > _EPS
    if np.any(nonzero):
        mape = float(np.mean(abs_err[nonzero] / denom[nonzero]) * 100.0)
    else:
        mape = float("nan")

    # R^2 = 1 - SS_res / SS_tot.
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _EPS else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "r2": r2,
        "max_abs_err": max_abs_err,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_columns: list[str],
) -> dict:
    """Return per-output and overall accuracy metrics (original units).

    ``y_true`` and ``y_pred`` are 2-D arrays of shape ``(n_samples, n_out)``.
    """
    y_true = np.atleast_2d(np.asarray(y_true, dtype=float))
    y_pred = np.atleast_2d(np.asarray(y_pred, dtype=float))
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}."
        )

    per_output = {
        col: _per_column(y_true[:, j], y_pred[:, j])
        for j, col in enumerate(output_columns)
    }
    overall = _per_column(y_true.ravel(), y_pred.ravel())
    return {
        "n_test": int(y_true.shape[0]),
        "per_output": per_output,
        "overall": overall,
    }


def format_report(metrics: dict) -> str:
    """Render a compact fixed-width table for console output."""
    header = f"{'output':<16}{'MAE':>12}{'RMSE':>12}{'MAPE%':>10}{'R2':>10}{'maxErr':>12}"
    lines = [header, "-" * len(header)]
    for col, m in metrics["per_output"].items():
        lines.append(
            f"{col:<16}{m['mae']:>12.4g}{m['rmse']:>12.4g}"
            f"{m['mape_pct']:>10.3g}{m['r2']:>10.4f}{m['max_abs_err']:>12.4g}"
        )
    o = metrics["overall"]
    lines.append("-" * len(header))
    lines.append(
        f"{'overall':<16}{o['mae']:>12.4g}{o['rmse']:>12.4g}"
        f"{o['mape_pct']:>10.3g}{o['r2']:>10.4f}{o['max_abs_err']:>12.4g}"
    )
    return "\n".join(lines)
