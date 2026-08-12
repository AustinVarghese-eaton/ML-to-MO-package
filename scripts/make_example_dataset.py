"""Generate the deterministic synthetic example dataset (datasets/example/data.csv)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def main() -> None:
    rng = np.random.default_rng(42)
    n = 600
    Vge = rng.uniform(10, 20, n)
    Ic = rng.uniform(1, 100, n)
    Tj = rng.uniform(25, 150, n)
    Vce = 0.8 + 0.012 * Ic + 0.002 * Tj - 0.03 * Vge + rng.normal(0, 0.02, n)
    Eon = (
        0.5 + 0.03 * Ic + 0.004 * Tj - 0.05 * Vge
        + 0.0002 * Ic * Tj + rng.normal(0, 0.05, n)
    )

    out = Path(__file__).resolve().parents[1] / "datasets" / "example" / "data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Vge", "Ic", "Tj", "Vce", "Eon"])
        for row in zip(Vge, Ic, Tj, Vce, Eon):
            w.writerow([round(float(v), 6) for v in row])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
