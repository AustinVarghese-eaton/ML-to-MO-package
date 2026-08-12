"""Modelica-safe formatting for floats and arrays.

No Python/NumPy artifacts (no ``np.float64(...)``, no scientific ``e`` ambiguity issues,
no trailing commas). 1D -> ``{a,b,c}``; 2D -> ``{{a,b},{c,d}}``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

_PRECISION = 17  # round-trip precision for IEEE-754 doubles


def fmt_float(value: float) -> str:
    """Format a single float with full double precision, rejecting NaN/Inf."""
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"Refusing to format non-finite value: {value!r}")
    # repr gives the shortest round-trippable form for Python floats.
    s = repr(v)
    if s in ("inf", "-inf", "nan"):  # defensive; already guarded above
        raise ValueError(f"Refusing to format non-finite value: {value!r}")
    return s


def fmt_vec(values: Sequence[float]) -> str:
    """Format a 1D sequence as ``{a,b,c}``."""
    return "{" + ",".join(fmt_float(v) for v in values) + "}"


def fmt_mat(matrix: Iterable[Sequence[float]]) -> str:
    """Format a 2D sequence as ``{{a,b},{c,d}}``."""
    rows = [fmt_vec(row) for row in matrix]
    return "{" + ",".join(rows) + "}"
