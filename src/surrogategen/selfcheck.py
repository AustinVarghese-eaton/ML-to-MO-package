"""Automated pre-zip self-check (the prompt's 12-point checklist as code)."""

from __future__ import annotations

import re

import numpy as np

from surrogategen.export.templates import LAYER_FILES
from surrogategen.train import WeightBundle


class SelfCheckError(ValueError):
    """Raised when a generated package fails one or more structural checks."""


def _expected_within(rel_path: str, pkg: str) -> str:
    """Expected ``within`` clause for a file given its path under the package root."""
    parts = rel_path.split("/")
    filename = parts[-1]
    dirs = parts[1:-1]  # directory chain between the package root and the file
    if filename == "package.mo":
        # A package.mo declares the package named by its own folder; the within clause
        # is therefore the *enclosing* scope (drop the last directory).
        scope_dirs = dirs[:-1]
    else:
        scope_dirs = dirs
    if not scope_dirs and filename == "package.mo" and not dirs:
        return "within ;"
    scope = ".".join([pkg] + scope_dirs)
    return f"within {scope};"


def _class_name(rel_path: str, pkg: str) -> str:
    parts = rel_path.split("/")
    stem = parts[-1].rsplit(".", 1)[0]
    if stem == "package":
        return pkg if len(parts) == 2 else parts[-2]
    return stem


def _check_balanced(text: str, errors: list[str], rel_path: str) -> None:
    for open_ch, close_ch in (("(", ")"), ("{", "}"), ("[", "]")):
        if text.count(open_ch) != text.count(close_ch):
            errors.append(
                f"{rel_path}: unbalanced '{open_ch}{close_ch}' "
                f"({text.count(open_ch)} vs {text.count(close_ch)})."
            )


def _order_items(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def run(
    files: dict[str, str],
    package_name: str,
    bundle: WeightBundle,
    input_connectors: list[str],
    output_connectors: list[str],
    u_test: list[float],
) -> None:
    """Validate the generated files; raise :class:`SelfCheckError` listing all problems."""
    pkg = package_name
    errors: list[str] = []

    # --- Per-file checks (within, end, balanced brackets) -------------------
    for rel_path, text in files.items():
        if rel_path.endswith("package.order"):
            _check_balanced(text, errors, rel_path)
            continue

        # 1. within clause matches folder path.
        expected = _expected_within(rel_path, pkg)
        first = text.splitlines()[0].strip() if text.strip() else ""
        if first != expected:
            errors.append(f"{rel_path}: within clause '{first}' != expected '{expected}'.")

        # 2. matching end Name;.
        name = _class_name(rel_path, pkg)
        if f"end {name};" not in text:
            errors.append(f"{rel_path}: missing 'end {name};'.")

        # 10/12. balanced brackets.
        _check_balanced(text, errors, rel_path)

    # 6/12. SurrogateMLP is a function with algorithm (not equation).
    mlp = files.get(f"{pkg}/Networks/SurrogateMLP.mo", "")
    if "function SurrogateMLP" not in mlp:
        errors.append("SurrogateMLP.mo: must declare 'function SurrogateMLP'.")
    if "algorithm" not in mlp:
        errors.append("SurrogateMLP.mo: function must have an 'algorithm' section.")
    if re.search(r"^equation\b", mlp, flags=re.MULTILINE):
        errors.append("SurrogateMLP.mo: function must not contain an 'equation' section.")

    # 7. SurrogateBlock is a model with equation and calls SurrogateMLP once.
    block = files.get(f"{pkg}/Networks/SurrogateBlock.mo", "")
    if "model SurrogateBlock" not in block:
        errors.append("SurrogateBlock.mo: must declare 'model SurrogateBlock'.")
    if "equation" not in block:
        errors.append("SurrogateBlock.mo: model must have an 'equation' section.")
    if block.count("SurrogateMLP(") != 1:
        errors.append("SurrogateBlock.mo: must call SurrogateMLP exactly once.")
    if block.count("RealInput") != len(input_connectors):
        errors.append(
            f"SurrogateBlock.mo: RealInput count {block.count('RealInput')} "
            f"!= n_in {len(input_connectors)}."
        )
    if block.count("RealOutput") != len(output_connectors):
        errors.append(
            f"SurrogateBlock.mo: RealOutput count {block.count('RealOutput')} "
            f"!= n_out {len(output_connectors)}."
        )

    # 8. RunSurrogate model calls SurrogateMLP once; uTest length == n_in.
    run_model = files.get(f"{pkg}/Examples/RunSurrogate.mo", "")
    if "model RunSurrogate" not in run_model:
        errors.append("RunSurrogate.mo: must declare 'model RunSurrogate'.")
    if run_model.count("SurrogateMLP(") != 1:
        errors.append("RunSurrogate.mo: must call SurrogateMLP exactly once.")
    if len(u_test) != bundle.n_in:
        errors.append(f"uTest length {len(u_test)} != n_in {bundle.n_in}.")

    # 3. package.order completeness.
    _expect_order(files, f"{pkg}/package.order", ["Layers", "Networks", "Examples"], errors)
    _expect_order(files, f"{pkg}/Layers/package.order", LAYER_FILES, errors)
    _expect_order(
        files, f"{pkg}/Networks/package.order", ["SurrogateMLP", "SurrogateBlock"], errors
    )
    _expect_order(files, f"{pkg}/Examples/package.order", ["RunSurrogate"], errors)

    # 4/5. weight dims and bias lengths (derived from the layer chain, so any
    # hidden-layer configuration is supported).
    n_layers = len(bundle.layers)
    for i, (W, b) in enumerate(bundle.layers, start=1):
        W_arr = np.asarray(W)
        if W_arr.ndim != 2:
            errors.append(f"W{i} must be a 2-D matrix, got shape {W_arr.shape}.")
            continue
        rows, cols = W_arr.shape
        expected_cols = bundle.n_in if i == 1 else len(bundle.layers[i - 2][0])
        if cols != expected_cols:
            errors.append(f"W{i} cols {cols} != expected {expected_cols}.")
        if i == n_layers and rows != bundle.n_out:
            errors.append(f"W{i} rows {rows} != n_out {bundle.n_out}.")
        if len(b) != rows:
            errors.append(f"b{i} length {len(b)} != {rows}.")

    # 9. no NaN/Inf in emitted constants.
    for i, (W, b) in enumerate(bundle.layers, start=1):
        if not np.all(np.isfinite(np.asarray(W, dtype=float))):
            errors.append(f"W{i} contains NaN/Inf.")
        if not np.all(np.isfinite(np.asarray(b, dtype=float))):
            errors.append(f"b{i} contains NaN/Inf.")
    for name, arr in (
        ("x_mean", bundle.x_mean), ("x_scale", bundle.x_scale),
        ("y_mean", bundle.y_mean), ("y_scale", bundle.y_scale),
    ):
        if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
            errors.append(f"{name} contains NaN/Inf.")

    if errors:
        raise SelfCheckError("Self-check failed:\n  - " + "\n  - ".join(errors))


def _expect_order(files: dict[str, str], path: str, expected: list[str],
                  errors: list[str]) -> None:
    if path not in files:
        errors.append(f"Missing {path}.")
        return
    items = _order_items(files[path])
    if items != expected:
        errors.append(f"{path}: order {items} != expected {expected}.")
