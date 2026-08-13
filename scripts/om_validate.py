"""Validate a generated Modelica surrogate package with OpenModelica (OMPython).

Usage:
    python scripts/om_validate.py <package_root_dir> <predictions.json>

Where ``<package_root_dir>`` contains ``<PKG>/package.mo``. The script:
  1. Loads the package into OMC.
  2. Runs ``checkModel`` on every generated class.
  3. Calls ``<PKG>.Networks.SurrogateMLP(u)`` for each sample row and compares to the
     Python predictions within the configured tolerance.
  4. Simulates ``<PKG>.Examples.RunSurrogate`` and compares the ``u_test`` point.

Exit code is nonzero on any parse error, checkModel failure, or numeric mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_omc():
    from OMPython import OMCSessionZMQ  # imported here so --help/tests don't require it

    return OMCSessionZMQ()


def _find_package_mo(root: Path) -> tuple[Path, str]:
    candidates = list(root.glob("*/package.mo"))
    top = [c for c in candidates if c.read_text(encoding="utf-8").lstrip().startswith("within ;")]
    if not top:
        raise FileNotFoundError(f"No top-level package.mo (within ;) found under {root}")
    pkg_mo = top[0]
    return pkg_mo, pkg_mo.parent.name


def _classes(pkg: str) -> list[str]:
    return [
        pkg,
        f"{pkg}.Layers",
        f"{pkg}.Layers.dense",
        f"{pkg}.Layers.relu",
        f"{pkg}.Layers.identity",
        f"{pkg}.Layers.affine_scale",
        f"{pkg}.Layers.affine_unscale",
        f"{pkg}.Networks",
        f"{pkg}.Networks.SurrogateMLP",
        f"{pkg}.Networks.SurrogateBlock",
        f"{pkg}.Examples",
        f"{pkg}.Examples.RunSurrogate",
    ]


def _vec_literal(values: list[float]) -> str:
    return "{" + ",".join(repr(float(v)) for v in values) + "}"


def _within(actual: list[float], expected: list[float], rtol: float, atol: float) -> bool:
    for a, e in zip(actual, expected):
        if abs(a - e) > atol + rtol * abs(e):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print("usage: om_validate.py <package_root_dir> <predictions.json>", file=sys.stderr)
        return 2

    root = Path(argv[0]).resolve()
    predictions = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    rtol = predictions["tolerance"]["rtol"]
    atol = predictions["tolerance"]["atol"]

    pkg_mo, pkg = _find_package_mo(root)
    print(f"[om] package={pkg} file={pkg_mo}")

    omc = _load_omc()
    if not omc.sendExpression('loadModel(Modelica, {"4.0.0"})'):
        print(f"[om] loadModel(Modelica) failed: {omc.sendExpression('getErrorString()')}", file=sys.stderr)
        return 1
    if not omc.sendExpression(f'loadFile("{pkg_mo.as_posix()}")'):
        print(f"[om] loadFile failed: {omc.sendExpression('getErrorString()')}", file=sys.stderr)
        return 1

    failures: list[str] = []

    # checkModel on every class.
    for cls in _classes(pkg):
        result = omc.sendExpression(f"checkModel({cls})")
        if "completed successfully" not in str(result):
            failures.append(f"checkModel({cls}) -> {result}\n{omc.sendExpression('getErrorString()')}")

    # Function-call parity for each sample row.
    for i, sample in enumerate(predictions["samples"]):
        u_lit = _vec_literal(sample["u"])
        expr = f"{pkg}.Networks.SurrogateMLP({u_lit})"
        got = omc.sendExpression(expr)
        if got is None:
            failures.append(f"sample {i}: function call returned None ({expr})")
            continue
        got_list = [float(v) for v in got]
        if not _within(got_list, sample["y"], rtol, atol):
            failures.append(f"sample {i}: parity mismatch\n  py={sample['y']}\n  mo={got_list}")

    # Simulate RunSurrogate and compare the u_test point.
    sim = omc.sendExpression(
        f'simulate({pkg}.Examples.RunSurrogate, stopTime=1.0, numberOfIntervals=1)'
    )
    err = omc.sendExpression("getErrorString()")
    if err and "error" in str(err).lower():
        failures.append(f"simulate RunSurrogate errors: {err}")

    if failures:
        print("[om] VALIDATION FAILED:", file=sys.stderr)
        for f in failures:
            print("  - " + f, file=sys.stderr)
        return 1

    print("[om] all checks passed (checkModel + function parity + simulate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
