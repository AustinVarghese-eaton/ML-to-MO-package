"""Emit the full Modelica package as a mapping of relative path -> file text.

The generated tree (rooted at ``<PackageName>/``)::

    <PKG>/package.mo, package.order
    <PKG>/Layers/{package.mo, package.order, *.mo}
    <PKG>/Networks/{package.mo, package.order, SurrogateMLP.mo, SurrogateBlock.mo}
    <PKG>/Examples/{package.mo, package.order, RunSurrogate.mo}

``SurrogateMLP`` (a function) is the single source of truth for the network math.
``SurrogateBlock`` (a model) and ``RunSurrogate`` (a model) both call it.
"""

from __future__ import annotations

from importlib import resources

from surrogategen.export.formatting import fmt_mat, fmt_vec
from surrogategen.train import WeightBundle

LAYER_FILES = ["dense", "relu", "identity", "affine_scale", "affine_unscale"]


def _static_layer(name: str, pkg: str) -> str:
    text = (
        resources.files("surrogategen.export.static_layers")
        .joinpath(f"{name}.mo")
        .read_text(encoding="utf-8")
    )
    return text.replace("{PKG}", pkg)


def _order_file(items: list[str]) -> str:
    return "\n".join(items) + "\n"


def _top_package(pkg: str) -> str:
    return (
        "within ;\n"
        f"package {pkg}\n"
        '  annotation(uses(Modelica(version="4.0.0")));\n'
        f"end {pkg};\n"
    )


def _subpackage(pkg: str, name: str, doc: str) -> str:
    return (
        f"within {pkg};\n"
        f'package {name} "{doc}"\n'
        f"end {name};\n"
    )


def _mapping_comment(prefix: str, names: list[str]) -> str:
    lines = [f"     {prefix}[{i + 1}] = {n}" for i, n in enumerate(names)]
    return "  /*\n" + "\n".join(lines) + "\n  */\n"


def _surrogate_mlp(pkg: str, bundle: WeightBundle) -> str:
    n_in = bundle.n_in
    n_out = bundle.n_out
    n_layers = len(bundle.layers)

    lines: list[str] = []
    lines.append(f"within {pkg}.Networks;")
    lines.append('function SurrogateMLP "Feed-forward surrogate; original units in and out"')
    lines.append(f'  input Real u[{n_in}] "Inputs in original units, order = confirmed input columns";')
    lines.append(f'  output Real y[{n_out}] "Outputs in original units, order = confirmed output columns";')
    lines.append(_mapping_comment("u", bundle.input_columns).rstrip("\n"))
    lines.append(_mapping_comment("y", bundle.output_columns).rstrip("\n"))
    lines.append("protected")
    lines.append(f"  constant Real x_mean[{n_in}] = {fmt_vec(bundle.x_mean)};")
    lines.append(f"  constant Real x_scale[{n_in}] = {fmt_vec(bundle.x_scale)};")
    lines.append(f"  constant Real y_mean[{n_out}] = {fmt_vec(bundle.y_mean)};")
    lines.append(f"  constant Real y_scale[{n_out}] = {fmt_vec(bundle.y_scale)};")
    for i, (W, b) in enumerate(bundle.layers, start=1):
        rows = len(W)
        cols = len(W[0])
        lines.append(f"  constant Real W{i}[{rows}, {cols}] = {fmt_mat(W)};")
        lines.append(f"  constant Real b{i}[{rows}] = {fmt_vec(b)};")
    lines.append(f"  Real x_s[{n_in}];")
    for i, (W, _b) in enumerate(bundle.layers[:-1], start=1):
        lines.append(f"  Real h{i}[{len(W)}];")
    lines.append(f"  Real y_s[{n_out}];")
    lines.append("algorithm")
    lines.append(f"  x_s := {pkg}.Layers.affine_scale(u, x_mean, x_scale);")
    prev = "x_s"
    for i, (W, _b) in enumerate(bundle.layers[:-1], start=1):
        lines.append(f"  h{i} := {pkg}.Layers.relu({pkg}.Layers.dense({prev}, W{i}, b{i}));")
        prev = f"h{i}"
    lines.append(f"  y_s := {pkg}.Layers.dense({prev}, W{n_layers}, b{n_layers});")
    lines.append(f"  y := {pkg}.Layers.affine_unscale(y_s, y_mean, y_scale);")
    lines.append("end SurrogateMLP;")
    return "\n".join(lines) + "\n"


def _spread_positions(n: int) -> list[float]:
    """Evenly spaced y-centres from 80 down to -80; single connector -> 0."""
    if n <= 1:
        return [0.0]
    step = 160.0 / (n - 1)
    return [80.0 - step * k for k in range(n)]


def _surrogate_block(pkg: str, bundle: WeightBundle,
                     in_conn: list[str], out_conn: list[str]) -> str:
    n_in = bundle.n_in
    n_out = bundle.n_out

    lines: list[str] = []
    lines.append(f"within {pkg}.Networks;")
    lines.append('model SurrogateBlock "Connector wrapper around SurrogateMLP for the canvas"')

    in_y = _spread_positions(len(in_conn))
    for name, yc in zip(in_conn, in_y):
        y1 = round(yc - 10.0, 4)
        y2 = round(yc + 10.0, 4)
        ext = f"{{{{-120,{y1}}},{{-100,{y2}}}}}"
        lines.append(f"  Modelica.Blocks.Interfaces.RealInput {name} annotation(")
        lines.append(
            f"    Placement(transformation(extent={ext}), iconTransformation(extent={ext})));"
        )

    out_y = _spread_positions(len(out_conn))
    for name, yc in zip(out_conn, out_y):
        y1 = round(yc - 10.0, 4)
        y2 = round(yc + 10.0, 4)
        ext = f"{{{{100,{y1}}},{{120,{y2}}}}}"
        lines.append(f"  Modelica.Blocks.Interfaces.RealOutput {name} annotation(")
        lines.append(
            f"    Placement(transformation(extent={ext}), iconTransformation(extent={ext})));"
        )

    lines.append("protected")
    lines.append(f"  Real uVec[{n_in}];")
    lines.append(f"  Real yVec[{n_out}];")
    lines.append("equation")
    lines.append("  uVec = {" + ",".join(in_conn) + "};")
    lines.append(f"  yVec = {pkg}.Networks.SurrogateMLP(uVec);")
    for i, name in enumerate(out_conn, start=1):
        lines.append(f"  {name} = yVec[{i}];")
    lines.append("  annotation(")
    lines.append("    Icon(graphics={")
    lines.append("      Rectangle(extent={{-100,-100},{100,100}}, lineColor={0,0,127}),")
    lines.append('      Text(extent={{-80,40},{80,-40}}, textString="MLP")}));')
    lines.append("end SurrogateBlock;")
    return "\n".join(lines) + "\n"


def _run_surrogate(pkg: str, bundle: WeightBundle, u_test: list[float]) -> str:
    n_in = bundle.n_in
    n_out = bundle.n_out
    lines: list[str] = []
    lines.append(f"within {pkg}.Examples;")
    lines.append('model RunSurrogate "Simulate the surrogate directly with editable test inputs."')
    lines.append(
        f"  parameter Real uTest[{n_in}] = {fmt_vec(u_test)} "
        '"Edit these to sweep inputs; order = confirmed input columns";'
    )
    lines.append(f'  Real y[{n_out}] "Surrogate outputs in original units";')
    lines.append(_mapping_comment("y", bundle.output_columns).rstrip("\n"))
    lines.append("equation")
    lines.append(f"  y = {pkg}.Networks.SurrogateMLP(uTest);")
    lines.append("end RunSurrogate;")
    return "\n".join(lines) + "\n"


def build_files(
    bundle: WeightBundle,
    package_name: str,
    u_test: list[float],
    input_connectors: list[str],
    output_connectors: list[str],
) -> dict[str, str]:
    """Return {relative_path: text} for every file in the package."""
    pkg = package_name
    files: dict[str, str] = {}

    files[f"{pkg}/package.mo"] = _top_package(pkg)
    files[f"{pkg}/package.order"] = _order_file(["Layers", "Networks", "Examples"])

    files[f"{pkg}/Layers/package.mo"] = _subpackage(
        pkg, "Layers", "Reusable neural-network layer functions"
    )
    files[f"{pkg}/Layers/package.order"] = _order_file(LAYER_FILES)
    for layer in LAYER_FILES:
        files[f"{pkg}/Layers/{layer}.mo"] = _static_layer(layer, pkg)

    files[f"{pkg}/Networks/package.mo"] = _subpackage(
        pkg, "Networks", "Surrogate network function and connector block"
    )
    files[f"{pkg}/Networks/package.order"] = _order_file(["SurrogateMLP", "SurrogateBlock"])
    files[f"{pkg}/Networks/SurrogateMLP.mo"] = _surrogate_mlp(pkg, bundle)
    files[f"{pkg}/Networks/SurrogateBlock.mo"] = _surrogate_block(
        pkg, bundle, input_connectors, output_connectors
    )

    files[f"{pkg}/Examples/package.mo"] = _subpackage(pkg, "Examples", "Runnable examples")
    files[f"{pkg}/Examples/package.order"] = _order_file(["RunSurrogate"])
    files[f"{pkg}/Examples/RunSurrogate.mo"] = _run_surrogate(pkg, bundle, u_test)

    return files
