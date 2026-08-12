"""Command-line interface: ``surrogategen build <config>``."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from surrogategen import __version__


def _cmd_build(args: argparse.Namespace) -> int:
    # Imported lazily so ``--help`` works without heavy deps installed.
    from surrogategen import config as config_mod
    from surrogategen import data as data_mod
    from surrogategen import packager, selfcheck
    from surrogategen import train as train_mod
    from surrogategen.export import templates

    cfg = config_mod.load(args.config)
    print(f"[build] package={cfg.package_name} dataset={cfg.dataset_path()}")

    prepared = data_mod.prepare(cfg)
    bundle = train_mod.train(prepared, cfg.training)

    in_conn = list(cfg.input_connectors().values())
    out_conn = list(cfg.output_connectors().values())

    files = templates.build_files(
        bundle,
        package_name=cfg.package_name,
        u_test=prepared.u_test,
        input_connectors=in_conn,
        output_connectors=out_conn,
    )

    selfcheck.run(files, cfg.package_name, bundle, in_conn, out_conn, prepared.u_test)
    print("[build] self-check passed")

    out_dir = Path(args.out).resolve()
    pkg_parent = out_dir / cfg.package_name
    if pkg_parent.exists():
        shutil.rmtree(pkg_parent)

    root = packager.write_package(files, out_dir, cfg.package_name)
    zip_path = out_dir / f"{cfg.package_name}.zip"
    packager.zip_package(root, zip_path)
    print(f"[build] wrote package -> {root}")
    print(f"[build] wrote zip     -> {zip_path}")

    predictions = _build_predictions(cfg, bundle, prepared)
    pred_path = out_dir / f"{cfg.package_name}.predictions.json"
    pred_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"[build] wrote parity  -> {pred_path}")

    return 0


def _build_predictions(cfg, bundle, prepared) -> dict:
    samples = []
    for u in prepared.sample_inputs:
        y = bundle.predict(u)[0].tolist()
        samples.append({"u": list(u), "y": y})
    u_test_y = bundle.predict(prepared.u_test)[0].tolist()
    return {
        "package_name": cfg.package_name,
        "backend": bundle.backend,
        "input_columns": bundle.input_columns,
        "output_columns": bundle.output_columns,
        "tolerance": {"rtol": cfg.tolerance.rtol, "atol": cfg.tolerance.atol},
        "u_test": {"u": list(prepared.u_test), "y": u_test_y},
        "samples": samples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surrogategen",
        description="Convert a dataset into a Modelica 4.0.0 NN surrogate package.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build a surrogate package from a YAML config.")
    build.add_argument("config", help="Path to the dataset YAML config file.")
    build.add_argument(
        "--out", default="out", help="Output directory (default: ./out)."
    )
    build.set_defaults(func=_cmd_build)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
