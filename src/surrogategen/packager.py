"""Write the package tree to disk and zip it (single root folder inside)."""

from __future__ import annotations

import zipfile
from pathlib import Path


def write_package(files: dict[str, str], out_dir: Path, package_name: str) -> Path:
    """Write all files under ``out_dir`` and return the package root directory."""
    out_dir = Path(out_dir)
    for rel_path, text in files.items():
        target = out_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    return out_dir / package_name


def zip_package(root_dir: Path, zip_path: Path) -> Path:
    """Zip ``root_dir`` so the archive contains exactly one top-level folder."""
    root_dir = Path(root_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    base = root_dir.name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(root_dir.rglob("*")):
            if file.is_file():
                arcname = f"{base}/{file.relative_to(root_dir).as_posix()}"
                zf.write(file, arcname)

    _verify_single_root(zip_path, base)
    return zip_path


def _verify_single_root(zip_path: Path, expected_root: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        roots = {name.split("/", 1)[0] for name in zf.namelist()}
    if roots != {expected_root}:
        raise ValueError(
            f"ZIP must contain a single root folder '{expected_root}', found: {sorted(roots)}"
        )
