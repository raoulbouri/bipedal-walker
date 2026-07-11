#!/usr/bin/env python3
"""Package the Colab upload bundle described in docs/colab_upload_manifest.md.

Standard-library only (shutil, pathlib, zipfile) -- no new project
dependencies. Produces a zip file with the exact directory structure shown
in the manifest's "Bundle contents" tree diagram (paths are NOT flattened
and NOT prefixed with an extra top-level folder -- the manifest's own tree
diagram roots directly at models/, meshes/, mjlab_biped/, docs/,
requirements-colab.txt, notebooks/).

Usage:
    python scripts/package_colab_bundle.py [output_path]

`output_path` may be a .zip file path or a directory (in which case
`colab_bundle.zip` is created inside it). Defaults to `colab_bundle.zip` in
the repo root.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirrors docs/colab_upload_manifest.md's bundle contents tree -- keep in sync.
# Each entry is a path relative to REPO_ROOT; the same relative path is used
# as the archive name inside the zip (i.e. the bundle root == the repo root
# for these specific files/dirs, matching the manifest's own diagram).
BUNDLE_FILES = [
    "models/mjcf/biped_warp.xml",
    "meshes/stl/__1.stl",
    "meshes/stl/__2.stl",
    "meshes/stl/_.stl",
    "meshes/stl/Foot.stl",
    "meshes/stl/Hip_Base.stl",
    "meshes/stl/Hip_Joint_A.stl",
    "meshes/stl/Hip_Joint_B.stl",
    "meshes/stl/Motor.stl",
    "meshes/stl/Part_1.stl",
    "meshes/stl/Tibia.stl",
    "meshes/stl/Tube.stl",
    "meshes/stl/Upper_Leg_A.stl",
    "meshes/stl/Upper_Leg_B.stl",
    "mjlab_biped/__init__.py",
    "mjlab_biped/entity.py",
    "mjlab_biped/observations.py",
    "mjlab_biped/commands.py",
    "mjlab_biped/init_noise.py",
    "mjlab_biped/domain_randomization.py",
    "mjlab_biped/rewards.py",
    "mjlab_biped/terminations.py",
    "mjlab_biped/config.py",
    "mjlab_biped/rl_cfg.py",
    "mjlab_biped/mjlab_task.py",
    "mjlab_biped/local_env.py",
    "mjlab_biped/README.md",
    "docs/mjlab_adapter_notes.md",
    "requirements-colab.txt",
    "notebooks/train_biped.ipynb",
    "scripts/colab_train.py",
    "scripts/colab_play.py",
]

DEFAULT_OUTPUT = REPO_ROOT / "colab_bundle.zip"


def _resolve_output_path(arg: str | None) -> Path:
    if arg is None:
        return DEFAULT_OUTPUT
    path = Path(arg)
    if path.is_dir() or (not path.suffix and not path.exists()):
        # Treat as a directory target (existing dir, or a path with no
        # extension that doesn't exist -- assume the caller meant a dir).
        return path / "colab_bundle.zip"
    return path


def package_bundle(output_path: Path | None = None) -> Path:
    """Build the Colab upload bundle zip. Returns the output path."""
    if output_path is None:
        output_path = DEFAULT_OUTPUT

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Verify every source file exists before zipping anything.
    missing = []
    resolved = []
    for rel_path in BUNDLE_FILES:
        src = REPO_ROOT / rel_path
        if not src.exists():
            missing.append(rel_path)
        else:
            resolved.append((src, rel_path))

    if missing:
        raise FileNotFoundError(
            "Missing bundle source file(s), cannot package: "
            + ", ".join(missing)
        )

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in resolved:
            zf.write(src, arcname)

    total_files = len(resolved)
    total_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Bundled {total_files} files into {output_path}")
    print(f"Zip size: {total_size_mb:.2f} MB")
    print(f"Output path: {output_path}")

    return output_path


def main(argv: list[str]) -> int:
    arg = argv[1] if len(argv) > 1 else None
    output_path = _resolve_output_path(arg)
    package_bundle(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
