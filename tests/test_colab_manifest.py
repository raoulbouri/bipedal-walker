"""Phase 7.A: regression tests keeping docs/colab_upload_manifest.md truthful.

These re-run the same static analysis the manifest doc itself was built
from (see its "How this was verified" section) so future changes to
mjlab_biped/ or biped_warp.xml that would invalidate the manifest are
caught automatically, instead of the doc silently going stale.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MJLAB_BIPED_DIR = REPO_ROOT / "mjlab_biped"
WARP_MODEL = REPO_ROOT / "models" / "mjcf" / "biped_warp.xml"
MESH_DIR = REPO_ROOT / "meshes" / "stl"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "train_biped.ipynb"

EXPECTED_MODEL_PATH = "models/mjcf/biped_warp.xml"
OTHER_MODEL_PATHS = [
    "models/mjcf/biped.xml",
    "models/mjcf/biped_no_jetson.xml",
    "models/mjcf/biped_raw.xml",
]


def test_model_path_referenced_matches_manifest():
    """entity.py and local_env.py must reference exactly biped_warp.xml,
    and no other model path may appear anywhere under mjlab_biped/."""
    entity_src = (MJLAB_BIPED_DIR / "entity.py").read_text()
    local_env_src = (MJLAB_BIPED_DIR / "local_env.py").read_text()

    assert EXPECTED_MODEL_PATH in entity_src, (
        f"{EXPECTED_MODEL_PATH!r} not found in mjlab_biped/entity.py - "
        "the manifest doc assumes this exact path is referenced there."
    )
    assert EXPECTED_MODEL_PATH in local_env_src, (
        f"{EXPECTED_MODEL_PATH!r} not found in mjlab_biped/local_env.py - "
        "the manifest doc assumes this exact path is referenced there."
    )

    py_files = sorted(MJLAB_BIPED_DIR.rglob("*.py"))
    for py_file in py_files:
        src = py_file.read_text()
        for other_path in OTHER_MODEL_PATHS:
            assert other_path not in src, (
                f"{py_file.relative_to(REPO_ROOT)} references {other_path!r}, "
                "a model path not documented in docs/colab_upload_manifest.md "
                "(which states the bundle contains only biped_warp.xml). "
                "Update the manifest if this is intentional."
            )


def test_all_referenced_meshes_exist():
    """Every mesh file="..." referenced in biped_warp.xml must exist under
    meshes/stl/, and the counts must match exactly (no orphans either way)."""
    tree = ET.parse(WARP_MODEL)
    root = tree.getroot()

    referenced = set()
    for mesh_elem in root.iter("mesh"):
        file_attr = mesh_elem.get("file")
        if file_attr:
            referenced.add(file_attr)

    assert referenced, "No <mesh file=...> elements found in biped_warp.xml"

    missing = []
    for filename in sorted(referenced):
        # file attr may include a subpath; just take the basename since
        # meshdir handles the directory part.
        candidate = MESH_DIR / Path(filename).name
        if not candidate.exists():
            missing.append(filename)

    assert not missing, (
        f"biped_warp.xml references mesh file(s) not present under "
        f"meshes/stl/: {missing}"
    )

    actual_stl_files = sorted(p.name for p in MESH_DIR.glob("*.stl"))
    assert len(referenced) == len(actual_stl_files), (
        f"Mesh count mismatch: biped_warp.xml references {len(referenced)} "
        f"mesh file(s) {sorted(referenced)}, but meshes/stl/ contains "
        f"{len(actual_stl_files)} .stl file(s) {actual_stl_files}. "
        "docs/colab_upload_manifest.md claims a 1:1, 13-file match - "
        "if this legitimately changed, update the manifest."
    )


def test_meshdir_resolves_relative_to_manifest_layout():
    """<compiler meshdir=...> must be exactly '../../meshes/stl/', the
    literal value the manifest's bundle-layout diagram depends on."""
    tree = ET.parse(WARP_MODEL)
    root = tree.getroot()
    compiler = root.find("compiler")
    assert compiler is not None, "biped_warp.xml has no <compiler> element"

    meshdir = compiler.get("meshdir")
    assert meshdir == "../../meshes/stl/", (
        f"Expected <compiler meshdir=\"../../meshes/stl/\">, got {meshdir!r}. "
        "docs/colab_upload_manifest.md's bundle layout diagram (models/mjcf/ "
        "two levels up, then into meshes/stl/) depends on this exact value - "
        "update the manifest if this legitimately changed."
    )


def test_mjlab_biped_imports_without_sim_package():
    """Regression test for the real bug found+fixed in this sub-task:
    mjlab_biped/__init__.py must tolerate sim/ being absent (as it will be
    in the minimal Colab bundle), evaluating BipedLocalEnv to None instead
    of crashing on import.

    Isolation must be genuine: naively running `sys.executable -c "import
    mjlab_biped"` with cwd set to an isolated temp dir does NOT catch a
    regression here, because this project's pyproject.toml
    ([tool.setuptools.packages.find] where=["."]) installs an *editable*
    "biped" package whose finder (a .pth file executed by the `site`
    module at interpreter startup) makes the top-level `sim` package
    importable regardless of the process's cwd or PYTHONPATH entries -
    this exact false negative was hit once already during pre-work on
    this sub-task (see docs/colab_upload_manifest.md).

    To get genuine isolation while still having numpy/mujoco available
    (mjlab_biped's non-local_env modules need them), the subprocess is
    launched with `-S` (skip automatic `site` processing, which is what
    executes .pth files, including the editable-install finder) and
    PYTHONPATH pointed directly at the venv's site-packages directory
    (a plain sys.path entry added via PYTHONPATH is not subjected to .pth
    processing, so numpy/mujoco import as ordinary packages but the
    editable-install finder never runs and `sim` is not importable).
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        shutil.copytree(MJLAB_BIPED_DIR, tmp_dir / "mjlab_biped")

        site_packages = sysconfig.get_paths()["purelib"]
        env = dict(os.environ)
        env["PYTHONPATH"] = site_packages
        # Make sure no ambient PYTHONPATH/editable hooks from the parent
        # process leak in beyond the deliberate site-packages entry above.
        env.pop("PYTHONNOUSERSITE", None)

        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import mjlab_biped; assert mjlab_biped.BipedLocalEnv is None; print('OK')",
            ],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )

        assert result.returncode == 0, (
            "import mjlab_biped failed in an isolated environment without "
            f"sim/ present.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout, (
            f"Expected 'OK' in subprocess stdout, got: {result.stdout!r} "
            f"(stderr: {result.stderr!r})"
        )

        # Sanity check the isolation itself: `sim` really must not be
        # importable in this environment, or the test above would be a
        # false negative regardless of what __init__.py does.
        sim_check = subprocess.run(
            [sys.executable, "-S", "-c", "import sim"],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert sim_check.returncode != 0, (
            "Isolation check failed: `import sim` unexpectedly succeeded in "
            "the isolated subprocess, which means this test cannot "
            "distinguish a correctly-guarded mjlab_biped/__init__.py from a "
            "broken one. sys.path leakage in the isolated subprocess needs "
            "to be fixed."
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_notebook_bundle_unpack_cell_references_manifest():
    """The notebook's bundle-unpack cell must no longer carry the
    TODO(7.A) marker and must reference docs/colab_upload_manifest.md."""
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)

    cells = nb["cells"]
    assert len(cells) >= 5, f"Expected at least 5 cells, found {len(cells)}"

    # Find the bundle-validation cell by looking for the distinctive check content
    # (looking for "expected_paths" which is only in that cell's code)
    bundle_cell = None
    for cell in cells:
        if cell["cell_type"] == "code":
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            if "expected_paths" in source and "colab_upload_manifest.md" in source:
                bundle_cell = cell
                break

    assert bundle_cell is not None, (
        "Could not find the bundle-validation cell in the notebook. "
        "Looking for a code cell containing 'expected_paths' and "
        "'colab_upload_manifest.md'."
    )

    source = bundle_cell["source"]
    if isinstance(source, list):
        source = "".join(source)

    assert "TODO(7.A)" not in source, (
        "The bundle-validation cell in notebooks/train_biped.ipynb still "
        "contains the TODO(7.A) marker - it should have been replaced with "
        "the real bundle check."
    )
    assert "colab_upload_manifest.md" in source, (
        "The bundle-validation cell in notebooks/train_biped.ipynb does not "
        "reference colab_upload_manifest.md by name."
    )
