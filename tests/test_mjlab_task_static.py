"""Phase 7.C: static-analysis-only tests for mjlab_biped/mjlab_task.py.

mjlab_task.py imports the real mjlab package (and torch transitively) and
is a Colab-only module -- it cannot be imported or executed on this Mac
(no mjlab install, no GPU; see docs/mjlab_adapter_notes.md). Every test
here therefore treats the file as TEXT/AST, never `import`s it, matching
the style already established in tests/test_colab_manifest.py for the
sibling "cannot run this on the Mac" problem.

This file also exercises scripts/package_colab_bundle.py (a plain,
runnable, stdlib-only script) and cross-checks it against
docs/colab_upload_manifest.md for drift.
"""
import ast
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MJLAB_TASK_PATH = REPO_ROOT / "mjlab_biped" / "mjlab_task.py"
MANIFEST_PATH = REPO_ROOT / "docs" / "colab_upload_manifest.md"
PACKAGE_SCRIPT_PATH = REPO_ROOT / "scripts" / "package_colab_bundle.py"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "train_biped.ipynb"


def _mjlab_task_source() -> str:
    return MJLAB_TASK_PATH.read_text()


def test_mjlab_task_is_valid_python_syntax():
    """mjlab_task.py cannot be imported on the Mac (no mjlab package), so
    the only thing verifiable here is that it is syntactically valid
    Python -- ast.parse must not raise SyntaxError."""
    source = _mjlab_task_source()
    try:
        ast.parse(source, filename=str(MJLAB_TASK_PATH))
    except SyntaxError as exc:
        raise AssertionError(
            f"mjlab_biped/mjlab_task.py has invalid Python syntax: {exc}"
        )


def test_mjlab_task_imports_frozen_constants_not_hardcoded():
    """mjlab_task.py must translate the already-frozen, already-tested
    pure-Python specs (entity.py, rewards.py, terminations.py, rl_cfg.py,
    config.py) into real mjlab objects -- it must import those classes,
    not redefine their values inline. Same "must import, don't hardcode"
    pattern used in tests/test_rl_cfg.py's dimension-consistency test."""
    source = _mjlab_task_source()

    required_substrings = [
        "from mjlab_biped.entity import BipedEntityCfg",
        "from mjlab_biped.rewards import RewardCfg",
        "from mjlab_biped.terminations import TerminationCfg",
        "from mjlab_biped.rl_cfg import",
        "from mjlab_biped.config import",
    ]
    for substr in required_substrings:
        assert substr in source, (
            f"mjlab_biped/mjlab_task.py is missing the expected import "
            f"{substr!r} -- it should reuse the frozen pure-Python specs "
            "rather than hardcoding their values."
        )


def test_mjlab_task_referenced_local_files_exist():
    """The model path literal referenced in mjlab_task.py (via
    entity.py's BipedEntityCfg().model_path, and matching the same
    literal checked in tests/test_colab_manifest.py) must correspond to a
    real file relative to the repo root."""
    entity_path = REPO_ROOT / "mjlab_biped" / "entity.py"
    entity_src = entity_path.read_text()

    model_literal = "models/mjcf/biped_warp.xml"
    assert model_literal in entity_src, (
        f"{model_literal!r} not found in mjlab_biped/entity.py -- "
        "mjlab_task.py's _get_biped_spec() loads the model via "
        "BipedEntityCfg().model_path, which is expected to resolve to "
        "this literal path."
    )

    resolved = REPO_ROOT / model_literal
    assert resolved.exists(), (
        f"Referenced model file {model_literal!r} does not exist at "
        f"{resolved}"
    )


def _package_script_bundle_files() -> list[str]:
    """Load scripts/package_colab_bundle.py's BUNDLE_FILES constant by
    importing the script as a standalone module (it has no third-party
    deps, so this is safe/cheap and more robust than re-parsing source)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "package_colab_bundle", PACKAGE_SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.BUNDLE_FILES)


def _manifest_mjlab_biped_py_files() -> set[str]:
    """Extract every mjlab_biped/*.py filename mentioned in the manifest's
    "Bundle contents" tree diagram (the fenced code block right after that
    heading) -- deliberately scoped to that one block so mentions of e.g.
    tests/test_colab_manifest.py elsewhere in the doc's prose don't leak
    in as false positives."""
    import re

    manifest_text = MANIFEST_PATH.read_text()
    fence_match = re.search(r"```\n(.*?)\n```", manifest_text, re.DOTALL)
    assert fence_match, "No fenced tree-diagram code block found in the manifest"
    tree_block = fence_match.group(1)

    # Only take lines that are inside the mjlab_biped/ subtree section.
    in_mjlab_biped = False
    names = set()
    for line in tree_block.splitlines():
        stripped = line.strip()
        if "mjlab_biped/" in stripped:
            in_mjlab_biped = True
            continue
        if in_mjlab_biped:
            if stripped.startswith("├──") or stripped.startswith("│") or stripped.startswith("└──"):
                match = re.search(r"([A-Za-z0-9_]+\.py)", stripped)
                if match:
                    names.add(match.group(1))
            if stripped.startswith("└──") or (not stripped) or ("docs/" in stripped):
                # A new top-level tree entry ends the mjlab_biped/ subtree.
                if "docs/" in stripped:
                    in_mjlab_biped = False
    return names


def test_bundle_manifest_and_packaging_script_agree():
    """Cross-check scripts/package_colab_bundle.py's file list against
    docs/colab_upload_manifest.md's tree diagram: every mjlab_biped/*.py
    file mentioned in the manifest must appear in the packaging script's
    list, and vice versa -- catches drift if either is updated without
    the other."""
    script_files = _package_script_bundle_files()
    script_py_names = {
        Path(p).name for p in script_files if p.startswith("mjlab_biped/") and p.endswith(".py")
    }

    manifest_py_names = _manifest_mjlab_biped_py_files()

    missing_from_script = manifest_py_names - script_py_names
    assert not missing_from_script, (
        f"docs/colab_upload_manifest.md mentions mjlab_biped/*.py file(s) "
        f"not present in scripts/package_colab_bundle.py's BUNDLE_FILES: "
        f"{sorted(missing_from_script)}"
    )

    missing_from_manifest = script_py_names - manifest_py_names
    assert not missing_from_manifest, (
        f"scripts/package_colab_bundle.py's BUNDLE_FILES includes "
        f"mjlab_biped/*.py file(s) not mentioned in "
        f"docs/colab_upload_manifest.md's tree diagram: "
        f"{sorted(missing_from_manifest)}"
    )


def test_package_bundle_script_produces_expected_zip_contents():
    """Actually run scripts/package_colab_bundle.py and inspect the
    resulting zip's contents for the key expected entries."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        output_zip = tmp_dir / "colab_bundle.zip"
        result = subprocess.run(
            [sys.executable, str(PACKAGE_SCRIPT_PATH), str(output_zip)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"package_colab_bundle.py failed: stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
        assert output_zip.exists(), "Expected output zip was not created"

        with zipfile.ZipFile(output_zip) as zf:
            names = set(zf.namelist())

        assert "models/mjcf/biped_warp.xml" in names

        mesh_files = [
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
        ]
        assert len(mesh_files) == 13
        for mesh_path in mesh_files:
            assert mesh_path in names, f"Missing mesh {mesh_path} in zip"

        assert "mjlab_biped/mjlab_task.py" in names
        assert "notebooks/train_biped.ipynb" in names
        assert "requirements-colab.txt" in names
    finally:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_notebook_has_smoke_test_before_training_cell():
    """Verify the smoke-test-before-training ordering CLAUDE.md's Sub-task
    7.C plan describes is actually present in the notebook, not just
    described in prose: the training cell must come AFTER a smoke-test
    cell (containing both 'reset' and 'assert').

    2026-07-12: the training cell now invokes `scripts/colab_train.py`
    rather than the bare `train` console script directly (see
    docs/mjlab_adapter_notes.md's "train/play CLI task discovery" section
    -- the bare console script can't see mjlab_biped's task registration
    at all, since it only auto-discovers external tasks via an
    entry-point mechanism that requires mjlab_biped to be a properly
    pip-installed package, which this bundle deliberately is not)."""
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)

    cells = nb["cells"]

    def cell_source(cell) -> str:
        source = cell.get("source", "")
        if isinstance(source, list):
            return "".join(source)
        return source

    train_idx = None
    for i, cell in enumerate(cells):
        if "colab_train.py" in cell_source(cell):
            train_idx = i
            break

    assert train_idx is not None, (
        "No cell containing the training invocation 'colab_train.py' "
        "found in notebooks/train_biped.ipynb"
    )

    smoke_idx = None
    for i, cell in enumerate(cells):
        src = cell_source(cell)
        if "reset" in src and "assert" in src:
            smoke_idx = i
            break

    assert smoke_idx is not None, (
        "No cell containing both 'reset' and 'assert' (the smoke-test "
        "cell) found in notebooks/train_biped.ipynb"
    )

    assert smoke_idx < train_idx, (
        f"Expected the smoke-test cell (index {smoke_idx}) to appear "
        f"before the training cell (index {train_idx}), but it does not."
    )
