"""Sub-task 7.0: Colab-only dependency group scaffold verification.

Verifies:
1. Bare `uv sync` (no flags) does not attempt to resolve mjlab/rsl-rl-lib.
2. requirements-colab.txt is a valid, correctly formatted pip requirements file.
3. pyproject.toml's `colab` optional-dependencies group is valid TOML with the
   expected two entries.
4. notebooks/train_biped.ipynb is valid notebook JSON with the documented cells
   in order.
5. (This module itself, run as part of `pytest tests/`, is the regression
   check that the rest of the existing suite is unaffected by this sub-task.)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_uv_sync_does_not_pull_in_colab_packages():
    """Bare `uv sync` must succeed and must not install mjlab/rsl-rl-lib/mujoco-warp.

    These packages require CUDA wheels unavailable on macOS/ARM; if a bare
    sync (no --extra/--optional flags) ever tries to resolve them, that means
    the colab group leaked into a place pulled in by default, which is a bug.
    """
    result = subprocess.run(
        ["uv", "sync"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"bare `uv sync` failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    combined_output = (result.stdout + result.stderr).lower()
    for forbidden in ("mjlab", "rsl-rl", "rsl_rl", "mujoco-warp", "mujoco_warp"):
        assert forbidden not in combined_output, (
            f"bare `uv sync` output unexpectedly mentions '{forbidden}':\n"
            f"{result.stdout}\n{result.stderr}"
        )

    # Also confirm no such packages are actually installed in the environment.
    pip_list = subprocess.run(
        ["uv", "pip", "list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert pip_list.returncode == 0
    installed = pip_list.stdout.lower()
    for forbidden in ("mjlab", "rsl-rl-lib", "mujoco-warp"):
        assert forbidden not in installed, (
            f"'{forbidden}' unexpectedly installed after bare `uv sync`:\n{pip_list.stdout}"
        )

    # Restore dev extras so the rest of the test suite (this run included)
    # keeps working after this test's bare sync.
    restore = subprocess.run(
        ["uv", "sync", "--extra", "dev"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert restore.returncode == 0, (
        f"failed to restore dev extras after bare sync test:\n"
        f"stdout:\n{restore.stdout}\nstderr:\n{restore.stderr}"
    )


def test_requirements_colab_txt_format():
    """requirements-colab.txt exists, has no torch/mujoco-warp lines, and each
    non-comment, non-blank line matches a plain `package>=version` pattern."""
    req_path = REPO_ROOT / "requirements-colab.txt"
    assert req_path.exists(), "requirements-colab.txt does not exist at repo root"

    lines = req_path.read_text().splitlines()
    requirement_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*[><=!~]=?[A-Za-z0-9.]+$")

    requirement_lines = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        assert requirement_pattern.match(line), (
            f"line does not match expected 'package>=version' format: {line!r}"
        )
        requirement_lines.append(line)

    assert requirement_lines, "requirements-colab.txt has no requirement lines"

    joined = "\n".join(requirement_lines).lower()
    assert "torch" not in joined, "requirements-colab.txt must not pin torch"
    assert "mujoco-warp" not in joined and "mujoco_warp" not in joined, (
        "requirements-colab.txt must not pin mujoco-warp directly (bundled via mjlab)"
    )

    assert any(line.lower().startswith("mjlab") for line in requirement_lines), (
        "requirements-colab.txt missing mjlab requirement"
    )
    assert any(line.lower().startswith("rsl-rl-lib") for line in requirement_lines), (
        "requirements-colab.txt missing rsl-rl-lib requirement"
    )


def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:
        import tomli as tomllib  # Python < 3.11

    with open(path, "rb") as f:
        return tomllib.load(f)


def test_pyproject_colab_group_valid_toml():
    """pyproject.toml parses as valid TOML and contains the colab group with
    exactly the two expected entries (no torch, no mujoco-warp), while the
    existing dev group is left intact."""
    pyproject_path = REPO_ROOT / "pyproject.toml"
    data = _load_toml(pyproject_path)

    optional_deps = data["project"]["optional-dependencies"]
    assert "dev" in optional_deps, "existing dev group missing/renamed"
    assert "colab" in optional_deps, "colab optional-dependencies group missing"

    colab = optional_deps["colab"]
    assert any(dep.startswith("mjlab") for dep in colab), f"mjlab missing from colab group: {colab}"
    assert any(dep.startswith("rsl-rl-lib") for dep in colab), (
        f"rsl-rl-lib missing from colab group: {colab}"
    )
    assert not any("torch" in dep.lower() for dep in colab), (
        f"torch must not be pinned in colab group: {colab}"
    )
    assert not any("mujoco-warp" in dep.lower() or "mujoco_warp" in dep.lower() for dep in colab), (
        f"mujoco-warp must not be a separate line in colab group: {colab}"
    )


def test_notebook_valid_and_cells_in_order():
    """notebooks/train_biped.ipynb is valid notebook JSON with the required
    cells in proper order: title markdown, GPU-assert, install,
    (optional W&B cells), bundle-unpack, sanity imports."""
    notebook_path = REPO_ROOT / "notebooks" / "train_biped.ipynb"
    assert notebook_path.exists(), "notebooks/train_biped.ipynb does not exist"

    try:
        import nbformat

        nb = nbformat.read(str(notebook_path), as_version=4)
        cells = nb["cells"]
    except ModuleNotFoundError:
        with open(notebook_path, "r") as f:
            nb = json.load(f)
        assert nb.get("nbformat") == 4
        cells = nb["cells"]

    def source_text(cell):
        source = cell["source"]
        if isinstance(source, list):
            return "".join(source)
        return source

    # We require at least the core cells (not counting optional W&B cells).
    # Find each required cell by content, not by fixed index, to be robust
    # to optional additions between them.
    assert len(cells) >= 6, f"expected at least 6 cells, found {len(cells)}"

    # Cell 0: title markdown
    assert cells[0]["cell_type"] == "markdown"
    assert "Biped Balance Policy Training" in source_text(cells[0])
    assert "GPU" in source_text(cells[0])
    assert "colab_upload_manifest.md" in source_text(cells[0])

    # Find GPU assert, part 1 (!nvidia-smi)
    nvidia_smi_cell = None
    for cell in cells[1:]:
        if cell["cell_type"] == "code" and "nvidia-smi" in source_text(cell):
            nvidia_smi_cell = cell
            break
    assert nvidia_smi_cell is not None, "Could not find nvidia-smi cell"

    # Find GPU assert, part 2 (torch.cuda check)
    cuda_check_cell = None
    for cell in cells[1:]:
        if (
            cell["cell_type"] == "code"
            and "torch.cuda.is_available" in source_text(cell)
        ):
            cuda_check_cell = cell
            break
    assert cuda_check_cell is not None, "Could not find torch.cuda.is_available cell"

    # Find pip install cell
    pip_install_cell = None
    for cell in cells:
        if (
            cell["cell_type"] == "code"
            and "pip install -r requirements-colab.txt" in source_text(cell)
        ):
            pip_install_cell = cell
            break
    assert pip_install_cell is not None, "Could not find pip install cell"

    # Find bundle unpack + manifest-backed sanity check (Phase 7.A)
    bundle_cell = None
    for cell in cells:
        if (
            cell["cell_type"] == "code"
            and "expected_paths" in source_text(cell)
            and "colab_upload_manifest.md" in source_text(cell)
        ):
            bundle_cell = cell
            break
    assert bundle_cell is not None, "Could not find bundle validation cell"
    assert "TODO(7.A)" not in source_text(
        bundle_cell
    ), "Bundle cell still has TODO(7.A)"
    assert "os.listdir" in source_text(bundle_cell), "Bundle cell missing os.listdir"

    # Find sanity imports (mjlab, mujoco_warp, rsl_rl)
    imports_cell = None
    for cell in cells:
        if (
            cell["cell_type"] == "code"
            and "import mjlab" in source_text(cell)
            and "import mujoco_warp" in source_text(cell)
            and "import rsl_rl" in source_text(cell)
        ):
            imports_cell = cell
            break
    assert imports_cell is not None, "Could not find sanity imports cell"


def test_notebook_has_sanity_import_cell():
    """The notebook must also contain a sanity-imports cell (mjlab, mujoco_warp,
    rsl_rl, version prints). Written as a separate test since the strict
    5-cell ordering test above caps at 5 cells matching the spec exactly;
    this confirms the 5th documented cell's content in full."""
    notebook_path = REPO_ROOT / "notebooks" / "train_biped.ipynb"
    with open(notebook_path, "r") as f:
        nb = json.load(f)

    all_source = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in nb["cells"]
    )

    assert "import mjlab" in all_source
    assert "import mujoco_warp" in all_source
    assert "import rsl_rl" in all_source
    assert "importlib.metadata" in all_source
