"""Colab entry point for training the biped task, in place of the bare `train` console script.

**Why this exists (found 2026-07-12, via the user's own live Colab error):**
mjlab's `train`/`play` console scripts only auto-discover external task
packages through a Python entry-point group, `"mjlab.tasks"` (see
`mjlab/__init__.py`'s `_import_registered_packages()`, which runs
`entry_points().select(group="mjlab.tasks")` unconditionally the moment
`import mjlab` happens). That mechanism only finds *properly pip-installed*
packages carrying entry-point metadata (a `.dist-info/entry_points.txt`).
`mjlab_biped/` ships in the Colab bundle as a bare, uninstalled directory
(no `pyproject.toml`, no wheel) -- so it is invisible to that discovery
step. Confirmed live: `python -c "from mjlab.tasks.registry import
list_tasks"` in a fresh process only lists mjlab's own built-in tasks;
only a process that has *already* executed `import mjlab_biped.mjlab_task`
sees `"Mjlab-Biped-Balance-v0"`. The bare `train Mjlab-Biped-Balance-v0`
CLI invocation is exactly such a fresh process, hence `error: argument
{...}: invalid choice: 'Mjlab-Biped-Balance-v0'`.

**The fix:** import `mjlab_biped.mjlab_task` (registering the task as a
side effect, same as it always has) in the *same* Python process, before
calling mjlab's own `train.py:main()` function directly -- bypassing the
entry-point discovery gap entirely, with zero packaging risk. Verified
locally (macOS CPU): this reaches mjlab's real GPU-selection step (fails
only there, for the expected reason -- no GPU on the Mac) instead of
"invalid choice", proving task selection now succeeds.

Usage (same CLI syntax as the `train` console script itself):
    python scripts/colab_train.py Mjlab-Biped-Balance-v0 \
        --env.scene.num-envs 64 --agent.max-iterations 200
"""

import sys
from pathlib import Path

# Bundle layout has mjlab_biped/ as a sibling of scripts/ (repo root) --
# ensure it's importable regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mjlab_biped.mjlab_task  # noqa: E402,F401  (registers the task as a side effect)

from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
