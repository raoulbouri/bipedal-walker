"""Colab entry point for evaluating a checkpoint, in place of the bare `play` console script.

Same fix as `scripts/colab_train.py` -- see that file's module docstring
for the full explanation of why the bare `play` console script can't see
`mjlab_biped`'s task registration. `play.py`'s `main()` has the identical
`import mjlab.tasks` + `list_tasks()`-driven task-selection structure as
`train.py`'s, so the same fix (import the task-registering module in the
same process, then call mjlab's own main() directly) applies unchanged.

Usage (same CLI syntax as the `play` console script itself):
    python scripts/colab_play.py Mjlab-Biped-Balance-v0 \
        --checkpoint-file logs/rsl_rl/.../model_200.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mjlab_biped.mjlab_task  # noqa: E402,F401  (registers the task as a side effect)

from mjlab.scripts.play import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
