# Colab Upload Bundle Manifest (Phase 7.A)

The exact, minimal set of files to upload to Colab before running
`notebooks/train_biped.ipynb`. Every entry below was determined by static
analysis of what `mjlab_biped/` actually imports/opens (`grep`'d directly,
not assumed) — see the "How this was verified" section at the bottom.

## Bundle contents

```
bundle-root/
├── models/
│   └── mjcf/
│       └── biped_warp.xml
├── meshes/
│   └── stl/
│       ├── __1.stl
│       ├── __2.stl
│       ├── _.stl
│       ├── Foot.stl
│       ├── Hip_Base.stl
│       ├── Hip_Joint_A.stl
│       ├── Hip_Joint_B.stl
│       ├── Motor.stl
│       ├── Part_1.stl
│       ├── Tibia.stl
│       ├── Tube.stl
│       ├── Upper_Leg_A.stl
│       └── Upper_Leg_B.stl
├── mjlab_biped/
│   ├── __init__.py
│   ├── entity.py
│   ├── observations.py
│   ├── commands.py
│   ├── init_noise.py
│   ├── domain_randomization.py
│   ├── rewards.py
│   ├── terminations.py
│   ├── config.py
│   ├── rl_cfg.py
│   ├── mjlab_task.py      (Phase 7.C — imports real mjlab, Colab-only)
│   ├── recovery.py        (Phase 7.R.3 — fallen-pose reset + recovery reward)
│   ├── local_env.py       (present but inert on Colab, see note below)
│   └── README.md
├── docs/
│   └── mjlab_adapter_notes.md   (Phase 7.C — read before debugging mjlab_task.py)
├── requirements-colab.txt
├── notebooks/
│   ├── train_biped.ipynb
│   └── view_biped.ipynb   (added 2026-07-12 — live parallel-envs viser viewer)
└── scripts/
    ├── colab_train.py   (added 2026-07-12 — use instead of the bare `train` CLI)
    └── colab_play.py    (added 2026-07-12 — use instead of the bare `play` CLI)
```

**`notebooks/view_biped.ipynb`, added 2026-07-12:** a separate companion
to `train_biped.ipynb` (does not replace it) that embeds mjlab's **viser**
web viewer directly in the Colab output as an iframe — the same technique
as mjlab's own demo notebook (`serve_kernel_port_as_iframe`), but pointed
at `Mjlab-Biped-Balance-v0`. Lets you watch **N parallel copies** of the
robot at once: `agent="zero"/"random"` needs no checkpoint (verifies
parallel rendering before training), `agent="trained"` replays the newest
`logs/rsl_rl/**/model_*.pt` checkpoint. It launches the viewer via
`scripts/colab_play.py` as a background subprocess, pins the port with
viser's `_VISER_PORT_OVERRIDE`, and parses the actually-bound port from
viser's `listening *:<port>` startup log. Verified end-to-end locally
(macOS CPU, real mjlab 1.5.0): the viewer subprocess binds the port and
accepts TCP connections. Train with thousands of envs (GPU parallelism);
view with ~16 (browser clarity).

**`scripts/colab_train.py`/`colab_play.py`, added 2026-07-12:** the bare
`train`/`play` console scripts cannot see `mjlab_biped`'s task
registration — they only auto-discover external tasks through a
`"mjlab.tasks"` Python entry-point group, which requires a properly
pip-installed package; this bundle deliberately ships `mjlab_biped/` as
an unzipped directory, not an installed package, so it's invisible to
that mechanism. Confirmed live (a real user's Colab session hit `error:
invalid choice: 'Mjlab-Biped-Balance-v0'`) and root-caused via direct
inspection of mjlab's own source
(`mjlab/__init__.py::_import_registered_packages()`,
`mjlab/scripts/train.py`/`play.py`'s `main()`). These two driver scripts
import `mjlab_biped.mjlab_task` (registering the task as a side effect)
in the same process before calling mjlab's real CLI entry point — same
flags as the console scripts themselves, just invoked via `python
scripts/colab_train.py <task> <flags...>` instead of `train <task>
<flags...>`. See `docs/mjlab_adapter_notes.md`'s "train/play CLI task
discovery" section for the full writeup.

**Only `biped_warp.xml`** — there is no `biped_warp_no_jetson.xml`; the
Jetson mass on/off axis is an in-memory `DomainRandomizer` toggle (Phase
6.C), not a second model file.

**Layout matters:** `biped_warp.xml`'s `<compiler meshdir="../../meshes/stl/">`
is relative to the XML's own location. The `models/mjcf/` ↔ `meshes/stl/`
relative nesting above must be preserved exactly (two levels up from
`models/mjcf/`, then into `meshes/stl/`) — flattening the bundle would
break mesh loading. `mjlab_biped/entity.py` and `local_env.py` both
reference the model via the literal relative path
`"models/mjcf/biped_warp.xml"`, so the **notebook's working directory at
import time must be `bundle-root/`** (the parent of `models/`).

**`sim/` is deliberately NOT part of the bundle.** `mjlab_biped/local_env.py`
(the macOS-only single-env visualization companion, not the training path)
imports `sim.biped_sim.BipedSim`. `mjlab_biped/__init__.py` wraps that
import in a `try/except ModuleNotFoundError` (fixed in this sub-task,
2026-07-11 — this was a real bug caught by writing this manifest: the
`__init__.py` used to import `local_env` unconditionally, which would have
made `import mjlab_biped` crash on Colab with `sim/` absent). With the fix,
`mjlab_biped.BipedLocalEnv` simply evaluates to `None` on Colab, and every
other name in the package (the actual Colab-training-relevant surface —
`entity`, `observations`, `commands`, `init_noise`,
`domain_randomization`, `rewards`, `terminations`, `config`) imports
normally. `local_env.py` is included in the bundle anyway (it's a small
file and costs nothing to upload), but it is never imported/used during
Colab training.

## What each piece is for

| Path | Purpose |
|---|---|
| `models/mjcf/biped_warp.xml` | The Warp-compatible model (`implicitfast` integrator, full mesh collision -- physically identical to `biped.xml` as of 2026-07-13, see docs/warp_model.md) that `mjlab.MjSpec.from_file(...)` loads. |
| `meshes/stl/*.stl` | The 13 mesh assets `biped_warp.xml` references by relative path. |
| `mjlab_biped/` | The task package: robot entity, observation/reward/termination/command/DR managers, env config assembly, and (once 7.B lands) the RSL-RL runner config. This is what gets wired into mjlab's real `register_mjlab_task`. |
| `requirements-colab.txt` | Colab pip-install pins (`mjlab`, `rsl-rl-lib`; deliberately no `torch` pin, see Phase 7.0). |
| `notebooks/train_biped.ipynb` | The training notebook itself. |

## How this was verified (static analysis, 2026-07-11)

Ran directly against the actual source, not assumed from memory:

```
grep -rn "models/mjcf\|meshes/stl\|\.xml\|\.stl" mjlab_biped/*.py
```
→ the only model path referenced anywhere in `mjlab_biped/` is the
literal string `"models/mjcf/biped_warp.xml"` (in `entity.py` and
`local_env.py`).

```
grep -n "meshdir" models/mjcf/biped_warp.xml
grep -o 'file="[^"]*"' models/mjcf/biped_warp.xml | sort -u
```
→ `meshdir="../../meshes/stl/"`; exactly 13 unique mesh filenames
referenced.

```
ls meshes/stl/*.stl | wc -l
```
→ exactly 13 files present, a 1:1 match with the 13 referenced above (no
orphaned/missing mesh files).

```
grep -n "^import\|^from" mjlab_biped/__init__.py mjlab_biped/local_env.py
```
→ found `mjlab_biped/__init__.py` unconditionally imported `local_env`,
which does `from sim.biped_sim import BipedSim` (an absolute import of the
top-level `sim/` package, not part of this bundle). **This is what caught
the bug fixed above** — confirmed both failure and fix with a genuinely
isolated Python venv (no editable install of the parent `biped` package,
which would otherwise make `sim` importable regardless of the bundle
layout and mask the bug): `import mjlab_biped` raised
`ModuleNotFoundError: No module named 'sim'` before the fix, and succeeds
with `mjlab_biped.BipedLocalEnv is None` after it.

## Not yet verified (deferred, real Colab step)

The manifest's completeness is proven by the static analysis above and a
pytest regression test (`tests/test_colab_manifest.py`) that re-runs the
same greps and fails if the source ever references a path not on this
list. What is NOT provable on macOS: actually resolving and importing
`mjlab`/`rsl_rl`/`mujoco_warp` themselves (CUDA-only wheels) — that is the
Phase 7.A "☁️ smoke train" step, run for real on a Colab GPU runtime once
this bundle is uploaded.
