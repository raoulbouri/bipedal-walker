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
│   ├── rl_cfg.py          (Phase 7.B — not yet created as of 7.A)
│   ├── local_env.py       (present but inert on Colab, see note below)
│   └── README.md
├── requirements-colab.txt
└── notebooks/
    └── train_biped.ipynb
```

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
| `models/mjcf/biped_warp.xml` | The Warp-compatible model (implicit integrator, primitive collisions, Phase 6.0) that `mjlab.MjSpec.from_file(...)` loads. |
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
