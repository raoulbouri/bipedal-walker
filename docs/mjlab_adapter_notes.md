# mjlab Adapter Notes (Phase 7.C)

`mjlab_biped/mjlab_task.py` is the first code in this repo that imports
the *real* `mjlab` package. Every prior Phase 6/7 file (`entity.py`,
`observations.py`, `rewards.py`, `terminations.py`, `commands.py`,
`init_noise.py`, `domain_randomization.py`, `rl_cfg.py`, `config.py`)
deliberately stayed a **pure-Python mirror** of the mjlab API shape,
testable on the Mac without a GPU. `mjlab_task.py` is the translation
layer from that mirror into real mjlab manager-API objects.

**UPDATE 2026-07-11 — this file's claim that it "cannot be executed or
unit-tested on the Mac" was WRONG.** `mjlab`, `mujoco-warp`, and
`warp-lang` all ship real macOS ARM64 wheels on PyPI (`pip install
mjlab` installs cleanly on macOS, no CUDA needed) — the earlier belief
that Warp "excludes Darwin" referred to the *CUDA-enabled* build variant
only, not the base package. Warp's macOS build runs its CPU backend
(`wp.init()` reports `Devices: "cpu": "i386", CUDA not enabled in this
build`), and mjlab constructs and steps real `ManagerBasedRlEnv`
instances on that CPU backend without any special flag. **See "Local CPU
testing" below** — this is now the standard way to validate any change
to this file *before* burning a Colab session on it, and is how the
three bugs listed under "RESOLVED" below were actually found and fixed
(not guessed, not found in Colab).

## Local CPU testing (works — do this before every Colab run)

In a throwaway venv (never the Mac core `uv` env — `mjlab`/`torch`/
`mujoco-warp` must stay out of `pyproject.toml`'s core deps per
CLAUDE.md):

```bash
python3 -m venv /tmp/mjlab_cpu_test/.venv
source /tmp/mjlab_cpu_test/.venv/bin/activate
pip install mjlab   # installs mjlab, mujoco-warp, warp-lang, torch (CPU), rsl-rl-lib
cd /path/to/biped   # this repo
python3 -c "
import sys, torch
sys.path.insert(0, '.')
import mjlab_biped.mjlab_task as biped_task
print('Registered task:', biped_task.TASK_ID)

env_cfg = biped_task.make_biped_env_cfg(num_envs=1)
from mjlab.envs import ManagerBasedRlEnv
env = ManagerBasedRlEnv(cfg=env_cfg, device='cpu')
obs, extras = env.reset()
print('actor obs shape:', obs['actor'].shape)
print('critic obs shape:', obs['critic'].shape)
assert torch.isfinite(obs['actor']).all()
assert torch.isfinite(obs['critic']).all()
zero_action = torch.zeros(1, 6, device=obs['actor'].device)
obs, reward, terminated, truncated, extras = env.step(zero_action)
print('post-step reward:', reward)
print('Smoke test passed.')
"
```

First run takes ~25s (Warp JIT-compiles and caches its CPU kernels the
first time each is used, in `~/Library/Caches/warp/`); subsequent runs
are fast. This is the exact smoke-test cell from
`notebooks/train_biped.ipynb` (cell 7-8) — running it locally first
catches any `mjlab_task.py` API-guess bug for free, before it costs a
Colab session. **What this does NOT cover:** actual GPU-parallel training
(`num_envs` in the thousands), MuJoCo Warp's CUDA path specifically (the
CPU backend is a different code path and could in principle diverge —
low risk, not yet observed), and anything gated behind
`torch.cuda.is_available()` in the notebook's own cells (skip straight
to the task-registration/smoke-test cells locally, skip the `!nvidia-smi`
and CUDA-assert cells).

## What was verified before writing it (2026-07-11, direct source reads)

- `register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg, runner_cls=None)`
  — exact signature, from `src/mjlab/tasks/registry.py`.
- The full import block used in this adapter is copied verbatim (module
  paths only, not values) from `src/mjlab/tasks/cartpole/
  cartpole_env_cfg.py` — a real, working, minimal mjlab task. This is
  the single most trustworthy fact in this whole file: if these import
  paths are wrong, the cartpole example itself would be broken, which is
  extremely unlikely for mjlab's own shipped example.
- Field names for `SceneCfg`, `ObservationGroupCfg`
  (`terms`/`concatenate_terms`/`enable_corruption`), `RewardTermCfg`
  (`func`/`weight`/`params`), `TerminationTermCfg`
  (`func`/`params`/`time_out`), `EventTermCfg`
  (`func`/`mode`/`params`/`interval_range_s`), `JointPositionActionCfg`
  (`entity_name`/`actuator_names`/`scale`/`use_default_offset`),
  `SimulationCfg(mujoco=MujocoCfg(timestep=...))` — from
  `src/mjlab/tasks/velocity/velocity_env_cfg.py`.
- `train`/`play` are real console scripts installed by the `mjlab` PyPI
  package itself (`[project.scripts]` in mjlab's own `pyproject.toml`),
  confirming `uv run train <task_id>` is the correct invocation (not
  `python -m mjlab.scripts.train`, though that would also work).
- The train CLI is `train <task_id> [--env.* flags] [--agent.* flags]`
  (positional task id, then `tyro`-parsed nested flags) — from
  `src/mjlab/scripts/train.py`. The exact flag path for overriding
  `num_envs` (`--env.scene.num-envs N` vs something else) is itself
  UNVERIFIED — see below.

## UNVERIFIED — must be checked empirically on Colab before trusting a training run

Ranked by how early they'll surface (a smoke-test import/reset should
catch #1-3 immediately; #4-5 might only surface once training actually
starts producing NaN/garbage). **Update 2026-07-11: #1-4 are now
resolved** (found and fixed via local CPU testing, not Colab — see
above); only #5 (history stacking) and #6 (CLI flags) remain open.

1. ~~**Raw `<sensor>` readout on an mjlab `Entity`.**~~ **RESOLVED
   2026-07-11**, found via local CPU testing (see above), not Colab. The
   guessed `entity.data.sensor_data["<name>"]` doesn't exist at all
   (live `AttributeError: 'EntityData' object has no attribute
   'sensor_data'`). Real mechanism: mjlab has a first-class `Sensor`
   abstraction (`mjlab/sensor/`), and its `Scene._add_sensors()` method
   **auto-discovers every raw `<sensor>` element already compiled into
   the entity's own XML** (exactly the ones `postprocess.py` bakes into
   `biped_warp.xml` — `torso_gyro`, `torso_acc`, `touch_l`, `touch_r`,
   the 16 joint pos/vel sensors) and wraps each as
   `BuiltinSensor.from_existing(name)`. No `BuiltinSensorCfg` needs to be
   declared in Python for sensors that already exist in the XML — they
   just show up. Read via `env.scene.sensors[key].data`, where `key` is
   entity-prefixed (`"robot/torso_gyro"`, not `"torso_gyro"` — confirmed
   via a direct `Scene(cfg.scene, device="cpu").sensors.keys()` dump).
   `gyro()`, `accelerometer()`, `foot_touch()` in `mjlab_task.py` fixed
   accordingly.
2. ~~**`projected_gravity_b`, `root_quat_w`, `root_lin_vel_w`,
   `root_pos_w`, `com_pos_w` attribute names.**~~ **RESOLVED 2026-07-11.**
   Real `EntityData` (`mjlab/entity/data.py`) uses a `_link_` infix these
   guesses dropped: `root_link_pos_w`, `root_link_quat_w`,
   `root_link_lin_vel_w`, `root_link_ang_vel_w`. There is no
   `projected_gravity_b` attribute at all — mjlab, unlike IsaacLab,
   doesn't precompute it; `mjlab_task.py`'s `projected_gravity()` now
   derives it the same way IsaacLab does internally: rotate world-frame
   gravity into the body frame via `mjlab.utils.lab_api.math
   .quat_apply_inverse(root_link_quat_w, [0,0,-1])` (that helper is real
   and already used internally by `EntityData.root_com_lin_vel_b`).
   Whole-robot CoM is `root_com_pos_w`, backed by MuJoCo's
   `subtree_com` at the root body (the root's kinematic subtree is the
   whole robot, since it's the floating base) — confirmed correct, not
   a guess requiring a separate `subtreecom` XML sensor.
3. ~~**`env.action_manager.action` / `.prev_action`,
   `env.command_manager.get_command(...)`.**~~ **PARTIALLY RESOLVED
   2026-07-11.** `env.action_manager.action`/`.prev_action` are real,
   confirmed properties (`mjlab/managers/action_manager.py`) — no change
   needed. `env.command_manager.get_command(name)` is also a real method,
   but **`make_biped_env_cfg()` never passes a `commands=` dict to
   `ManagerBasedRlEnvCfg`**, so at runtime `env.command_manager` is
   mjlab's `NullCommandManager`, whose `get_command()` always returns
   `None` — this would have broken observation concatenation the moment
   `velocity_command()`/`command_tracking_fn()` ran, caught live via the
   local smoke test. Since the current gate is explicitly zero-command
   balance (`mjlab_biped/commands.py`'s `CommandRangeCfg()` defaults to
   `(0.0, 0.0)` on all three axes), both functions were changed to return
   a literal zero tensor — correct for this gate, but a real
   `CommandTermCfg` (resampling `commands.py`'s existing numpy sampler
   logic each episode) still needs wiring in before Phase 7's non-zero
   velocity curriculum. Track that as the new open item here when that
   curriculum work starts.
4. ~~**`RslRlModelCfg`'s exact field names.**~~ **RESOLVED 2026-07-11**
   — this was the first UNVERIFIED item to actually bite: the original
   guess (`actor_hidden_dims`/`critic_hidden_dims`/
   `actor_obs_normalization`/`critic_obs_normalization`/`init_noise_std`
   on one combined "policy" config) raised a live `TypeError:
   RslRlModelCfg.__init__() got an unexpected keyword argument
   'actor_hidden_dims'` on the very first `import mjlab_biped.mjlab_task`
   in Colab. Confirmed against `src/mjlab/rl/config.py` directly: mjlab's
   `RslRlOnPolicyRunnerCfg` has SEPARATE `actor: RslRlModelCfg` and
   `critic: RslRlModelCfg` fields, not one combined policy config.
   `RslRlModelCfg`'s real fields are `hidden_dims` (singular per
   network), `activation`, `obs_normalization` (singular bool), plus
   `cnn_cfg`/`distribution_cfg`/`rnn_type`/`rnn_hidden_dim`/
   `rnn_num_layers`/`class_name`. `init_noise_std` is NOT a direct field
   — it lives inside `distribution_cfg["init_std"]`, and only the actor
   needs a `distribution_cfg` (the critic has no action distribution,
   just a scalar value estimate). `obs_groups` values are tuples
   (`RslRlBaseRunnerCfg`'s own default:
   `{"actor": ("actor",), "critic": ("critic",)}`), not lists. Both
   `mjlab_task.py` and the Mac-side mirror `mjlab_biped/rl_cfg.py` (which
   had the same wrong structure, since it was built from the same wrong
   guess) were fixed to match this real shape — see `rl_cfg.py`'s module
   docstring for the corrected mirror.
5. **Observation history stacking mechanism** (the H=5 actor stack from
   Phase 7.pre). CLAUDE.md's Sub-task 7.B already flagged this as a
   required Sonnet pre-verification item before 7.C, and it remains
   open — `ACTOR_OBS_GROUP` in `mjlab_task.py` currently has NO stacking
   applied. **Dimension correction 2026-07-11:** the live smoke test
   reports the single-frame actor group as **28-dim, not 24-dim** as
   previously assumed — `joint_pos_rel`/`joint_vel_rel` each return 8
   values (6 actuated + 2 passive ankle joints), not 6, since mjlab's
   built-in term reports every hinge joint on the entity, not just the
   actuated ones. The frozen 120-dim (`24 * 5`) stacked-spec figure in
   `docs/observation_spec.md`/`mjlab_biped/observations.py`'s
   `STACKED_ACTOR_OBS_DIM` is now stale and needs reconciling against
   this real 28-dim per-frame count (`28 * 5 = 140`) before stacking is
   implemented. This is still the most consequential open item:
   **training must not start until this is resolved**, because a
   single-frame actor makes the task an unsolvable POMDP (see
   `docs/observation_spec.md`'s rationale for why the stack exists at
   all).
6. ~~**The exact `--env.*`/`--agent.*` CLI flag paths.**~~ **PARTIALLY
   RESOLVED 2026-07-12.** `--env.scene.num-envs N` and
   `--agent.max-iterations N` (the flags already used in the notebook's
   training cell) are confirmed real and accepted — verified locally by
   running `train`'s actual `tyro`-parsed CLI against the real biped task
   and confirming no "unrecognized argument" error for either flag (see
   item #7 below for the *task name* problem this same investigation
   found, which is a different failure mode entirely). Still open: the
   full space of other `--env.*`/`--agent.*` overrides beyond these two
   (e.g. `--agent.num-steps-per-env`, DR toggles) hasn't been
   individually exercised — `tyro` builds the flag surface directly from
   `TrainConfig`'s dataclass fields (`mjlab/scripts/train.py`), so any
   field name there is a reasonable guess, just not each one confirmed.

## Sub-task 7.C.1 (2026-07-12) — `train`/`play` CLI task discovery

**Real, live bug hit by the user in Colab**, not a guess:
`train Mjlab-Biped-Balance-v0` failed with `error: argument {...}:
invalid choice: 'Mjlab-Biped-Balance-v0'`, even though the user had
already confirmed `import mjlab_biped.mjlab_task` correctly registers
the task (`mjlab.tasks.registry.list_tasks()` includes it in a process
that has done that import). The user's own diagnosis was exactly right:
the `train`/`play` console scripts are fresh interpreters that never
import `mjlab_biped` themselves.

**Root cause, found by reading mjlab's actual source (not guessed):**
- `mjlab/scripts/train.py`'s `main()` (and `play.py`'s, identically)
  does `import mjlab.tasks` right before building the CLI's task-choice
  type from `list_tasks()`. That import only pulls in **mjlab's own
  built-in task packages** (`mjlab/tasks/__init__.py` calls
  `import_packages(__name__, ...)`, scanning mjlab's own subpackages
  only) — it has no knowledge of `mjlab_biped` at all.
- The *real* intended mechanism for external packages: `mjlab/__init__.py`
  (executed automatically the moment anything does `import mjlab`, which
  `train.py`'s own imports trigger) calls
  `_import_registered_packages()`, which does
  `entry_points().select(group="mjlab.tasks")` and imports whatever it
  finds — a genuine plugin system, but keyed off **installed package
  entry-point metadata** (a real `.dist-info/entry_points.txt`, the kind
  `pip install` produces from a package's `pyproject.toml`
  `[project.entry-points."mjlab.tasks"]` table).
- `mjlab_biped/` in the Colab bundle is deliberately **not** an installed
  package — it's an unzipped directory added to `sys.path`/cwd. It has
  no `pyproject.toml`, no wheel, no entry-point metadata. So
  `entry_points().select(group="mjlab.tasks")` finds nothing for it, and
  the task is never auto-registered before the CLI validates the task
  name — exactly reproducing the user's symptom.

**Fix:** `scripts/colab_train.py` and `scripts/colab_play.py` — tiny
driver scripts that `import mjlab_biped.mjlab_task` (registering the
task as a side effect, the same import the earlier notebook cell already
does) in the *same* Python process before calling mjlab's real
`train.py`/`play.py` `main()` function directly. This sidesteps the
entry-point gap entirely with zero packaging risk (no `pyproject.toml`
to write for `mjlab_biped`, no `pip install -e .` to get right, no new
UNVERIFIED surface). Same CLI flag syntax as the console scripts
themselves — only the invocation changes, from `train <task> <flags>`
to `python scripts/colab_train.py <task> <flags>`.

**Verified locally (macOS CPU, no GPU):** `python scripts/colab_train.py
Mjlab-Biped-Balance-v0 --env.scene.num-envs 2 --agent.max-iterations 1`
gets **past task selection entirely** (no "invalid choice") and fails
only at mjlab's own GPU-selection step (`IndexError: list index out of
range` in `select_gpus([0])`, because there is no GPU on this Mac) — the
expected, correct failure mode locally, and proof the fix resolves the
actual bug. `python scripts/colab_play.py Mjlab-Biped-Balance-v0`
likewise passes task selection and fails only on
`play.py`'s own downstream requirement for a real
`--checkpoint-file`/`--wandb-run-path`. On a real Colab GPU runtime,
`colab_train.py` should proceed into actual training.

**An alternative not taken:** packaging `mjlab_biped` as a real
pip-installable package with a `[project.entry-points."mjlab.tasks"]`
table, so the *bare* `train`/`play` console scripts would work unmodified.
Rejected for now as strictly more moving parts (a new `pyproject.toml`,
verifying `pip install -e .`'s editable-install entry-point metadata
actually gets picked up in a Colab environment) for no behavioral
difference — the driver-script fix is simpler and already verified.
Revisit only if there's a concrete reason the bare console scripts must
work unmodified (there isn't one currently).

## The Phase 7.C smoke-test-first plan (see the notebook)

This can now be (and should be) run locally first — see "Local CPU
testing" above. The notebook's first real cell after installing
dependencies is the same **minimal smoke test**: import
`mjlab_biped.mjlab_task`, construct the env cfg, build one real mjlab env
with `num_envs=1`, call `reset()`, call `step()` once with a zero action,
and print the actor/critic observation shapes. As of 2026-07-11 this
produces `(1, 28)` (actor, single-frame — see item #5's dimension
correction) and `(1, 43)` (critic) locally, not crashing, no NaN
*before* the real training cell runs on Colab. Any `AttributeError` here
points directly at one of the UNVERIFIED items above — fix that specific
function, re-run the smoke test, repeat, rather than debugging inside a
multi-hour training run.

## History-stacking fallback ladder (from CLAUDE.md's Phase 7 v4 plan,
restated here since it becomes actionable once mjlab's real API is
confirmed)

1. Find mjlab's native `history_length` (or equivalently-named) kwarg on
   `ObservationTermCfg`/`ObservationGroupCfg` and set it to 5.
2. If no native support: wrap `ACTOR_OBS_GROUP`'s functions in a small
   ring-buffer wrapper class (mirroring `mjlab_biped.observations
   .ObsHistory`'s already-tested logic, reimplemented in torch) that
   mjlab's observation manager can call each step.
3. If neither works cleanly: fall back to RSL-RL's
   `ActorCriticRecurrent` (LSTM) instead of stacking, and drop the
   fixed-length history entirely (the LSTM's hidden state does the same
   job implicitly).
