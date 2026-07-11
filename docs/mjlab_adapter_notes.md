# mjlab Adapter Notes (Phase 7.C)

`mjlab_biped/mjlab_task.py` is the first code in this repo that imports
the *real* `mjlab` package. Every prior Phase 6/7 file (`entity.py`,
`observations.py`, `rewards.py`, `terminations.py`, `commands.py`,
`init_noise.py`, `domain_randomization.py`, `rl_cfg.py`, `config.py`)
deliberately stayed a **pure-Python mirror** of the mjlab API shape,
testable on the Mac without a GPU. `mjlab_task.py` is the translation
layer from that mirror into real mjlab manager-API objects, and it
**cannot be executed or unit-tested on the Mac** — there is no `mjlab`
install here (Colab-only dependency, see Phase 7.0) and no GPU.

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
starts producing NaN/garbage):

1. **Raw `<sensor>` readout on an mjlab `Entity`.** `gyro()`,
   `accelerometer()`, `foot_touch()` guess
   `entity.data.sensor_data["<name>"]`. This is the single biggest
   unknown — none of the fetched mjlab examples (cartpole, velocity,
   manipulation) needed raw IMU/touch sensor access, so there was no
   real example to copy. **First thing to check in Colab:** load
   `biped_warp.xml` into a real `mjlab.entity.Entity`, step it once, and
   run `dir(entity.data)` / inspect its actual attributes interactively
   before trusting these three functions.
2. **`projected_gravity_b`, `root_quat_w`, `root_lin_vel_w`,
   `root_pos_w`, `com_pos_w` attribute names** on `entity.data`. These
   follow the Isaac-Lab-style naming convention (`_w` = world frame,
   `_b` = body frame) that mjlab's manager API is explicitly modeled
   after, so they are a reasonable guess, but not confirmed against
   mjlab's own source (the fetched files didn't need body-pose access
   in a way that showed these specific names).
3. **`env.action_manager.action` / `.prev_action`, `env.command_manager
   .get_command("base_velocity")`.** Reasonable Isaac-Lab-convention
   guesses, not confirmed against mjlab source directly.
4. **`RslRlModelCfg`'s exact field names.** This adapter assumes it
   mirrors the fields IsaacLab's `RslRlPpoActorCriticCfg` documents
   (`actor_hidden_dims`, `critic_hidden_dims`, `activation`,
   `actor_obs_normalization`, `critic_obs_normalization`,
   `init_noise_std`) — but mjlab's own class is named differently
   (`RslRlModelCfg`, not `RslRlPpoActorCriticCfg`), so mjlab may be a
   genuinely separate implementation with different field names, not
   just a rename. Confirmed to exist and be importable from `mjlab.rl`;
   field names not confirmed.
5. **Observation history stacking mechanism** (the H=5 actor stack from
   Phase 7.pre). CLAUDE.md's Sub-task 7.B already flagged this as a
   required Sonnet pre-verification item before 7.C, and it remains
   open — `ACTOR_OBS_GROUP` in `mjlab_task.py` currently has NO stacking
   applied (single-frame 24-dim, not the frozen 120-dim spec). This is
   the most consequential open item: **training must not start until
   this is resolved**, because a single-frame actor makes the task an
   unsolvable POMDP (see `docs/observation_spec.md`'s rationale for why
   the stack exists at all).
6. **The exact `--env.*`/`--agent.*` CLI flag paths** for overriding
   `num_envs`, `max_iterations`, etc. from the command line (vs editing
   `mjlab_task.py`'s Python defaults directly, which always works
   regardless of CLI flag names).

## The Phase 7.C smoke-test-first plan (see the notebook)

Because none of the above can be checked from the Mac, the notebook's
first real cell after installing dependencies is a **minimal smoke
test**: import `mjlab_biped.mjlab_task`, construct the env cfg, build
one real mjlab env with `num_envs=1`, call `reset()`, call `step()` once
with a zero action, and print the actor/critic observation shapes. This
must produce `(1, 24)`-or-`(1, 120)` and `(1, 39)` (not crash, not NaN)
*before* the real training cell runs. Any `AttributeError` here points
directly at one of the UNVERIFIED items above — fix that specific
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
