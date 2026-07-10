# Env Config Assembly + Task Registration Spec (Phase 6.E)

Frozen structure for assembling the full environment config out of Phases
6.A-6.D's pieces, and for "registering" it — since real `mjlab` is
Colab-only (not a Mac core dep), this phase builds a **local registry**
that mirrors mjlab's `register_mjlab_task` API shape closely enough that
Phase 7's Colab wiring is a thin substitution, not a rewrite.

## `SceneCfg`

| Field | Default | Notes |
|---|---|---|
| `num_envs` | `4096` | large GPU-parallel default; overridden via CLI at Colab launch time (`--env.scene.num-envs N`), per CLAUDE.md's `uv run train ... --env.scene.num-envs 64` smoke example |
| `env_spacing` | `2.0` (m) | grid spacing between parallel envs, irrelevant on a single local CPU env but part of the frozen shape |
| `terrain_type` | `"plane"` | matches the flat-floor `biped_warp.xml` |
| `entity_cfg` | `BipedEntityCfg()` (Phase 6.A) | the robot entity |

## `SimCfg`

| Field | Default | Notes |
|---|---|---|
| `timestep` | `0.002` | frozen since Phase 0, matches `biped_warp.xml` |
| `integrator` | `"implicit"` | frozen since Phase 6.0 |
| `decimation` | `10` | physics substeps per control step; `decimation * timestep == 0.02` (50 Hz control rate) is a hard invariant, tested |

## `BipedEnvCfg`

Bundles everything: `scene: SceneCfg`, `sim: SimCfg`,
`reward_cfg: RewardCfg` (Phase 6.D), `termination_cfg: TerminationCfg`
(Phase 6.D), `command_cfg: CommandRangeCfg` (Phase 6.C),
`init_noise_cfg: InitNoiseCfg` (Phase 6.C),
`domain_randomization_cfg: DomainRandomizationCfg` (Phase 6.C, `enabled`
defaults to its own Phase 6.C default of `False`), `episode_length_s: float
= 20.0`.

`__post_init__` validates: `sim.decimation * sim.timestep` is within
`1e-9` of `0.02` (the frozen 50 Hz control rate) — fail loudly on
construction if violated, same pattern as `BipedEntityCfg`'s own
`control_dt`/`control_decimation` check from Phase 6.A.

## `play_env_cfg` variant

A second `BipedEnvCfg` instance for local/eval play: `scene.num_envs = 4`
(small), `domain_randomization_cfg = DomainRandomizationCfg(enabled=False)`
explicitly (even though `False` is already the Phase 6.C default, set it
explicitly here so the play variant's intent is self-documenting and
doesn't silently rely on an upstream default that could change).

## Local task registry

Since `mjlab.register_mjlab_task` doesn't exist locally, build a minimal
substitute with the same essential shape:

```python
TASK_REGISTRY: dict[str, dict] = {}

def register_mjlab_task(task_id: str, env_cfg, play_env_cfg, rl_cfg=None) -> None:
    """Register a task by id. Raises if task_id is already registered
    (no silent overwrite - this would be a genuine bug if it happened
    at Colab-registration time too)."""
    ...

def get_task_cfg(task_id: str) -> dict:
    """Look up a registered task's cfg dict. Raises KeyError with a
    clear message (including the list of registered ids) if not found."""
    ...
```

Register exactly one task at module import time:
`register_mjlab_task("Mjlab-Biped-Balance-v0", env_cfg=BipedEnvCfg(), play_env_cfg=<the play variant above>, rl_cfg=None)`
(`rl_cfg` is `None` here — Phase 7.B is the RSL-RL PPO config, out of
scope for 6.E, deliberately left as a `None` placeholder field so the
registry's shape doesn't need to change when 7.B lands).

## Verification requirements

1. `"Mjlab-Biped-Balance-v0"` is registered and discoverable via
   `get_task_cfg(...)` immediately after importing `mjlab_biped`.
2. `BipedEnvCfg()` constructs with no missing fields (dataclass
   defaults all resolve).
3. The `decimation * timestep == 0.02` invariant is enforced — both the
   valid default case and a deliberately-broken case (e.g.
   `SimCfg(decimation=7)`) are tested, the broken case must raise.
4. The registered task's `play_env_cfg` has `scene.num_envs < 
   env_cfg.scene.num_envs` (smaller) and
   `domain_randomization_cfg.enabled is False`.
5. Registering the same `task_id` twice raises (no silent overwrite).
6. Looking up an unregistered task_id raises `KeyError` with a message
   that includes the actually-registered ids (a debugging convenience,
   not just a bare `KeyError`).

## Deferred to Phase 7

- Wiring `BipedEnvCfg` into mjlab's real `ManagerBasedRlEnvCfg` and
  `register_mjlab_task` on Colab.
- `rl_cfg` (Phase 7.B, RSL-RL PPO hyperparameters).
- Actually running `uv run train`/`uv run play` — that's a live Colab-GPU
  step, out of scope for this Mac-side config-assembly phase.

## Note: local single-env visualization (separate from this spec)

This phase's `BipedEnvCfg`/registry is a **vectorized-shaped config**,
never actually steppable locally (no vectorized CPU env exists or is
planned - mjlab/Warp owns that). A **separate**, non-mjlab-shaped
single-environment wrapper (`mjlab_biped/local_env.py`) is being built
alongside this phase (not part of the frozen registry/config shape above)
purely so the assembled observation/reward/termination/command/noise
pieces can be sanity-checked and visually inspected on macOS via the
existing `mujoco.viewer` + `mjpython` workflow, ahead of the real Colab
training run. It intentionally does not appear in `TASK_REGISTRY`.
