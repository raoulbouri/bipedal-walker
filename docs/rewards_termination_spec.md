# Reward + Termination Stubs Spec (Phase 6.D)

Frozen structure for the reward and termination functions. Per CLAUDE.md's
Phase 6.D scope, these are **deliberately minimal placeholders** — the
user finalizes the actual reward shaping and curriculum in Phase 7. This
spec fixes the *structure* (term list, signature, batching convention,
termination thresholds) so Phase 6.E can assemble around it without
rework; every weight is a tunable default marked for Phase 7 revision.

## Batching convention (no torch)

mjlab/RSL-RL (Colab-only, not a Mac core dep) will eventually vectorize
these over `num_envs` using torch tensors. Since Phase 6 is developed and
tested on the Mac (CPU, numpy only), every reward/termination function
here is written as **pure numpy**, accepting and returning arrays with a
leading `(num_envs,)` batch dimension (or `(num_envs, D)` for vector
inputs). This satisfies CLAUDE.md's "each reward term returns a finite
per-env tensor of shape `(num_envs,)`" requirement without adding a torch
dependency; wrapping these in torch at the mjlab boundary in Phase 7 is a
trivial `torch.from_numpy`/dtype cast, not a logic change.

## Inputs shared across all reward/termination functions

All functions take a subset of these batched arrays (shapes shown for
`num_envs = N`):

| Name | Shape | Source |
|---|---|---|
| `torso_quat` | `(N, 4)` | `torso_quat` sensor, `[w,x,y,z]` MuJoCo convention |
| `base_height` | `(N,)` | `torso_pos[..., 2]` sensor (z-component only) |
| `base_linvel` | `(N, 3)` | `torso_linvel` sensor |
| `action` | `(N, 6)` | current control-step's action (6 actuated joints) |
| `previous_action` | `(N, 6)` | previous control-step's action |
| `velocity_command` | `(N, 3)` | `[vx, vy, yaw_rate]`, from Phase 6.C's command sampler |

## Upright measure (shared helper, reused by reward + termination)

`upright(torso_quat) -> (N,)`: `1 - 2*(qx**2 + qy**2)`, where
`torso_quat = [qw, qx, qy, qz]`. This is the closed-form `(2,2)` entry of
the rotation matrix implied by a unit quaternion — the world-frame
z-component of the torso's own local z-axis. Range `[-1, 1]`:
`+1` = perfectly upright, `0` = torso horizontal (90° tilt), `-1` = upside
down. Verified independently against Phase 6.B's `project_gravity` test
quaternions before dispatch: identity → `1.0` exactly; the 90°-about-Y
test quaternion `[cos45°, 0, sin45°, 0]` → `~0.0` (matches the expected
"fully sideways" case).

## Reward terms — `mjlab_biped/rewards.py`

Each term is a separate function `term_name(...) -> np.ndarray` of shape
`(N,)`, plus a `RewardCfg` dataclass holding the weights and a
`compute_reward(cfg, ...) -> np.ndarray` that combines them. Every weight
is marked `# TODO(user): finalize in Phase 7` in the dataclass — these
defaults are reasonable locomotion-RL starting points, not tuned values.

| Term | Formula | Default weight | Notes |
|---|---|---|---|
| `alive_bonus` | constant `1.0` per env, every step | `+1.0` | flat per-step survival bonus |
| `upright` | `upright(torso_quat)` | `+1.0` | see helper above; range `[-1,1]` |
| `command_tracking` | `-norm(base_linvel[:, :2] - velocity_command[:, :2])` (xy linear velocity tracking only; yaw-rate tracking deferred) | `0.0` | **zero-weight placeholder** — command range is `{0}` for the Phase 7 balance-first gate, so this term computes a real tracking error (ready for the walking curriculum) but contributes nothing to the reward until the user raises its weight |
| `control_effort` | `-sum(action**2, axis=-1)` | `-0.001` | small quadratic effort penalty |
| `action_rate` | `-sum((action - previous_action)**2, axis=-1)` | `-0.01` | anti-vibration/high-frequency penalty |

`compute_reward` returns `sum(weight_i * term_i for all i)`, shape `(N,)`.
Must be finite (`np.all(np.isfinite(reward))`) for any finite input.

## Termination — `mjlab_biped/terminations.py`

Two conditions, each a function returning a boolean `(N,)` array:

| Term | Condition | Default threshold | Notes |
|---|---|---|---|
| `fall_tilt` | `upright(torso_quat) < tilt_threshold` | `tilt_threshold = 0.5` (≈60° from vertical) | fires when torso tilts more than ~60° from upright |
| `fall_height` | `base_height < height_threshold` | `height_threshold = 0.15` m | below `STAND_HEIGHT=0.2030` but safely above the ~0.045-0.053m passive-settle-on-the-floor height measured in Phases 1-2, so a genuinely fallen robot triggers this while a crouching-but-still-up robot does not |
| `fall` | `fall_tilt OR fall_height` | — | combined fall termination |
| `time_out` | `step_count >= max_episode_steps` | config, no physical default (episode-length dependent) | truncation, not a fall; kept as a separate flag so downstream code can distinguish truncation from failure |

A `TerminationCfg` dataclass holds `tilt_threshold`, `height_threshold`,
`max_episode_steps` (all marked `# TODO(user): finalize in Phase 7` where
the roadmap explicitly says so — the fall thresholds are physically
motivated defaults from Phases 0-2 baselines, not placeholders in the same
sense as the reward weights, but still open to Phase 7 retuning against
the actual trained policy's behavior).

## Verification requirements

1. Every reward term and `compute_reward` returns a finite `(N,)` array
   for batch sizes `N=1` and `N>1` (e.g. `N=4`), given randomized-but-finite
   inputs.
2. `fall` does NOT fire at the `stand` keyframe's exact state (upright=1.0,
   height=0.2030) — the boundary case the roadmap explicitly requires.
3. `fall_tilt` fires on a manually-injected tilted state (e.g., the 90°
   quaternion from Phase 6.B's tests, upright≈0 < 0.5) even at nominal
   height; `fall_height` fires on a manually-injected low-height state
   (e.g., height=0.05, matching the Phase 1/2 passive-settle figure) even
   at upright orientation. Each condition must be tested independently
   (not just the combined `fall`) so a bug in one doesn't hide behind the
   other.
3. Threshold boundary tests: `upright` exactly at `tilt_threshold` and
   `base_height` exactly at `height_threshold` must NOT fire (strict `<`,
   not `<=`) — test both a value just above and just below each threshold.
