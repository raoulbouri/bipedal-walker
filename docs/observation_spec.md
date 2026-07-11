# Observation Spec (Phase 7.pre, v1)

Frozen term list, order, and scaling for the actor and critic observation
groups. This is the single source of truth other code (the observation
builder, RL training config, and the deferred Phase 7.D BipedSim parity
eval) must match exactly.

## Design decisions

- **No orientation estimate in the actor (v1 change).** The real robot has
  no sensor-fusion/EKF stack producing an orientation estimate, so the
  actor may no longer depend on `torso_quat`/`projected_gravity` at all —
  not even indirectly. It is a critic-only (privileged, training-time)
  term now.
- **No foot-contact sensing in the actor (v1 change).** No hardware
  provides `foot_touch` on the real robot, so it also moved to the
  critic-only group.
- **5-frame history stack on the actor (v1 change).** Because a single
  frame's raw gyro is angular *velocity*, not angle, dropping orientation
  from a single frame makes tilt unobservable within that frame. The
  actor's raw 24-dim per-frame observation is stacked into a 5-frame
  history (`ACTOR_HISTORY_LEN = 5`, oldest -> newest concatenation,
  `STACKED_ACTOR_OBS_DIM = 120`) so the policy network can integrate gyro
  into an implicit orientation estimate itself, rather than the sim
  providing one directly. `ObsHistory` (in `mjlab_biped/observations.py`)
  implements this as a pure numpy ring buffer — it is not a filter or
  estimator, just a fixed window of raw frames.
- **Accelerometer excluded from actor, included in critic.** Unchanged
  from v0: the accelerometer is noisy on real hardware (a modeled
  sim-to-real gap, see Phase 8), so the deployable actor policy must not
  depend on it. The privileged critic (training-time only, discarded at
  deployment) can use it since it only ever runs in sim.
- **Sampling semantics** (frozen since Phase 3): observations are read once
  per control step (50 Hz), at the control boundary, after the last physics
  substep of that step.
- **Previous action and velocity command are not MuJoCo sensors** — they
  come from environment/controller state (the action applied last step,
  and the currently-sampled command), not `data.sensordata`.

## Actor group (deployment-available only) — 24 dims per frame, 120 dims stacked

| # | Term | Dims | Source | Notes |
|---|------|------|--------|-------|
| 1 | `joint_pos_rel` | 6 | `pos_hip_roll_l`, `pos_knee_l`, `pos_ankle_l`, `pos_hip_roll_r`, `pos_knee_r`, `pos_ankle_r` | actuated joints only (not the passive `foot_l`/`foot_r`); relative to each joint's default (stand-keyframe) angle, which is 0 rad for every actuated joint, so numerically equal to the raw sensor value for this model |
| 2 | `joint_vel_rel` | 6 | `vel_hip_roll_l`, `vel_knee_l`, `vel_ankle_l`, `vel_hip_roll_r`, `vel_knee_r`, `vel_ankle_r` | actuated joints only, raw angular velocity |
| 3 | `gyro` | 3 | `torso_gyro` | raw body-frame angular velocity |
| 4 | `previous_action` | 6 | env/controller state | the action vector applied at the previous control step, same order as `joint_pos_rel`'s 6 actuated joints; zero-initialized on episode reset |
| 5 | `velocity_command` | 3 | command sampler | `[vx, vy, yaw_rate]`; fixed at `[0, 0, 0]` for the Phase 7 zero-command balance gate, widened later in the walking curriculum |

**Actor per-frame total: 6+6+3+6+3 = 24 dims.**

**Actor stacked total: 24 × `ACTOR_HISTORY_LEN` (5) = `STACKED_ACTOR_OBS_DIM` (120) dims.**
The stack is oldest -> newest concatenation of the 5 most recent per-frame
observations (`ObsHistory.reset()`/`.push()`); on episode reset all 5 slots
are filled with the first frame.

Deployment whitelist (sensor names the actor's underlying computation may
read): `pos_hip_roll_l`, `pos_knee_l`, `pos_ankle_l`, `pos_hip_roll_r`,
`pos_knee_r`, `pos_ankle_r`, `vel_hip_roll_l`, `vel_knee_l`, `vel_ankle_l`,
`vel_hip_roll_r`, `vel_knee_r`, `vel_ankle_r`, `torso_gyro`. Everything else
in the 24-sensor suite (`torso_quat`, `torso_acc`, `torso_pos`,
`torso_linvel`, `torso_angvel`, `touch_l`, `touch_r`, and the two passive
`pos_foot_l/r`/`vel_foot_l/r` sensors) is **not** read by the actor.

## Critic group (privileged, training-only) — 39 dims, single frame

Critic = all 5 actor terms (24 dims, single frame, **not** stacked) **plus**:

| # | Term | Dims | Source | Notes |
|---|------|------|--------|-------|
| 6 | `projected_gravity` | 3 | derived from `torso_quat` | world gravity direction `[0, 0, -1]` rotated into the torso body frame via the inverse of `torso_quat`; privileged in v1 (moved out of the actor) |
| 7 | `foot_touch` | 2 | `touch_l`, `touch_r` | raw touch sensor readings (not thresholded to boolean); privileged in v1 (moved out of the actor) |
| 8 | `accelerometer` | 3 | `torso_acc` | raw accelerometer (gravity-reaction + net acceleration); excluded from actor per the design decision above |
| 9 | `base_linvel` | 3 | `torso_linvel` | world-frame linear velocity of the torso site |
| 10 | `base_height` | 1 | `torso_pos[2]` | z-component only of the torso site's world position |
| 11 | `com` | 3 | whole-body center of mass | world-frame CoM position, mass-weighted over all bodies (same computation as `sim.BipedSim.com()`) |

**Critic total: 24 + 3+2+3+3+1+3 = 39 dims.**

The critic is single-frame (no history stacking) — it is training-only and
discarded at deployment, so there is no need for it to self-integrate
orientation the way the actor does.

## Term ordering

The actor's per-frame vector is a flat concatenation of terms 1–5 in the
table order above; the stacked vector concatenates 5 such frames,
oldest -> newest. The critic vector concatenates terms 1–5 (the actor's
own per-frame builder output, unstacked) then terms 6–11 in order. This
ordering is a golden-tested invariant — changing it is a breaking change
for any trained policy/checkpoint and must bump a spec version if it ever
happens.

## Privileged-leak guard

The actor observation builder's set of underlying sensor/state dependencies
must be a subset of the deployment whitelist above. In particular it must
**never** read `torso_quat`, `touch_l`, `touch_r`, `torso_acc`, `torso_pos`,
`torso_linvel`, `torso_angvel`, or compute CoM. This is enforced by an
automated test (`tests/test_observations.py`), not just this document.

## v0 (superseded 2026-07-11)

The original v0 spec gave the actor 29 dims/frame, including
`projected_gravity` (derived from `torso_quat`) and `foot_touch`, with no
history stacking (a single frame was sufficient because orientation was
directly observable). v1 removes both terms from the actor because the
real robot has neither an orientation estimate nor foot-contact hardware,
and compensates for the resulting loss of single-frame orientation
observability by stacking 5 frames of raw gyro (and the other actor terms)
so the policy can integrate tilt itself. The critic's total dimension count
is unchanged (39) because the two terms removed from the actor were simply
relocated into the critic-only privileged terms instead of being dropped
entirely. See `MEMORY.md`'s 2026-07-11 "Phase 7 plan v4" entry for the full
rationale and decision history.
