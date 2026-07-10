# Observation Spec (Phase 6.B)

Frozen term list, order, and scaling for the actor and critic observation
groups. This is the single source of truth other code (the observation
builder, RL training config, and the deferred Phase 7.D BipedSim parity
eval) must match exactly.

## Design decisions

- **Projected gravity, not raw quaternion.** The actor sees the gravity
  vector rotated into the torso's body frame (3 numbers), not `torso_quat`
  directly. This is the standard locomotion-RL choice: it's rotation-
  invariant about yaw in the way that matters for balance, and avoids the
  raw-quaternion double-cover sign ambiguity (`q` and `-q` are the same
  orientation but differ componentwise).
- **Accelerometer excluded from actor, included in critic.** The
  accelerometer is noisy on real hardware (a modeled sim-to-real gap, see
  Phase 8), so the deployable actor policy must not depend on it. The
  privileged critic (training-time only, discarded at deployment) can use
  it since it only ever runs in sim.
- **No history stacking in v0.** Each term is the current control-step's
  value only; no stacked past frames. Can be added later without changing
  this spec's term *names*, only by wrapping the builder.
- **Sampling semantics** (frozen since Phase 3): observations are read once
  per control step (50 Hz), at the control boundary, after the last physics
  substep of that step.
- **Previous action and velocity command are not MuJoCo sensors** — they
  come from environment/controller state (the action applied last step,
  and the currently-sampled command), not `data.sensordata`.

## Actor group (deployment-available only) — 29 dims

| # | Term | Dims | Source | Notes |
|---|------|------|--------|-------|
| 1 | `joint_pos_rel` | 6 | `pos_hip_roll_l`, `pos_knee_l`, `pos_ankle_l`, `pos_hip_roll_r`, `pos_knee_r`, `pos_ankle_r` | actuated joints only (not the passive `foot_l`/`foot_r`); relative to each joint's default (stand-keyframe) angle, which is 0 rad for every actuated joint, so numerically equal to the raw sensor value for this model |
| 2 | `joint_vel_rel` | 6 | `vel_hip_roll_l`, `vel_knee_l`, `vel_ankle_l`, `vel_hip_roll_r`, `vel_knee_r`, `vel_ankle_r` | actuated joints only, raw angular velocity |
| 3 | `projected_gravity` | 3 | derived from `torso_quat` | world gravity direction `[0, 0, -1]` rotated into the torso body frame via the inverse of `torso_quat` |
| 4 | `gyro` | 3 | `torso_gyro` | raw body-frame angular velocity |
| 5 | `foot_touch` | 2 | `touch_l`, `touch_r` | raw touch sensor readings (not thresholded to boolean) |
| 6 | `previous_action` | 6 | env/controller state | the action vector applied at the previous control step, same order as `joint_pos_rel`'s 6 actuated joints; zero-initialized on episode reset |
| 7 | `velocity_command` | 3 | command sampler | `[vx, vy, yaw_rate]`; fixed at `[0, 0, 0]` for the Phase 7 zero-command balance gate, widened later in the walking curriculum |

**Actor total: 6+6+3+3+2+6+3 = 29 dims.**

Deployment whitelist (sensor names the actor's underlying computation may
read): `pos_hip_roll_l`, `pos_knee_l`, `pos_ankle_l`, `pos_hip_roll_r`,
`pos_knee_r`, `pos_ankle_r`, `vel_hip_roll_l`, `vel_knee_l`, `vel_ankle_l`,
`vel_hip_roll_r`, `vel_knee_r`, `vel_ankle_r`, `torso_quat` (consumed only
to derive `projected_gravity`, never exposed raw), `torso_gyro`,
`touch_l`, `touch_r`. Everything else in the 24-sensor suite
(`torso_acc`, `torso_pos`, `torso_linvel`, `torso_angvel`, and the two
passive `pos_foot_l/r`/`vel_foot_l/r` sensors) is **not** read by the actor.

## Critic group (privileged, training-only) — 39 dims

Critic = all 7 actor terms (29 dims) **plus**:

| # | Term | Dims | Source | Notes |
|---|------|------|--------|-------|
| 8 | `accelerometer` | 3 | `torso_acc` | raw accelerometer (gravity-reaction + net acceleration); excluded from actor per the design decision above |
| 9 | `base_linvel` | 3 | `torso_linvel` | world-frame linear velocity of the torso site |
| 10 | `base_height` | 1 | `torso_pos[2]` | z-component only of the torso site's world position |
| 11 | `com` | 3 | whole-body center of mass | world-frame CoM position, mass-weighted over all bodies (same computation as `sim.BipedSim.com()`) |

**Critic total: 29 + 3+3+1+3 = 39 dims.**

## Term ordering

Both groups are flat-concatenated vectors in the exact table order above
(actor: terms 1–7 in order; critic: terms 1–7 then 8–11 in order). This
ordering is a golden-tested invariant — changing it is a breaking change
for any trained policy/checkpoint and must bump a spec version if it ever
happens.

## Privileged-leak guard

The actor observation builder's set of underlying sensor/state dependencies
must be a subset of the deployment whitelist above. In particular it must
**never** read `torso_acc`, `torso_pos`, `torso_linvel`, `torso_angvel`, or
compute CoM. This is enforced by an automated test
(`tests/test_observations.py`), not just this document.
