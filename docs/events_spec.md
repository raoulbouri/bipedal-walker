# Command + Events Spec (Phase 6.C)

Frozen configuration for the velocity command sampler, init-state
randomization, and domain randomization (DR). All three are "events" in
mjlab's terminology (`EventTermCfg`) — this doc specifies what they sample,
their default ranges, and the reset semantics, ahead of the Colab-only
`mjlab` wiring in Phase 7.

## Design decisions

- **RNG ownership**: every sampler here takes a `numpy.random.Generator`
  (constructed via `np.random.default_rng(seed)`) as an explicit argument
  — no module-level or hidden global RNG state. This matches the Phase 1
  harness design ("a single `numpy.random.Generator` owned by the harness,
  seeded explicitly").
- **No separate `biped_warp_no_jetson.xml`.** Per instruction, only one
  Warp model variant exists. The Jetson on/off DR axis is implemented by
  mutating the *already-compiled* `biped_warp.xml` model's torso body mass
  and inertia in-memory between the two frozen constant sets already used
  by `scripts/postprocess.py` for `biped.xml`/`biped_no_jetson.xml`
  (`TORSO_BASE=0.099kg`+`JETSON=0.180kg`→0.279kg vs `TORSO_BASE=0.099kg`
  alone), rather than loading a second XML file.
- **No leakage across episodes.** Every DR "apply" call resamples factors
  from the ORIGINAL (pristine, never-before-mutated) baseline arrays
  captured once at model-load time, not from the model's current (possibly
  already-randomized) state. This makes repeated resets non-compounding by
  construction — apply(seed_A) then apply(seed_B) must produce exactly the
  same result as a fresh apply(seed_B) from baseline.
- **Command range default is exactly `{0}` (all three axes)** for the
  Phase 7 zero-command balance gate. The sampler itself supports a general
  `(low, high)` range per axis so it's ready for the later walking
  curriculum without a code change — only a config change.

## Velocity command sampler

Samples `(vx, vy, yaw_rate)` once per episode reset from independent
uniform ranges.

| Axis | Default range (balance-first gate) | Units |
|---|---|---|
| `vx` | `(0.0, 0.0)` | m/s |
| `vy` | `(0.0, 0.0)` | m/s |
| `yaw_rate` | `(0.0, 0.0)` | rad/s |

A zero-width range must always sample exactly the range's value (no
floating-point drift). Widening these ranges is the Phase 7 walking
curriculum's job — a config change only, not a sampler code change.

## Init-state randomization

Applied once per episode reset, added on top of the `stand` keyframe's
nominal pose (all 8 hinge joints at 0 rad, base at `[0, 0, 0.2030]`,
identity orientation).

| Perturbation | Range | Notes |
|---|---|---|
| Joint angle noise | `±0.02 rad` uniform, independent per actuated joint (6 joints) | Matches the Phase 4 steady-state-error scale — small enough not to violate joint limits or start the episode already falling, large enough to give the policy pose diversity |
| Base xy position noise | `±0.01 m` uniform, independent x/y | Small horizontal jitter |
| Base z position noise | `±0.003 m` uniform | Smaller than xy since foot contact is height-sensitive (Phase 0's STAND_HEIGHT calibration margin was sub-mm to a few mm) |
| Base orientation noise | tilt angle magnitude `U(0, 0.05 rad)` (~0-2.9°) about a uniformly random 3D axis, composed with the identity stand orientation via quaternion multiplication | Small tilt, not a full random orientation — the robot must still start near-upright |

The two passive foot joints (`foot_l/r`) are NOT randomized — they are
unactuated and their resting angle is a physical consequence of ground
contact, not a controllable initial condition.

## Domain randomization (DR)

Each DR term is a multiplicative factor sampled per episode reset and
applied to the corresponding array(s) of a **fresh copy of the pristine
baseline** (captured once at load time, before any DR has ever been
applied).

| Term | Range (multiplicative factor) | Applies to |
|---|---|---|
| Body mass | `U(0.9, 1.1)` per body, independent | `model.body_mass[i]` for every body with `i >= 1` (skip worldbody); inertia is scaled by the same factor as the Phase 0/2 pipeline convention (`di * (new_mass/old_mass)`) to stay physically consistent |
| Friction | `U(0.7, 1.3)`, one shared factor applied to all geom frictions | `model.geom_friction[:, 0]` (sliding friction only, first column) for every geom with nonzero baseline friction |
| Actuator kp/kv | `U(0.8, 1.2)`, independent factor per actuator per gain | see implementation note below — for all 6 actuators |
| Jetson on/off | Bernoulli(0.5) discrete choice | Torso body (`composite_part_1__1_`) mass + diagonal inertia, toggled between `TORSO_BASE=0.099` (inertia `"2.1e-4 1.8e-4 1.4e-4"`) and `TORSO_BASE+JETSON=0.279` (inertia `"6e-4 5e-4 4e-4"`) — these are the exact constants from `scripts/postprocess.py`, kept in sync manually (documented here, not imported, to avoid a runtime dependency from `mjlab_biped` on `scripts/`) |

**Default: DR is OFF** (all factors fixed at 1.0 / Jetson fixed at whatever
the loaded model already has) for the Phase 7 first training gate. Turned
on for the Phase 7 Stage 3 robustness gate — this is a config flag, not a
code change.

### Implementation note: kp/kv storage in the compiled model

Verified directly against the compiled `biped_warp.xml` (MuJoCo `position`
actuator): `actuator_gainprm[i] = [kp, 0, 0]` and
`actuator_biasprm[i] = [0, -kp, -kv]` — **kp appears in two places**
(`gainprm[0]` as `+kp` and `biasprm[1]` as `-kp`), because MuJoCo computes
`force = gainprm[0]*ctrl + biasprm[1]*length + biasprm[2]*velocity =
kp*(target - length) - kv*velocity`. Randomizing kp must scale **both**
`gainprm[i, 0]` and `biasprm[i, 1]` by the *same* factor (preserving the
`biasprm[1] == -gainprm[0]` invariant), or the actuator's physics silently
breaks (the ctrl-term and length-term kp would disagree). Randomizing kv
only touches `biasprm[i, 2]`.

### Guard against leakage

The randomizer must store the pristine baseline arrays (mass, friction,
gains, torso inertia) once, at construction time, before any DR is ever
applied. Every subsequent `apply(rng)` call must:
1. Start from the stored baseline (a copy, never the model's live/mutated
   state).
2. Resample fresh factors from `rng`.
3. Write `baseline * factor` (or the discrete Jetson choice) directly into
   the live model's arrays — overwriting, not multiplying into whatever
   was there from a previous `apply()` call.

This is tested directly: `apply(rng_A)` then `apply(rng_B)` must produce
bitwise-identical arrays to constructing a fresh randomizer and calling
`apply(rng_B)` once.

## Deferred to Phase 6.E / Phase 7

- Wiring these samplers into mjlab's actual `EventTermCfg` — this phase
  only builds the pure-Python samplers and their config, testable locally
  against a raw `MjModel`.
- Statistical acceptance thresholds for "samples match the configured
  distribution over N resets" — this phase's tests check per-sample bounds
  and mean/std sanity over a moderate N (documented per-test), not a
  formal goodness-of-fit test.
