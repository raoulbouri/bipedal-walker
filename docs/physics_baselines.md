# Physics Baselines

## Zero-Control Settle Baseline (biped.xml, 5.0 seconds)

### Description

This baseline replaces the previously-documented settle baseline (base z ≈ 0.053m) that predated the Phase 0 torque derating (±2.5 N·m) and STAND_HEIGHT recalibration (0.2030m), and was measured with an ambiguous "ctrl=0" interpretation.

This new baseline uses `BipedSim` with explicit zero control held via zero-order hold across all substeps for exactly 5.0 simulated seconds (250 control steps at control_dt=0.02s), starting from the `stand` keyframe on `biped.xml`.

### Measured Values

- **Final base z-height**: 0.052619 m
- **Minimum z-height (over full trajectory)**: 0.044228 m
- **Maximum z-height (over full trajectory)**: 0.203384 m
- **Max |qvel| at t=4.0s (step 200)**: 0.000309 rad/s
- **Max |qvel| at t=5.0s (final step)**: 0.000311 rad/s
- **Final contact count**: 4 active contacts

### Control Model

With position actuators and `ctrl=0`, the "drop" from initial stand height is not fully passive. The actuators servo toward 0 rad targets, which happen to coincide with the stand keyframe's joint angles. This is a deliberate, documented choice for this baseline, not an oversight.

### Test Configuration

- Model: `models/mjcf/biped.xml` (Jetson variant, primary model per roadmap)
- Control dt: 0.02 s (20 ms)
- Physics substeps per control step: 10 (2 ms each)
- Simulation duration: 5.0 seconds
- Control input: All zeros (zero-order hold)
- Initial state: Stand keyframe

## Phase 2 Physics Validation Baselines (2026-07-10)

All measurements on `biped.xml`, from `tests/test_physics.py`.

### Free-fall (gravity check)
Elevated to z=2.0m from `stand` keyframe, zero velocity/ctrl, stepped 0.3s (150 steps
at dt=0.002s). Zero contacts throughout. Max relative error vs analytic
`z(t)=z0-0.5*g*t^2`: **0.188829%** at t=0.3s, against a tolerance of **0.25%** (not
the originally-planned 0.1% - confirmed via a control run with actuators fully
disabled that this is pure `implicitfast` integrator discretization error at
dt=0.002s over 0.3s, not a bug or actuator-coupling artifact).

### Inertia sanity
Total mass 0.784kg confirmed. Exactly one body (`root`, the freejoint carrier) has
zero mass - a known, previously-documented structural fact (all real mass lives in
child bodies), not an anomaly. All other 34 bodies have positive mass and
principal inertias satisfying the triangle inequality. Mass matrix at the `stand`
pose is symmetric and positive-definite (via `mj_fullM`).

### Knee pendulum energy & period
Isolated single-DOF-equivalent rig (ankle_l + foot_l subtree reparented onto a
fixed anchor at the real `motor` body's stand-pose world transform, no floor, no
other bodies - avoids spurious contact confounds from a naive full-robot weld).
Released from 0.3 rad, dissipation zeroed. Effective inertia (via `mj_fullM`)
I_eff=1.332752e-04 kg·m². Effective stiffness (finite-differenced `qfrc_bias`)
k_est=-1.872412e-02 N·m/rad. Analytic period T=2π√(I/k)=**0.530095s**. Measured
period (peak-to-peak, 9 peaks over 5s): **0.534250s** (0.78% discrepancy, within
5% tolerance). Energy drift over 5s: **0.9956%** (within 2% tolerance).

### Contact quality at rest
Settled `stand` pose, 4 active contacts. Penetration depths 0.230-0.435mm (all
< 2mm tolerance). Sum of contact normal forces: 7.691037N vs expected weight
7.691040N (mass×g) - **0.0000% error**. Contact count stable at exactly 4
throughout a 1s post-settle window (no chattering).

### Friction breakaway (redesigned test)
The roadmap's original design (ramped force at the full robot's CoM, <1mm/s creep
below 0.8×μN) was found to be confounded: this robot has no balance controller and
marginal stability, so pushing the full articulated robot causes TOPPLING (up to
28cm/s asymmetric foot drift at supposedly sub-threshold force), not clean
foot-floor sliding - two different failure modes the original design conflated.
Redesigned as an isolated single-foot-slider rig (foot_1 body, its real mesh/mass/
friction, freejoint-mounted on a copy of the real floor). Drift over 1s at
0.8×/1.0×/1.2×μN (μ=1.0, N=0.0883N): **13.208mm / 96.749mm / 993.274mm** - a clean
75.2× breakaway ratio (well above the 5× threshold used as the qualitative
signature, since MuJoCo's soft-contact creep doesn't match an absolute <1mm/s
"no-slip" criterion).

### Timestep sensitivity
Settled base z-height (2s passive settle from `stand`) at dt=2ms/1ms/0.5ms:
**0.05261003 / 0.05259610 / 0.05260308 m** - maximum pairwise disagreement
0.0265%, far inside the 10% tolerance. Confirms dt=2ms with `implicitfast` is
well within the converged regime.

### Long-horizon (30s) sanity
Energy at t=0/1s/15s/30s: **1.240335 / 0.354517 / 0.354652 / 0.347871 J**. No
NaN/Inf across 15000 steps. ~54% of individual steps show a tiny energy uptick
(expected contact-solver noise in a regularized constraint solver, max single-step
increase **2.597e-3**, well under the 0.01 bound) - NOT a bug, so the test does not
assert strict per-step monotonic decrease. Instead: max single-step increase
bounded, AND the windowed-maximum envelope (0.5s/250-step windows) confirmed
non-increasing throughout, correctly capturing "dissipates to a floor, never
diverges" while tolerating expected micro-noise.
