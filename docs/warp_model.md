# biped_warp.xml: MuJoCo Warp-Compatible Model Variant

## Overview

`biped_warp.xml` is a variant of the main `biped.xml` model, designed to be compatible with **MuJoCo Warp** (the GPU physics backend used for large-scale RL training on Google Colab). As of 2026-07-13, it is **physically identical** to `biped.xml` — same integrator, same mesh-on-mesh collision, same masses, sensors, and actuators. There is no longer any intentional physics difference between the training-time model and the CPU/deployment-representative model (see "Collision Geometry" below for why the two variants exist as separate files at all despite being physically the same).

## Key Differences from CPU Variant

### Integrator: ImplicitFast (corrected 2026-07-11 — previously wrongly set to "implicit")

- **CPU variant (`biped.xml`)**: uses `implicitfast` integrator
- **Warp variant (`biped_warp.xml`)**: also uses `implicitfast` integrator (matches CPU exactly)

**Correction, 2026-07-11:** this doc previously stated the Warp variant used `"implicit"` because "MuJoCo Warp does not support `implicitfast`." That premise was never verified against a real mjlab install and turned out to be wrong: a live local CPU install of `mjlab`/`mujoco-warp` shows mjlab's own integrator map (`mjlab/sim/sim.py` `_INTEGRATOR_MAP`) only recognizes `"euler"` and `"implicitfast"` — `"implicit"` isn't a valid option at all and raises `KeyError('implicit')` the moment an env is constructed. This would have failed on Colab too, just later than the smoke test would have caught it. Fixed by switching `biped_warp.xml` to `"implicitfast"`, which is strictly better than the previous plan anyway: it now matches `biped.xml`'s CPU-validated integrator exactly, eliminating one whole axis of sim-to-sim drift between the Phase 0-4 validated model and the model actually trained against.

### Collision Geometry: Mesh Throughout (corrected 2026-07-13 — previously primitive proxies)

- **CPU variant**: All body geometry is full-resolution mesh collision (from `biped_raw.xml`, set via Stage 2's whole-body collision enablement)
- **Warp variant**: Now also full-resolution mesh collision on every body — identical to the CPU variant.

**Correction, 2026-07-13:** this doc previously specified primitive collision proxies (capsule/box/sphere) for non-foot bodies, on the theory that "MuJoCo Warp's mesh collision support is memory-intensive and computationally expensive when scaling to thousands of parallel environments." That premise was never verified against real mujoco_warp — checked directly and found mujoco_warp's own test fixtures (`test_data/aloha_pot/*.obj`) use genuine mesh collision geoms, confirming it works there, not just in theory. For this specific model (only ~33 small, simple bodies), the primitive substitution introduced a real, previously undiagnosed physics discrepancy between the training-time model and the deployment-representative CPU model — discovered while evaluating a real trained checkpoint (`scripts/eval_checkpoint.py`): the policy survived far longer on the old primitive `biped_warp.xml` than on `biped.xml`, an unexplained divergence at the time.

**Fix:** `postprocess.py`'s `add_warp_variant()` now uses the exact same whole-body-collision logic as `build()` (flip `contype`/`conaffinity`/`friction` on the existing visual mesh geom, add no new geometry) instead of appending primitive proxy geoms. Verified this makes the two models produce **bit-identical passive-drop trajectories** over a full 5 s settle (`tests/test_warp_compat.py::test_warp_matches_cpu_settle`, max qpos/qvel diff < 1e-6, empirically exactly 0.0) — they are now the same physics, not just "close."

**If this becomes a real memory/throughput problem on actual Colab GPU hardware at scale** (unverifiable from this Mac — Warp's CPU backend here confirms correctness, not GPU memory behavior at large `num_envs`), the fallback is **decomposed multi-primitive proxies per body** (several capsules per limb, not one crude one), not reverting to the previous single-primitive-per-body scheme.

**Diagnostic finding from the checkpoint that motivated this fix (not resolved by the fix itself):** re-running `Model 499.pt` (trained under the old primitive-collision physics) against the corrected mesh-collision model shows it now falls *faster* (0.10s) than it did on its own old training environment (0.50s) — confirming the models are now physically consistent with each other (this checkpoint's behavior on the new `biped_warp.xml` and on `biped.xml` is identical), but also revealing that this specific checkpoint learned behavior tied to the old primitive dynamics that doesn't transfer even to its own corrected training environment. Re-training against the corrected `biped_warp.xml` is necessary before drawing conclusions about policy quality — this checkpoint was never trained under the physics it's now being evaluated against.

## Verified Properties

All Phases 0–4 backward-compatibility gates have been re-validated for `biped_warp.xml`:

✓ **Structure**: 36 bodies (incl. worldbody), 15 DOF (7 floating base + 8 hinges), 6 actuators, 24 sensors  
✓ **Masses**: Total 0.784 kg (with Jetson Nano), identical to CPU variant  
✓ **Integrator**: `implicitfast` at dt=0.002 s (corrected 2026-07-11, was `implicit`)  
✓ **Static feasibility**: Stand keyframe achieves floor contact (ncon > 0)  
✓ **Settle stability**: 5 s passive drop produces finite trajectory, no NaN/Inf  
✓ **Actuators**: 6 position servos, kp=40.0, kv=10.0, forcerange=[-2.5, 2.5] N⋅m  
✓ **Sensors**: Full 24-sensor suite (joint pos/vel, IMU, touch)  

## Usage in Training

`biped_warp.xml` is loaded by the Phase 6 mjlab environment (`mjlab_biped/`) and trained via RSL-RL on Colab GPU. There is **no separate `biped_warp_no_jetson.xml` file** (correcting a stale claim in an earlier version of this doc) — per Phase 6.C's design, the Jetson on/off mass axis is an in-memory `DomainRandomizer` toggle applied to the single `biped_warp.xml`, not a second model file.

## Known Differences from CPU Variant

**None remaining as of 2026-07-13.** With both the integrator (fixed 2026-07-11) and collision geometry (fixed 2026-07-13) now unified, `biped_warp.xml` and `biped.xml` are physically identical — verified via bit-identical passive-drop trajectories (`test_warp_matches_cpu_settle`) and identical collision geom signatures (`test_warp_matches_cpu_collision`).

**Historical note (no longer applicable, kept for context):** before the 2026-07-13 fix, zero-action rollouts on the old primitive-collision `biped_warp.xml` showed a transient height "pop" (0.203m → ~0.376m over ~10 steps) not present on `biped.xml` — a contact-resolution artifact of the primitive proxies. This is resolved by the mesh-collision fix, not merely documented around.

## Limitations and Deferred Items

- **Accelerometer latency/noise:** Not modeled in the Warp variant (Phase 6.C); to be set to measured hardware values during Phase 8's sim-to-real calibration  
- **Terrain variation:** Flat plane only; domain randomization (Phase 7.C) does not include terrain roughness  
- **Real servo dead-zone/backlash:** Not modeled; deferred to Phase 8 once real servo system ID is available  

## References

- **MuJoCo Warp docs:** [https://github.com/google-deepmind/mujoco/tree/main/python/mujoco_warp](https://github.com/google-deepmind/mujoco/tree/main/python/mujoco_warp)  
- **CLAUDE.md Phase 6.0:** Full design rationale and validation methodology  
- **CLAUDE.md Phase 2–4 baselines:** Physics, sensor, and actuator validation data (CPU `biped.xml`)
