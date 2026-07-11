# biped_warp.xml: MuJoCo Warp-Compatible Model Variant

## Overview

`biped_warp.xml` is a variant of the main `biped.xml` model, designed to be compatible with **MuJoCo Warp** (the GPU physics backend used for large-scale RL training on Google Colab). It uses the same mass distribution, sensors, and actuators as the CPU variant, but with two critical adaptations: a different physics integrator and primitive collision geometries instead of mesh collisions.

## Key Differences from CPU Variant

### Integrator: ImplicitFast (corrected 2026-07-11 — previously wrongly set to "implicit")

- **CPU variant (`biped.xml`)**: uses `implicitfast` integrator
- **Warp variant (`biped_warp.xml`)**: also uses `implicitfast` integrator (matches CPU exactly)

**Correction, 2026-07-11:** this doc previously stated the Warp variant used `"implicit"` because "MuJoCo Warp does not support `implicitfast`." That premise was never verified against a real mjlab install and turned out to be wrong: a live local CPU install of `mjlab`/`mujoco-warp` shows mjlab's own integrator map (`mjlab/sim/sim.py` `_INTEGRATOR_MAP`) only recognizes `"euler"` and `"implicitfast"` — `"implicit"` isn't a valid option at all and raises `KeyError('implicit')` the moment an env is constructed. This would have failed on Colab too, just later than the smoke test would have caught it. Fixed by switching `biped_warp.xml` to `"implicitfast"`, which is strictly better than the previous plan anyway: it now matches `biped.xml`'s CPU-validated integrator exactly, eliminating one whole axis of sim-to-sim drift between the Phase 0-4 validated model and the model actually trained against.

### Collision Geometry: Primitives vs. Full-Mesh

- **CPU variant**: All body geometry is full-resolution mesh collision (from `biped_raw.xml`, set via Stage 2's whole-body collision enablement)
- **Warp variant**: Non-foot bodies use primitive collision geoms; foot bodies retain mesh collision

**Collision proxy specifications:**
| Mesh / Body | Primitive Type | Size Params |
|---|---|---|
| Motor (ST3215 servo) | Capsule | radius=0.015, length=0.05 |
| Upper_Leg_A/B | Capsule | radius=0.012, length=0.08 |
| Tibia (shank) | Capsule | radius=0.008, length=0.10 |
| Tube (strut) | Capsule | radius=0.008, length=0.10 |
| Hip_Base | Box | size=0.02 × 0.05 × 0.02 |
| Hip_Joint_A/B | Sphere | radius=0.01 |
| Part_1 (torso strut) | Capsule | radius=0.008, length=0.06 |
| Foot, Foot_1 | Mesh | Foot.stl (unchanged) |

**Reason:** MuJoCo Warp's mesh collision support is memory-intensive and computationally expensive when scaling to thousands of parallel environments. Primitive collisions are much faster and more memory-efficient. Full-resolution mesh collision is retained *only* for the feet (`foot_col_l`, `foot_col_r`), which are critical for ground contact detection and touch sensor accuracy. All other bodies use approximating primitives, which are sufficient for collision detection and do not significantly impact the fidelity of body-to-body contact (internal chain contacts are largely excluded by MuJoCo's parent-child filtering anyway).

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

- **Settle behavior under zero action (measured 2026-07-11, real mjlab env, post-integrator-fix):** starting from the `STAND_HEIGHT=0.2030` keyframe with the position servos holding a zero target, base height rises to a brief transient peak around ~0.376 m over the first ~10 control steps (a contact-resolution "pop" from the primitive collision proxies settling out initial interpenetration against the mesh-accurate CPU baseline), then decays and holds around ~0.247 m — stable, no NaN, no further drift observed over 60 steps. This is a real, still-open physics deviation from the CPU model (which settles near its own keyframe height) attributable to the primitive-vs-mesh collision proxy swap below, not the integrator (now identical to CPU). Revisit if it interferes with early training (e.g. as a confound in the height-termination reward).
- Number of contacts: Mesh collision produces many small contacts (26+ at stand), while primitives produce fewer, larger contacts  
- Foot contact fidelity: Identical (both use `foot_col_*` mesh geoms)  
- Touch sensor accuracy: Identical (tied to foot mesh collision, not affected by torso/limb primitive swaps)  

## Limitations and Deferred Items

- **Accelerometer latency/noise:** Not modeled in the Warp variant (Phase 6.C); to be set to measured hardware values during Phase 8's sim-to-real calibration  
- **Terrain variation:** Flat plane only; domain randomization (Phase 7.C) does not include terrain roughness  
- **Real servo dead-zone/backlash:** Not modeled; deferred to Phase 8 once real servo system ID is available  

## References

- **MuJoCo Warp docs:** [https://github.com/google-deepmind/mujoco/tree/main/python/mujoco_warp](https://github.com/google-deepmind/mujoco/tree/main/python/mujoco_warp)  
- **CLAUDE.md Phase 6.0:** Full design rationale and validation methodology  
- **CLAUDE.md Phase 2–4 baselines:** Physics, sensor, and actuator validation data (CPU `biped.xml`)
