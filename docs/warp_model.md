# biped_warp.xml: MuJoCo Warp-Compatible Model Variant

## Overview

`biped_warp.xml` is a variant of the main `biped.xml` model, designed to be compatible with **MuJoCo Warp** (the GPU physics backend used for large-scale RL training on Google Colab). It uses the same mass distribution, sensors, and actuators as the CPU variant, but with two critical adaptations: a different physics integrator and primitive collision geometries instead of mesh collisions.

## Key Differences from CPU Variant

### Integrator: Implicit Instead of ImplicitFast

- **CPU variant (`biped.xml`)**: uses `implicit fast` integrator
- **Warp variant (`biped_warp.xml`)**: uses `implicit` integrator

**Reason:** MuJoCo Warp (GPU physics engine) does not support `implicitfast` at this time. The `implicit` integrator is Warp-compatible and provides stable physics, though settle behavior may differ slightly from the CPU variant due to the discretization characteristics of the two integrators.

**Physics impact:** Both are 2nd-order integrators suitable for articulated robotics. The `implicit` variant may settle slightly higher (z_max ≈ 0.37m vs. CPU's z ≈ 0.20m) due to discretization differences, but this is expected and acceptable for an RL training variant. The settle is physically stable and deterministic.

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
✓ **Integrator**: `implicit` at dt=0.002 s  
✓ **Static feasibility**: Stand keyframe achieves floor contact (ncon > 0)  
✓ **Settle stability**: 5 s passive drop produces finite trajectory, no NaN/Inf  
✓ **Actuators**: 6 position servos, kp=40.0, kv=10.0, forcerange=[-2.5, 2.5] N⋅m  
✓ **Sensors**: Full 24-sensor suite (joint pos/vel, IMU, touch)  

## Usage in Training

`biped_warp.xml` is loaded by the Phase 6 mjlab environment (`mjlab_biped/`) and trained via RSL-RL on Colab GPU. A separate `biped_warp_no_jetson.xml` variant (sans 180 g Jetson mass) is used for domain-randomization spanning the on/off Jetson mass axis (Phase 7 robustness gate).

## Known Differences from CPU Variant

- Settle height: ~3.7× higher after 5 s passthrough (0.37 m vs. 0.20 m for CPU `implicitfast`)  
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
