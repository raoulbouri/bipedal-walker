# mjlab_biped: Biped RL Training Task

MuJoCo Warp + mjlab task package for training the biped balance policy on Google Colab.

## Phase Status

- **Phase 6.A (CURRENT):** Entity configuration (`entity.py`) ✓
- **Phase 6.B (NEXT):** Observation manager (actor + critic groups)
- **Phase 6.C:** Command + domain randomization
- **Phase 6.D:** Reward + termination (stubs)
- **Phase 6.E:** Full env config + task registration

## Package Structure

```
mjlab_biped/
├── __init__.py              # Package exports
├── entity.py                # Robot EntityCfg (Phase 6.A)
├── config.py                # Full ManagerBasedRlEnvCfg (Phase 6.E)
├── observations.py          # Observation groups (Phase 6.B, TBD)
├── commands.py              # Command + DR events (Phase 6.C, TBD)
├── rewards.py               # Reward/termination funcs (Phase 6.D, TBD)
└── README.md                # This file
```

## Model

- **biped_warp.xml:** MuJoCo Warp-compatible MJCF variant (Phase 6.0)
  - Integrator: `implicit` (Warp-compatible, not the CPU `implicitfast`)
  - Collision: primitive proxies for non-foot bodies, mesh for feet
  - Physics: timestep 0.002 s, control rate 50 Hz (10 physics steps/control)
  - Total mass: 0.784 kg (with Jetson Nano), 0.604 kg (without)

## Actuators

6 position servos (XmlActuatorCfg reusing XML definitions):
- `hip_roll_l`, `knee_l`, `ankle_l`
- `hip_roll_r`, `knee_r`, `ankle_r`

Per-joint gains (frozen Phase 4 characterization):
- kp = 40.0, kv = 10.0
- forcerange = ±2.5 N⋅m (derated from 3.0 N⋅m spec for voltage margin)

## Sensors

24-sensor suite (onboard-available signals only, no privileged access in actor):
- Joint pos/vel: 6 actuated + 2 passive ankles (16 sensors)
- IMU on torso: framequat, gyro, accelerometer, framepos, framelinvel, frameangvel (6 sensors)
- Foot touch: left + right (2 sensors)

## Initial State

Stand keyframe configuration (Phase 0 recalibrated):
- Base position: [0, 0, 0.2030] m (z ensures both feet contact floor)
- Base quaternion: [1, 0, 0, 0] (identity, upright)
- All joint angles: 0 rad (default stand pose)
- All velocities: 0 rad/s (at rest)

## Usage (Phase 7+)

Once full mjlab integration is complete:

```python
from mjlab_biped import BipedEntityCfg, BipedEnvCfg

# Load configuration
cfg = BipedEnvCfg(...)

# Create environment (requires mjlab + mujoco_warp)
env = cfg(...)

# Train with RSL-RL on Colab
uv run train Mjlab-Biped-Balance-v0 --env.scene.num-envs 4096
```

## Roadmap

- **Phase 6.0 (COMPLETE):** Warp-compatible model + static feasibility check
- **Phase 6.A (CURRENT):** Entity configuration
- **Phase 6.B (NEXT):** Observation manager
- **Phase 6.C:** Command + domain randomization
- **Phase 6.D:** Reward + termination
- **Phase 6.E:** Full env config + registration
- **Phase 7:** Training + evaluation on Colab (mjlab + RSL-RL + GPU)

## Local Development (Phases 0-5)

For CPU testing on macOS, use the original simulation infrastructure:

```python
from sim import BipedSim

sim = BipedSim("models/mjcf/biped.xml")
sim.reset(keyframe="stand")
```

The `biped.xml` and validation suite remain the source of truth for
CPU-based physics; `biped_warp.xml` is training-only on Colab GPU.

## References

- **CLAUDE.md Phase 6:** Full roadmap with sub-task breakdown
- **MEMORY.md 2026-07-10:** mjlab API verified in mujoco_warp environment
- **docs/observation_spec.md:** Frozen observation structure (Phase 6.B)
- **docs/actuator_characterization.md:** Servo model (Phase 4)
- **docs/physics_baselines.md:** Validated physics (Phases 0-2)
