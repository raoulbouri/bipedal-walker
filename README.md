# Biped Robot Simulation

Local development environment for a biped robot chassis: Onshape CAD design → URDF → MuJoCo (MJCF) → physics simulation and controller development.

## Quick Start

### Setup

```bash
cd biped
uv sync
```

### Visualize the Model

```bash
uv run mjpython scripts/visualize.py
```

- **Space** to toggle dynamics
- **Mouse** to control camera (rotate: left-click + drag, pan: right-click + drag, zoom: scroll)

## Project Structure

```
biped/
├── scripts/           # Python utilities
│   ├── build_mjcf.py      # Stage 1: URDF → MuJoCo compiler → raw MJCF
│   ├── postprocess.py     # Stage 2: raw MJCF → RL-ready models
│   └── visualize.py       # Interactive MuJoCo viewer
├── models/            # Robot definitions
│   ├── mjcf/              # MuJoCo XML files
│   │   ├── biped.xml      # Final model with Jetson Nano (784 g)
│   │   ├── biped_no_jetson.xml   # Final model without Jetson (604 g)
│   │   └── biped_raw.xml  # Intermediate (Stage 1 output)
│   └── urdf/              # URDF files
│       ├── assembly_simple.urdf         # Onshape export (source)
│       └── biped_for_mujoco.urdf        # Stage 1 intermediate
├── meshes/
│   └── stl/               # STL mesh files (1.9 MB)
├── launch/
│   └── assembly_simple.launch   # RViz launch file (for URDF inspection)
├── docs/
│   └── reference_renders/ # Assembly poses from Onshape
├── pyproject.toml     # Dependency management (mujoco, numpy)
└── uv.lock            # Locked dependency versions
```

## Pipeline

```
Onshape assembly
    │  (URDF native export)
    ▼
models/urdf/assembly_simple.urdf  (35 links, 34 joints)
    │  build_mjcf.py
    ▼
models/urdf/biped_for_mujoco.urdf (compiler directives, joint fixes)
    │  mujoco.MjModel.from_xml_path() → mj_saveLastXML()
    ▼
models/mjcf/biped_raw.xml  (MuJoCo-compiled, no sim furniture)
    │  postprocess.py
    ▼
models/mjcf/biped.xml / biped_no_jetson.xml  (RL-ready: actuators, sensors, floating base)
```

## Regenerating from Onshape

After re-exporting the URDF from Onshape:

```bash
# Update models/urdf/assembly_simple.urdf, then:
uv run python scripts/build_mjcf.py     # Stage 1
uv run python scripts/postprocess.py    # Stage 2
```

## Model Specifications

### Actuated Joints (6)
- `hip_roll_l/r` (2)
- `hip_pitch_l/r` (2)
- `knee_l/r` (2)

### Passive Joints (2)
- `ankle_l/r` — passive, hardstop-limited based on physical measurements

### Sensors (24 total)
- Joint position/velocity for all 8 hinge joints
- IMU (torso): orientation (quat), angular velocity, linear acceleration, position, linear velocity
- Foot touch sensors (left/right)

### Actuator Control
- Position servos with `kp=5.0`, `kv=0.2`, `forcerange=±3 N·m`
- Matches ST3215 servo stall torque (~3 N·m)

### Mass (measured, not CAD-derived)
- **biped.xml**: 784 g (includes Jetson Orin Nano 180 g)
- **biped_no_jetson.xml**: 604 g (baseline)

## Known Limitations

- **No balance controller** — the robot will topple when physics runs. Next task: implement a gait/balance policy.
- **Full-resolution mesh collision** — safe for this ~35-body model, but not optimized. Can switch to primitive/convex-decomposed geometry if sim performance becomes a bottleneck during RL training.

## Next Steps

1. Implement a balance/gait controller (RL via MuJoCo or classical MPC/ZMP)
2. Sim-to-real transfer: validate in `biped_no_jetson.xml`, cross-check with `biped.xml`, deploy via Jetson
3. (Optional) Optimize collision geometry for faster contact-rich RL training

## Documentation

- **CLAUDE.md** — High-level pipeline overview and design decisions
- **MEMORY.md** — Detailed analysis log including debugging notes and verified findings

## Hardware Reference

- **Servo motor**: ST3215 / SCS215
  - Stall torque: ~3 N·m @ 12V
  - Speed: 5–6 rad/s no-load
  - Position range: 0–360° (0–4095 steps)
- **Jetson**: Orin Nano, mounted on torso
  - Mass: 180 g
  - Running on Ubuntu + Isaac Sim / Isaac ROS

## License

This project documentation is maintained locally for development and simulation. Hardware control code resides on the Jetson (see `~/Work/biped` on the Jetson).
