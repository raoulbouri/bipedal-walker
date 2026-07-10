"""
Robot entity configuration for the biped model.

This module defines the BipedEntityCfg dataclass that specifies:
- Model source (biped_warp.xml from Phase 6.0)
- Actuator targets (6 position servos: hip_roll, knee, ankle for each leg)
- Initial state (stand keyframe: all joint angles 0, base at STAND_HEIGHT)

When mjlab is available (Phase 7+), this cfg will be adapted to mjlab's
EntityCfg, EntityArticulationInfoCfg, and XmlActuatorCfg. For now, it
serves as the pure configuration source.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class ActuatorConfig:
    """Position actuator configuration."""
    target_names: Tuple[str, ...] = (
        "hip_roll_l",
        "knee_l",
        "ankle_l",
        "hip_roll_r",
        "knee_r",
        "ankle_r",
    )
    kp: float = 40.0
    kv: float = 10.0
    forcerange: Tuple[float, float] = (-2.5, 2.5)


@dataclass
class InitialStateConfig:
    """Initial state at the stand keyframe."""
    # STAND_HEIGHT = 0.2030 m (recalibrated Phase 0)
    # Floating base: position [0, 0, 0.2030], quaternion [1, 0, 0, 0] (identity)
    base_pos: Tuple[float, float, float] = (0.0, 0.0, 0.2030)
    base_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # Joint angles: all 8 hinges at 0 radians (6 actuated + 2 passive ankles)
    # Order: hip_roll_l, knee_l, ankle_l, hip_roll_r, knee_r, ankle_r, foot_l, foot_r
    joint_angles: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Velocities: all zero at rest
    joint_velocities: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class BipedEntityCfg:
    """
    Biped robot entity configuration for mjlab.

    Configures:
    - Model source: biped_warp.xml (MuJoCo Warp-compatible variant, Phase 6.0)
    - Actuators: 6 position servos with kp=40.0, kv=10.0, ±2.5 N⋅m
    - Initial state: stand keyframe (all joints 0, base at z=0.2030 m)

    Properties (frozen, per Phase 4 characterization):
    - Integrator: implicit (Warp-compatible)
    - Timestep: 0.002 s (500 Hz physics)
    - Control rate: 50 Hz (10 physics substeps per control step)
    - Total mass: 0.784 kg (with Jetson Nano)
    - Sensors: 24-sensor suite (joint pos/vel, IMU, foot touch)

    Note: This configuration is "pure data"—it does not import mjlab,
    torch, or RL-specific modules. It will be wrapped by mjlab's
    EntityCfg/EntityArticulationInfoCfg in Phase 7+ when mjlab is available.
    """

    # Model file path (relative to repo root)
    model_path: str = "models/mjcf/biped_warp.xml"

    # Actuator configuration
    actuators: ActuatorConfig = field(default_factory=ActuatorConfig)

    # Initial state configuration
    init_state: InitialStateConfig = field(default_factory=InitialStateConfig)

    # Metadata: frozen specifications from Phase 0-4
    num_bodies: int = 36  # Including worldbody
    num_joints: int = 8   # 6 actuated + 2 passive ankles (floating base is a freejoint)
    num_actuators: int = 6
    num_sensors: int = 24
    total_mass: float = 0.784  # kg (with Jetson Nano)

    # Physics configuration (frozen, Phase 2 validated)
    timestep: float = 0.002  # s (500 Hz)
    integrator: str = "implicit"  # Warp-compatible
    control_dt: float = 0.02  # s (50 Hz control rate)
    control_decimation: int = 10  # physics steps per control step

    def __post_init__(self):
        """Validate configuration consistency."""
        # Verify control decimation
        if abs(self.control_dt / self.timestep - self.control_decimation) > 1e-6:
            raise ValueError(
                f"control_dt ({self.control_dt}) / timestep ({self.timestep}) "
                f"must equal control_decimation ({self.control_decimation})"
            )

        # Verify actuator count
        if len(self.actuators.target_names) != self.num_actuators:
            raise ValueError(
                f"Actuator target count ({len(self.actuators.target_names)}) "
                f"must equal num_actuators ({self.num_actuators})"
            )

        # Verify initial state joint count
        if len(self.init_state.joint_angles) != self.num_joints:
            raise ValueError(
                f"Initial joint angle count ({len(self.init_state.joint_angles)}) "
                f"must equal num_joints ({self.num_joints})"
            )

    def get_qpos_dict(self) -> Dict[str, float]:
        """
        Build a dictionary of joint names → initial angles.

        Returns a dict mapping each joint name to its initial angle from the
        stand keyframe. Useful for mjlab's InitialStateCfg.
        """
        joint_names = list(self.actuators.target_names) + ["foot_l", "foot_r"]
        if len(joint_names) != len(self.init_state.joint_angles):
            raise ValueError(
                f"Joint names count ({len(joint_names)}) must match "
                f"joint angles count ({len(self.init_state.joint_angles)})"
            )
        return {name: angle for name, angle in zip(joint_names, self.init_state.joint_angles)}

    def get_qvel_dict(self) -> Dict[str, float]:
        """
        Build a dictionary of joint names → initial velocities.

        Returns a dict mapping each joint name to its initial velocity
        (all zeros at the stand keyframe).
        """
        joint_names = list(self.actuators.target_names) + ["foot_l", "foot_r"]
        if len(joint_names) != len(self.init_state.joint_velocities):
            raise ValueError(
                f"Joint names count ({len(joint_names)}) must match "
                f"joint velocities count ({len(self.init_state.joint_velocities)})"
            )
        return {name: vel for name, vel in zip(joint_names, self.init_state.joint_velocities)}
