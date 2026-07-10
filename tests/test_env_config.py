"""
Tests for the mjlab_biped environment configuration (Phase 6.A).

Verifies:
- BipedEntityCfg compiles to correct MjModel structure
- Actuator names and count correct
- Joint structure (8 hinges: 6 actuated + 2 passive)
- Sensor suite present (24 sensors)
- Initial state configuration valid
- Physics constants frozen (timestep, integrator, control rate)
"""

import pytest
import mujoco
from pathlib import Path

# Import the entity configuration
from mjlab_biped import BipedEntityCfg, ActuatorConfig, InitialStateConfig


class TestEntityConfiguration:
    """Verify BipedEntityCfg configuration consistency."""

    def test_entity_instantiation(self):
        """Test that BipedEntityCfg can be instantiated."""
        cfg = BipedEntityCfg()
        assert cfg is not None
        assert cfg.model_path == "models/mjcf/biped_warp.xml"

    def test_entity_post_init_validation(self):
        """Test that BipedEntityCfg validates constraints in __post_init__."""
        # Valid configuration should not raise
        cfg = BipedEntityCfg()
        assert cfg.control_dt == 0.02
        assert cfg.timestep == 0.002
        assert cfg.control_decimation == 10

        # Invalid decimation should raise
        with pytest.raises(ValueError, match="control_dt.*must equal.*control_decimation"):
            BipedEntityCfg(control_dt=0.01)  # Would make decimation = 5, not 10

    def test_actuator_configuration(self):
        """Test actuator configuration."""
        cfg = BipedEntityCfg()
        assert cfg.num_actuators == 6
        assert len(cfg.actuators.target_names) == 6

        # Verify actuator target names
        expected_names = (
            "hip_roll_l",
            "knee_l",
            "ankle_l",
            "hip_roll_r",
            "knee_r",
            "ankle_r",
        )
        assert cfg.actuators.target_names == expected_names

        # Verify gains (Phase 4 frozen values)
        assert cfg.actuators.kp == 40.0
        assert cfg.actuators.kv == 10.0
        assert cfg.actuators.forcerange == (-2.5, 2.5)

    def test_initial_state_configuration(self):
        """Test initial state (stand keyframe) configuration."""
        cfg = BipedEntityCfg()

        # Base position and orientation
        assert cfg.init_state.base_pos == (0.0, 0.0, 0.2030)
        assert cfg.init_state.base_quat == (1.0, 0.0, 0.0, 0.0)

        # Joint angles: 8 hinges all at 0 rad
        assert len(cfg.init_state.joint_angles) == 8
        assert all(angle == 0.0 for angle in cfg.init_state.joint_angles)

        # Joint velocities: all at 0 rad/s
        assert len(cfg.init_state.joint_velocities) == 8
        assert all(vel == 0.0 for vel in cfg.init_state.joint_velocities)

    def test_qpos_qvel_dicts(self):
        """Test qpos/qvel dictionary generation."""
        cfg = BipedEntityCfg()

        qpos_dict = cfg.get_qpos_dict()
        qvel_dict = cfg.get_qvel_dict()

        # Should have 8 joints
        assert len(qpos_dict) == 8
        assert len(qvel_dict) == 8

        # Verify all joint names present
        joint_names = list(cfg.actuators.target_names) + ["foot_l", "foot_r"]
        assert set(qpos_dict.keys()) == set(joint_names)
        assert set(qvel_dict.keys()) == set(joint_names)

        # All should be zero
        assert all(angle == 0.0 for angle in qpos_dict.values())
        assert all(vel == 0.0 for vel in qvel_dict.values())

    def test_physics_configuration(self):
        """Test physics constants are properly frozen."""
        cfg = BipedEntityCfg()

        # Frozen from Phase 2 validation
        assert cfg.timestep == 0.002
        assert cfg.integrator == "implicit"

        # Frozen control rate (50 Hz)
        assert cfg.control_dt == 0.02
        assert cfg.control_decimation == 10

    def test_model_metadata(self):
        """Test model structure metadata."""
        cfg = BipedEntityCfg()

        # From Phase 0-4 model audit
        assert cfg.num_bodies == 36  # Including worldbody
        assert cfg.num_joints == 8  # 6 actuated + 2 passive (floating base is freejoint)
        assert cfg.num_actuators == 6
        assert cfg.num_sensors == 24
        assert cfg.total_mass == 0.784  # kg with Jetson


class TestModelSpecification:
    """Verify biped_warp.xml spec matches configuration."""

    @pytest.fixture
    def model_path(self):
        """Get path to biped_warp.xml relative to repo root."""
        # This test runs from repo root
        path = Path("models/mjcf/biped_warp.xml")
        if not path.exists():
            pytest.skip(f"Model file not found at {path}")
        return str(path)

    def test_model_file_exists(self, model_path):
        """Test that biped_warp.xml exists."""
        assert Path(model_path).exists(), f"Model file not found: {model_path}"

    def test_model_compilation(self, model_path):
        """Test that biped_warp.xml compiles without error."""
        try:
            spec = mujoco.MjSpec.from_file(model_path)
            assert spec is not None
            model = spec.compile()
            assert model is not None
        except Exception as e:
            pytest.fail(f"Failed to compile model: {e}")

    def test_model_structure(self, model_path):
        """Verify compiled model matches configuration."""
        spec = mujoco.MjSpec.from_file(model_path)
        model = spec.compile()
        cfg = BipedEntityCfg()

        # Body count (including worldbody)
        assert model.nbody == cfg.num_bodies, (
            f"Body count mismatch: model has {model.nbody}, "
            f"config expects {cfg.num_bodies}"
        )

        # Joint count (hinges only, not including freejoint)
        # Note: freejoint is counted separately as a pseudo-joint in MuJoCo
        assert model.njnt == cfg.num_joints + 1, (
            f"Joint count mismatch: model has {model.njnt} joints "
            f"({model.njnt - 1} hinges + 1 freejoint), "
            f"config expects {cfg.num_joints} hinges"
        )

        # Actuator count
        assert model.nu == cfg.num_actuators, (
            f"Actuator count mismatch: model has {model.nu}, "
            f"config expects {cfg.num_actuators}"
        )

        # Sensor count
        assert model.nsensor == cfg.num_sensors, (
            f"Sensor count mismatch: model has {model.nsensor}, "
            f"config expects {cfg.num_sensors}"
        )

    def test_actuator_names(self, model_path):
        """Verify all actuator names match configuration."""
        spec = mujoco.MjSpec.from_file(model_path)
        model = spec.compile()
        cfg = BipedEntityCfg()

        # Get actuator names from model
        actuator_names = []
        for i in range(model.nu):
            actuator_names.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{cfg.actuators.target_names[i]}"))

        # All actuator names should map correctly
        for i, target_name in enumerate(cfg.actuators.target_names):
            actuator_name = f"act_{target_name}"
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            assert actuator_id >= 0, (
                f"Actuator '{actuator_name}' not found in model. "
                f"Available actuators: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]}"
            )

    def test_joint_names(self, model_path):
        """Verify all actuated joint names present in model."""
        spec = mujoco.MjSpec.from_file(model_path)
        model = spec.compile()
        cfg = BipedEntityCfg()

        # Check all actuated joints exist
        for joint_name in cfg.actuators.target_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            assert joint_id >= 0, (
                f"Joint '{joint_name}' not found in model"
            )

        # Check ankle joints exist (passive)
        for ankle_name in ["foot_l", "foot_r"]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ankle_name)
            assert joint_id >= 0, (
                f"Ankle joint '{ankle_name}' not found in model"
            )

    def test_actuator_force_ranges(self, model_path):
        """Verify actuator force ranges are ±2.5 N⋅m (Phase 0 derated value)."""
        spec = mujoco.MjSpec.from_file(model_path)
        model = spec.compile()
        cfg = BipedEntityCfg()

        for i in range(model.nu):
            # Check forcerange in model
            lower = model.actuator_forcerange[i, 0]
            upper = model.actuator_forcerange[i, 1]
            assert lower == -2.5, (
                f"Actuator {i}: lower force range is {lower}, expected -2.5"
            )
            assert upper == 2.5, (
                f"Actuator {i}: upper force range is {upper}, expected 2.5"
            )

    def test_physics_options(self, model_path):
        """Verify physics options: timestep, integrator."""
        spec = mujoco.MjSpec.from_file(model_path)
        model = spec.compile()
        cfg = BipedEntityCfg()

        # Timestep
        assert abs(model.opt.timestep - cfg.timestep) < 1e-9, (
            f"Timestep mismatch: model has {model.opt.timestep}, "
            f"config expects {cfg.timestep}"
        )

        # Integrator: implicit = 1
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT, (
            f"Integrator mismatch: model has {model.opt.integrator}, "
            f"expected {mujoco.mjtIntegrator.mjINT_IMPLICIT} (implicit)"
        )

    def test_sensor_count(self, model_path):
        """Verify 24-sensor suite present."""
        spec = mujoco.MjSpec.from_file(model_path)
        model = spec.compile()
        cfg = BipedEntityCfg()

        assert model.nsensor == 24, (
            f"Sensor count: model has {model.nsensor}, expected 24"
        )

        # Verify sensor types (rough check)
        sensor_types = []
        for i in range(model.nsensor):
            sensor_type = model.sensor_type[i]
            sensor_types.append(sensor_type)

        # Should include jointpos, jointvel, framequat, gyro, accelerometer, touch
        assert mujoco.mjtSensor.mjSENS_JOINTPOS in sensor_types, "Missing jointpos sensors"
        assert mujoco.mjtSensor.mjSENS_JOINTVEL in sensor_types, "Missing jointvel sensors"
        assert mujoco.mjtSensor.mjSENS_FRAMEQUAT in sensor_types, "Missing framequat sensor"
        assert mujoco.mjtSensor.mjSENS_GYRO in sensor_types, "Missing gyro sensor"
        assert mujoco.mjtSensor.mjSENS_ACCELEROMETER in sensor_types, "Missing accelerometer sensor"
        assert mujoco.mjtSensor.mjSENS_TOUCH in sensor_types, "Missing touch sensors"


class TestIntegration:
    """Integration tests combining config and model."""

    @pytest.fixture
    def biped_cfg_and_model(self):
        """Load both configuration and model."""
        cfg = BipedEntityCfg()
        model_path = Path("models/mjcf/biped_warp.xml")
        if not model_path.exists():
            pytest.skip(f"Model file not found at {model_path}")

        spec = mujoco.MjSpec.from_file(str(model_path))
        model = spec.compile()
        return cfg, model

    def test_entity_config_valid(self, biped_cfg_and_model):
        """Test that entity config and model are consistent."""
        cfg, model = biped_cfg_and_model

        # Model structure should match config
        assert model.nbody == cfg.num_bodies
        assert model.njnt == cfg.num_joints + 1  # +1 for freejoint
        assert model.nu == cfg.num_actuators
        assert model.nsensor == cfg.num_sensors

    def test_stand_keyframe_exists(self, biped_cfg_and_model):
        """Test that stand keyframe is present in model."""
        cfg, model = biped_cfg_and_model

        # Find keyframe by name
        keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        assert keyframe_id >= 0, "Stand keyframe not found in model"

    def test_stand_keyframe_qpos(self, biped_cfg_and_model):
        """Verify stand keyframe qpos matches configuration."""
        cfg, model = biped_cfg_and_model

        keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        assert keyframe_id >= 0

        # Get keyframe qpos (15 values: 7 base + 8 hinges)
        qpos = model.key_qpos[keyframe_id]

        # Base position should be [0, 0, 0.203]
        assert abs(qpos[0] - 0.0) < 1e-6, f"Base x: {qpos[0]}"
        assert abs(qpos[1] - 0.0) < 1e-6, f"Base y: {qpos[1]}"
        assert abs(qpos[2] - 0.2030) < 1e-3, f"Base z: {qpos[2]}, expected ~0.2030"

        # Base quaternion should be [1, 0, 0, 0]
        assert abs(qpos[3] - 1.0) < 1e-6, f"Quat w: {qpos[3]}"
        assert abs(qpos[4] - 0.0) < 1e-6, f"Quat x: {qpos[4]}"
        assert abs(qpos[5] - 0.0) < 1e-6, f"Quat y: {qpos[5]}"
        assert abs(qpos[6] - 0.0) < 1e-6, f"Quat z: {qpos[6]}"

        # Joint angles should all be 0
        for i in range(8):
            assert abs(qpos[7 + i] - 0.0) < 1e-6, f"Joint {i}: {qpos[7 + i]}"


class TestEntityOutput:
    """Test entity configuration output."""

    def test_entity_summary(self):
        """Print entity configuration summary for debugging."""
        cfg = BipedEntityCfg()

        print("\n" + "=" * 80)
        print("BIPED ENTITY CONFIGURATION (Phase 6.A)")
        print("=" * 80)
        print(f"Model path:       {cfg.model_path}")
        print(f"Timestep:         {cfg.timestep} s ({cfg.timestep**-1:.0f} Hz)")
        print(f"Integrator:       {cfg.integrator}")
        print(f"Control rate:     {cfg.control_dt} s ({cfg.control_dt**-1:.0f} Hz)")
        print(f"Control decimation: {cfg.control_decimation} (physics steps per control step)")
        print(f"\nStructure:")
        print(f"  Bodies:         {cfg.num_bodies}")
        print(f"  Joints:         {cfg.num_joints} hinges (+ 1 floating base freejoint)")
        print(f"  Actuators:      {cfg.num_actuators}")
        print(f"  Sensors:        {cfg.num_sensors}")
        print(f"  Total mass:     {cfg.total_mass} kg")
        print(f"\nActuators:")
        print(f"  Targets:        {cfg.actuators.target_names}")
        print(f"  Gains:          kp={cfg.actuators.kp}, kv={cfg.actuators.kv}")
        print(f"  Force range:    {cfg.actuators.forcerange} N⋅m")
        print(f"\nInitial state (stand keyframe):")
        print(f"  Base pos:       {cfg.init_state.base_pos}")
        print(f"  Base quat:      {cfg.init_state.base_quat}")
        print(f"  Joint angles:   {cfg.init_state.joint_angles}")
        print(f"  Joint vels:     {cfg.init_state.joint_velocities}")
        print("=" * 80 + "\n")
