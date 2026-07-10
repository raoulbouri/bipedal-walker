"""
Tests for sensor configuration in biped models.

Verifies:
1. Sensor count (24) and identity
2. IMU site configuration
3. Stand keyframe configuration
"""

import pytest
import mujoco


# Table of expected sensors: (name, type_enum, dimension)
EXPECTED_SENSORS = [
    ("pos_hip_roll_l", mujoco.mjtSensor.mjSENS_JOINTPOS, 1),
    ("vel_hip_roll_l", mujoco.mjtSensor.mjSENS_JOINTVEL, 1),
    ("pos_knee_l", mujoco.mjtSensor.mjSENS_JOINTPOS, 1),
    ("vel_knee_l", mujoco.mjtSensor.mjSENS_JOINTVEL, 1),
    ("pos_ankle_l", mujoco.mjtSensor.mjSENS_JOINTPOS, 1),
    ("vel_ankle_l", mujoco.mjtSensor.mjSENS_JOINTVEL, 1),
    ("pos_hip_roll_r", mujoco.mjtSensor.mjSENS_JOINTPOS, 1),
    ("vel_hip_roll_r", mujoco.mjtSensor.mjSENS_JOINTVEL, 1),
    ("pos_knee_r", mujoco.mjtSensor.mjSENS_JOINTPOS, 1),
    ("vel_knee_r", mujoco.mjtSensor.mjSENS_JOINTVEL, 1),
    ("pos_ankle_r", mujoco.mjtSensor.mjSENS_JOINTPOS, 1),
    ("vel_ankle_r", mujoco.mjtSensor.mjSENS_JOINTVEL, 1),
    ("pos_foot_l", mujoco.mjtSensor.mjSENS_JOINTPOS, 1),
    ("vel_foot_l", mujoco.mjtSensor.mjSENS_JOINTVEL, 1),
    ("pos_foot_r", mujoco.mjtSensor.mjSENS_JOINTPOS, 1),
    ("vel_foot_r", mujoco.mjtSensor.mjSENS_JOINTVEL, 1),
    ("torso_quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT, 4),
    ("torso_gyro", mujoco.mjtSensor.mjSENS_GYRO, 3),
    ("torso_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER, 3),
    ("torso_pos", mujoco.mjtSensor.mjSENS_FRAMEPOS, 3),
    ("torso_linvel", mujoco.mjtSensor.mjSENS_FRAMELINVEL, 3),
    ("torso_angvel", mujoco.mjtSensor.mjSENS_FRAMEANGVEL, 3),
    ("touch_l", mujoco.mjtSensor.mjSENS_TOUCH, 1),
    ("touch_r", mujoco.mjtSensor.mjSENS_TOUCH, 1),
]


@pytest.fixture(params=["model_path_jetson", "model_path_no_jetson"])
def model_path(request):
    """Parametrize fixture to test both model variants."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def model(model_path):
    """Load MuJoCo model from path."""
    return mujoco.MjModel.from_xml_path(model_path)


class TestSensorCount:
    """Test sensor count."""

    def test_sensor_count_is_24(self, model):
        """Assert total sensor count is 24."""
        assert model.nsensor == 24, f"Expected 24 sensors, got {model.nsensor}"


class TestSensorIdentity:
    """Test sensor name, type, and dimension match expected table."""

    def test_all_sensors_exist_and_match_table(self, model):
        """
        For each expected sensor in the table:
        - Verify it exists (name lookup returns valid id)
        - Verify its type matches
        - Verify its dimension matches
        """
        for sensor_name, expected_type, expected_dim in EXPECTED_SENSORS:
            # Look up sensor by name
            sensor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
            )
            assert (
                sensor_id >= 0
            ), f"Sensor '{sensor_name}' not found (id={sensor_id})"

            # Verify type
            actual_type = model.sensor_type[sensor_id]
            expected_type_value = expected_type.value
            assert (
                actual_type == expected_type_value
            ), (
                f"Sensor '{sensor_name}': type mismatch. "
                f"Expected {expected_type.name}={expected_type_value}, "
                f"got {actual_type}"
            )

            # Verify dimension
            actual_dim = model.sensor_dim[sensor_id]
            assert (
                actual_dim == expected_dim
            ), (
                f"Sensor '{sensor_name}': dim mismatch. "
                f"Expected {expected_dim}, got {actual_dim}"
            )


class TestIMUSite:
    """Test IMU site configuration."""

    def test_imu_site_exists(self, model):
        """Assert IMU site exists."""
        imu_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        assert imu_site_id >= 0, "IMU site not found"

    def test_imu_site_parent_body(self, model):
        """Assert IMU site's parent body is composite_part_1__1_."""
        imu_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        assert imu_site_id >= 0, "IMU site not found"

        # Get parent body of IMU site
        parent_body_id = model.site_bodyid[imu_site_id]
        parent_body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, parent_body_id
        )

        # Expected parent body
        expected_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "composite_part_1__1_"
        )
        assert (
            expected_body_id >= 0
        ), "Expected body 'composite_part_1__1_' not found"
        assert (
            parent_body_id == expected_body_id
        ), (
            f"IMU site parent body mismatch. "
            f"Expected '{expected_body_id}' (composite_part_1__1_), "
            f"got '{parent_body_id}' ({parent_body_name})"
        )


class TestStandKeyframe:
    """Test stand keyframe configuration."""

    def test_stand_keyframe_exists(self, model):
        """Assert stand keyframe exists."""
        # Find stand keyframe by iterating through keyframes
        stand_key_id = -1
        for i in range(model.nkey):
            keyframe_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_KEY, i
            )
            if keyframe_name == "stand":
                stand_key_id = i
                break

        assert stand_key_id >= 0, "Stand keyframe not found"

    def test_stand_keyframe_qpos_z_height(self, model):
        """Assert stand keyframe qpos[2] (z-height) is close to 0.2030.

        Was 0.2061 originally, but that left both feet floating 1-3mm above
        the floor (0 contacts, touch sensors read 0) — recalibrated during
        Phase 0's integration testing via mesh AABB corners so both feet
        have a small interpenetration margin at rest. See postprocess.py's
        STAND_HEIGHT comment for the full derivation.
        """
        # Find stand keyframe
        stand_key_id = -1
        for i in range(model.nkey):
            keyframe_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_KEY, i
            )
            if keyframe_name == "stand":
                stand_key_id = i
                break

        assert stand_key_id >= 0, "Stand keyframe not found"

        # Get qpos and check z-height (qpos[2])
        qpos = model.key_qpos[stand_key_id]
        z_height = qpos[2]

        # Expected z-height with tolerance
        expected_z_height = 0.2030
        tolerance = 0.0005

        assert (
            abs(z_height - expected_z_height) <= tolerance
        ), (
            f"Stand keyframe z-height mismatch. "
            f"Expected {expected_z_height} ± {tolerance}, "
            f"got {z_height} (diff={abs(z_height - expected_z_height)})"
        )
