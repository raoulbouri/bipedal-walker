"""
Sensor validation tests for the biped robot.

This module verifies that joint position and velocity sensors correctly
reflect the underlying qpos and qvel states.
"""

import csv
import numpy as np
import mujoco
import pytest


class TestJointSensors:
    """Test suite for joint position and velocity sensor validation."""

    def test_joint_pos_vel_sensors_match_state(self, model_path_jetson):
        """
        Verify that joint position and velocity sensors match the underlying state.

        This test:
        - Loads the biped model and creates MjData
        - Identifies all 8 joint names and their sensor addresses
        - Generates 20+ randomized test states with seed=42 for reproducibility
        - For each state, sets qpos and qvel to random values within joint ranges
        - Calls mj_forward to compute sensor values
        - Asserts that sensor readings match the state exactly (atol=1e-12)

        The 8 joints are:
        - hip_roll_l, hip_pitch_l, knee_l (left leg)
        - hip_roll_r, hip_pitch_r, knee_r (right leg)
        - ankle_l, ankle_r (ankles)

        Each joint has one pos_<name> and one vel_<name> sensor.
        """
        # Load model and create data
        model = mujoco.MjModel.from_xml_path(model_path_jetson)
        data = mujoco.MjData(model)

        # Define the 8 joint names
        joint_names = [
            "hip_roll_l",
            "hip_pitch_l",
            "knee_l",
            "hip_roll_r",
            "hip_pitch_r",
            "knee_r",
            "ankle_l",
            "ankle_r",
        ]

        # Build a dictionary with joint info: joint_id, qposadr, dofadr, range, pos_sensor_adr, vel_sensor_adr
        joint_info = {}
        for joint_name in joint_names:
            # Get joint ID
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            assert joint_id >= 0, f"Joint '{joint_name}' not found in model"

            # Get qpos and dof addresses
            qposadr = model.jnt_qposadr[joint_id]
            dofadr = model.jnt_dofadr[joint_id]

            # Get joint range
            jnt_range = model.jnt_range[joint_id]

            # Get sensor IDs and addresses
            pos_sensor_name = f"pos_{joint_name}"
            vel_sensor_name = f"vel_{joint_name}"

            pos_sensor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, pos_sensor_name
            )
            vel_sensor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, vel_sensor_name
            )

            assert pos_sensor_id >= 0, f"Sensor '{pos_sensor_name}' not found"
            assert vel_sensor_id >= 0, f"Sensor '{vel_sensor_name}' not found"

            # Get sensor addresses
            pos_sensor_adr = model.sensor_adr[pos_sensor_id]
            vel_sensor_adr = model.sensor_adr[vel_sensor_id]

            joint_info[joint_name] = {
                "joint_id": joint_id,
                "qposadr": qposadr,
                "dofadr": dofadr,
                "range": jnt_range,
                "pos_sensor_adr": pos_sensor_adr,
                "vel_sensor_adr": vel_sensor_adr,
            }

        # Log joint info for debugging
        print(f"\n--- Joint Information ---")
        for name, info in joint_info.items():
            print(
                f"{name:15s}: qposadr={info['qposadr']:2d}, dofadr={info['dofadr']:2d}, "
                f"range=[{info['range'][0]:7.4f}, {info['range'][1]:7.4f}], "
                f"pos_adr={info['pos_sensor_adr']:2d}, vel_adr={info['vel_sensor_adr']:2d}"
            )

        # Seed for reproducibility
        np.random.seed(42)

        # Number of test states
        num_test_states = 25

        print(f"\n--- Running {num_test_states} randomized test states ---\n")

        # Loop over randomized test states
        for state_idx in range(num_test_states):
            # Generate random qpos values within each joint's range
            for joint_name, info in joint_info.items():
                low, high = info["range"]
                random_qpos = np.random.uniform(low, high)
                data.qpos[info["qposadr"]] = random_qpos

            # Generate random qvel values (not range-limited, but realistic ~[-2, 2] rad/s)
            for joint_name, info in joint_info.items():
                random_qvel = np.random.uniform(-2.0, 2.0)
                data.qvel[info["dofadr"]] = random_qvel

            # Call mj_forward to compute sensor values
            mujoco.mj_forward(model, data)

            # Verify sensor readings match state exactly
            for joint_name, info in joint_info.items():
                qpos_val = data.qpos[info["qposadr"]]
                qvel_val = data.qvel[info["dofadr"]]
                sensor_pos_val = data.sensordata[info["pos_sensor_adr"]]
                sensor_vel_val = data.sensordata[info["vel_sensor_adr"]]

                # Assert position sensor matches qpos
                pos_match = np.isclose(sensor_pos_val, qpos_val, atol=1e-12)
                if not pos_match:
                    print(
                        f"FAIL at state {state_idx}, joint {joint_name}:\n"
                        f"  qpos={qpos_val:.16e}, sensor_pos={sensor_pos_val:.16e}\n"
                        f"  difference={abs(qpos_val - sensor_pos_val):.16e}"
                    )
                    assert (
                        pos_match
                    ), f"Position sensor '{joint_name}' does not match qpos at state {state_idx}"

                # Assert velocity sensor matches qvel
                vel_match = np.isclose(sensor_vel_val, qvel_val, atol=1e-12)
                if not vel_match:
                    print(
                        f"FAIL at state {state_idx}, joint {joint_name}:\n"
                        f"  qvel={qvel_val:.16e}, sensor_vel={sensor_vel_val:.16e}\n"
                        f"  difference={abs(qvel_val - sensor_vel_val):.16e}"
                    )
                    assert (
                        vel_match
                    ), f"Velocity sensor '{joint_name}' does not match qvel at state {state_idx}"

            if (state_idx + 1) % 5 == 0:
                print(f"  Completed {state_idx + 1}/{num_test_states} states ✓")

        print(f"\n--- Test PASSED ---\n")

    def test_individual_joint_ranges_and_sensors(self, model_path_jetson):
        """
        Test individual joint ranges and sensor availability.

        This test:
        - Verifies that each joint has a valid range
        - Verifies that each joint has exactly one pos and one vel sensor
        - Checks that joint ranges are sensible (both bounds non-zero, unless intentional)
        """
        model = mujoco.MjModel.from_xml_path(model_path_jetson)

        joint_names = [
            "hip_roll_l",
            "hip_pitch_l",
            "knee_l",
            "hip_roll_r",
            "hip_pitch_r",
            "knee_r",
            "ankle_l",
            "ankle_r",
        ]

        print(f"\n--- Individual Joint Ranges and Sensors ---\n")

        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            assert joint_id >= 0, f"Joint '{joint_name}' not found"

            # Get range
            jnt_range = model.jnt_range[joint_id]
            low, high = jnt_range[0], jnt_range[1]

            # Verify range is valid
            assert low < high, (
                f"Joint '{joint_name}' has invalid range: [{low}, {high}]. "
                f"Lower bound must be less than upper bound."
            )

            # Check sensors exist
            pos_sensor_name = f"pos_{joint_name}"
            vel_sensor_name = f"vel_{joint_name}"

            pos_sensor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, pos_sensor_name
            )
            vel_sensor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, vel_sensor_name
            )

            assert pos_sensor_id >= 0, (
                f"Position sensor '{pos_sensor_name}' not found for joint '{joint_name}'"
            )
            assert vel_sensor_id >= 0, (
                f"Velocity sensor '{vel_sensor_name}' not found for joint '{joint_name}'"
            )

            print(f"{joint_name:15s}: range=[{low:8.5f}, {high:8.5f}] rad, sensors OK")

        print(f"\n--- Test PASSED ---\n")

    def test_framequat_torso_quat_validation(self, model_path_jetson):
        """
        Validate that the torso_quat framequat sensor exactly matches data.xquat[torso_body_id].

        This test:
        - Loads the biped model and creates MjData
        - Resets to the "stand" keyframe (so leg joint angles are 0, torso/base relationship predictable)
        - Generates 20+ randomized unit quaternions with seed=123 for reproducibility
        - For each random quaternion:
          - Sets data.qpos[3:7] to the quaternion (free-floating base's orientation)
          - Calls mujoco.mj_forward() to compute sensor values
          - Reads the torso_quat sensor value and data.xquat[torso_body_id]
          - Uses sign-ambiguity-safe quaternion distance: dist = min(||q1-q2||, ||q1+q2||)
          - Asserts dist < 1e-9 (essentially exact match)

        The torso body "composite_part_1__1_" is the IMU site's parent, so the framequat
        sensor should exactly equal the body's orientation in the world frame.
        """
        # Load model and create data
        model = mujoco.MjModel.from_xml_path(model_path_jetson)
        data = mujoco.MjData(model)

        # Reset to "stand" keyframe
        keyframe_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        assert keyframe_id >= 0, "Keyframe 'stand' not found in model"
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

        # Get torso body ID
        torso_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "composite_part_1__1_"
        )
        assert torso_body_id >= 0, "Torso body 'composite_part_1__1_' not found in model"

        # Get torso_quat sensor ID and address
        torso_quat_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_quat"
        )
        assert torso_quat_sensor_id >= 0, "Sensor 'torso_quat' not found in model"

        torso_quat_sensor_adr = model.sensor_adr[torso_quat_sensor_id]

        print(f"\n--- Framequat Torso Validation ---")
        print(f"Torso body ID: {torso_body_id}")
        print(f"Torso quat sensor address: {torso_quat_sensor_adr}\n")

        # Seed for reproducibility
        np.random.seed(123)

        # Number of test quaternions
        num_test_quats = 20

        print(f"--- Generating {num_test_quats} randomized unit quaternions ---\n")

        # Helper function: sign-ambiguity-safe quaternion distance
        def quat_distance(q1, q2):
            """Compute minimum distance between q1 and q2, accounting for sign ambiguity."""
            dist_same_sign = np.linalg.norm(q1 - q2)
            dist_opposite_sign = np.linalg.norm(q1 + q2)
            return min(dist_same_sign, dist_opposite_sign)

        # Loop over randomized quaternions
        for quat_idx in range(num_test_quats):
            # Generate random unit quaternion
            q = np.random.randn(4)
            q /= np.linalg.norm(q)

            # Set base orientation to the random quaternion
            data.qpos[3:7] = q

            # Compute sensor values
            mujoco.mj_forward(model, data)

            # Read sensor value (4-element quaternion)
            sensor_quat = data.sensordata[torso_quat_sensor_adr : torso_quat_sensor_adr + 4]

            # Read body quaternion from world frame
            body_quat = data.xquat[torso_body_id]

            # Compute sign-ambiguity-safe distance
            dist = quat_distance(sensor_quat, body_quat)

            # Assert match (very tight tolerance, should be essentially exact)
            if dist >= 1e-9:
                print(
                    f"FAIL at quaternion {quat_idx}:\n"
                    f"  input base quat: {q}\n"
                    f"  sensor_quat:     {sensor_quat}\n"
                    f"  body_quat:       {body_quat}\n"
                    f"  distance:        {dist:.16e}"
                )
            assert dist < 1e-9, (
                f"Framequat sensor does not match body orientation at quaternion {quat_idx}: "
                f"distance={dist:.16e}"
            )

            if (quat_idx + 1) % 5 == 0:
                print(f"  Completed {quat_idx + 1}/{num_test_quats} quaternions ✓")

        print(f"\n--- Test PASSED ---\n")

    def test_gyro_accel_forced_static(self, model_path_jetson):
        """
        Test gyro and accelerometer readings using the forced-static methodology.

        Since the biped cannot reach a genuine static equilibrium (weak actuators,
        topples in free-floating mode), we instead force a hypothetical static instant
        by manually calling MuJoCo's internal computation stages and overriding qacc to zero.

        This test validates at THREE different orientations:
        1. Upright (stand keyframe)
        2. 90-degree rotation about world x-axis
        3. 45-degree rotation about world z-axis (custom orientation)

        For each orientation:
        - Reset to that state (qvel=0, qacc forced to 0)
        - Compute expected_accel = site_xmat.T @ [0, 0, 9.81]
        - Assert gyro values are all < 1e-6 (should be zero since qvel=0)
        - Assert accelerometer matches expected_accel with atol=1e-3
        """
        # Load model and create data
        model = mujoco.MjModel.from_xml_path(model_path_jetson)
        data = mujoco.MjData(model)

        # Get IMU site ID
        imu_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        assert imu_site_id >= 0, "IMU site 'imu' not found in model"

        # Get sensor IDs and addresses
        gyro_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_gyro"
        )
        accel_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_acc"
        )

        assert gyro_sensor_id >= 0, "Gyro sensor 'torso_gyro' not found"
        assert accel_sensor_id >= 0, "Accelerometer sensor 'torso_acc' not found"

        gyro_sensor_adr = model.sensor_adr[gyro_sensor_id]
        accel_sensor_adr = model.sensor_adr[accel_sensor_id]

        print(f"\n--- Gyro/Accelerometer Forced-Static Test ---")
        print(f"IMU site ID: {imu_site_id}")
        print(f"Gyro sensor address: {gyro_sensor_adr}")
        print(f"Accelerometer sensor address: {accel_sensor_adr}\n")

        # Define three test orientations
        orientations = [
            {
                "name": "Upright (stand keyframe)",
                "setup": lambda: self._setup_upright(model, data),
            },
            {
                "name": "90-degree rotation about world x-axis",
                "setup": lambda: self._setup_90deg_x_rotation(model, data),
            },
            {
                "name": "45-degree rotation about world z-axis",
                "setup": lambda: self._setup_45deg_z_rotation(model, data),
            },
        ]

        # Test each orientation
        for orientation in orientations:
            print(f"Testing: {orientation['name']}")

            # Setup orientation
            orientation["setup"]()

            # Apply forced-static sequence
            # Step 1: Forward position (compute kinematics/orientations from qpos)
            mujoco.mj_fwdPosition(model, data)

            # Step 2: Forward velocity (compute velocity-dependent quantities from qvel)
            mujoco.mj_fwdVelocity(model, data)

            # Step 3: FORCE zero acceleration (hypothetical perfect rest)
            data.qacc[:] = 0

            # Step 4: Compute acceleration-dependent sensors from forced qacc
            mujoco.mj_sensorAcc(model, data)

            # Step 5: Compute velocity-dependent sensors from qvel
            mujoco.mj_sensorVel(model, data)

            # Get gyro and accelerometer readings
            gyro_reading = data.sensordata[gyro_sensor_adr : gyro_sensor_adr + 3]
            accel_reading = data.sensordata[accel_sensor_adr : accel_sensor_adr + 3]

            # Compute expected accelerometer: site_xmat.T @ [0, 0, 9.81]
            site_xmat = data.site_xmat[imu_site_id].reshape(3, 3)
            expected_accel = site_xmat.T @ np.array([0.0, 0.0, 9.81])

            # Get gyro magnitude for display
            gyro_mag = np.linalg.norm(gyro_reading)

            # Check gyro (should be ~0, qvel is 0)
            gyro_ok = np.all(np.abs(gyro_reading) < 1e-6)
            if not gyro_ok:
                print(
                    f"  GYRO FAIL: readings={gyro_reading}, magnitudes={np.abs(gyro_reading)}"
                )
            assert gyro_ok, (
                f"Gyro reading not close to zero: {gyro_reading}"
            )

            # Check accelerometer
            accel_ok = np.allclose(accel_reading, expected_accel, atol=1e-3)
            if not accel_ok:
                print(
                    f"  ACCEL FAIL:")
                print(
                    f"    sensor_reading:  {accel_reading}"
                )
                print(
                    f"    expected_accel:  {expected_accel}"
                )
                print(
                    f"    difference:      {accel_reading - expected_accel}"
                )
                print(
                    f"    max_abs_diff:    {np.max(np.abs(accel_reading - expected_accel))}"
                )
            assert accel_ok, (
                f"Accelerometer reading does not match expected: "
                f"sensor={accel_reading}, expected={expected_accel}"
            )

            print(
                f"  ✓ Gyro OK (mag={gyro_mag:.2e}), "
                f"Accel OK (error={np.max(np.abs(accel_reading - expected_accel)):.2e})\n"
            )

        print(f"--- Test PASSED ---\n")

    @staticmethod
    def _setup_upright(model, data):
        """Reset to upright (stand keyframe) orientation."""
        keyframe_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        assert keyframe_id >= 0, "Keyframe 'stand' not found"
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

    @staticmethod
    def _setup_90deg_x_rotation(model, data):
        """Reset to 90-degree rotation about world x-axis."""
        mujoco.mj_resetData(model, data)
        # Rotation of 90 degrees about x-axis: quat = [cos(pi/4), sin(pi/4), 0, 0]
        # This rotates the body 90 degrees around the x-axis
        data.qpos[3:7] = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])

    @staticmethod
    def _setup_45deg_z_rotation(model, data):
        """Reset to 45-degree rotation about world z-axis."""
        mujoco.mj_resetData(model, data)
        # Rotation of 45 degrees about z-axis: quat = [cos(pi/8), 0, 0, sin(pi/8)]
        data.qpos[3:7] = np.array([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)])

    def test_gyro_accel_in_motion_validation(self, model_path_jetson):
        """
        Test gyro and accelerometer readings during real dynamic motion.

        This test validates that gyro and accelerometer sensor readings exactly match
        ground truth computed via mujoco.mj_objectVelocity and mujoco.mj_objectAcceleration
        during real (non-static) motion with actual dynamics-derived acceleration.

        This test:
        - Loads the biped model and creates MjData
        - Resets to the "stand" keyframe
        - Generates 20+ randomized dynamic states with seed=7 for reproducibility:
          * For each state: set data.qvel[:] to small random values (uniform in [-0.5, 0.5])
          * Randomize the base orientation (data.qpos[3:7]) to a unit quaternion for variety
        - For each state:
          * Calls mujoco.mj_forward(model, data) to compute REAL qacc from dynamics
          * Reads the torso_gyro sensor value
          * Computes ground truth gyro via mujoco.mj_objectVelocity(..., flg_local=1) → vel6[:3]
          * Asserts np.allclose(gyro_sensor, ground_truth_gyro, atol=1e-6)
          * Reads the torso_acc sensor value
          * Computes ground truth accel via mujoco.mj_objectAcceleration(..., flg_local=1) → acc6[3:]
          * Asserts np.allclose(accel_sensor, ground_truth_accel, atol=1e-6)
        - Verifies that gyro and accelerometer readings vary across the 20+ states
          (i.e., the test is not vacuous) by asserting that std(gyro_readings) > 1e-4
          across all states.

        This validates that MuJoCo's gyro and accelerometer sensors are exact functions
        of the current state and dynamics, not approximations.
        """
        # Load model and create data
        model = mujoco.MjModel.from_xml_path(model_path_jetson)
        data = mujoco.MjData(model)

        # Reset to "stand" keyframe
        keyframe_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        assert keyframe_id >= 0, "Keyframe 'stand' not found in model"
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

        # Get IMU site ID
        imu_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        assert imu_site_id >= 0, "IMU site 'imu' not found in model"

        # Get sensor IDs and addresses
        gyro_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_gyro"
        )
        accel_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_acc"
        )

        assert gyro_sensor_id >= 0, "Gyro sensor 'torso_gyro' not found"
        assert accel_sensor_id >= 0, "Accelerometer sensor 'torso_acc' not found"

        gyro_sensor_adr = model.sensor_adr[gyro_sensor_id]
        accel_sensor_adr = model.sensor_adr[accel_sensor_id]

        print(f"\n--- Gyro/Accelerometer In-Motion Validation Test ---")
        print(f"IMU site ID: {imu_site_id}")
        print(f"Gyro sensor address: {gyro_sensor_adr}")
        print(f"Accelerometer sensor address: {accel_sensor_adr}\n")

        # Seed for reproducibility
        np.random.seed(7)

        # Number of test states
        num_test_states = 25

        print(f"--- Generating {num_test_states} randomized dynamic states ---\n")

        # Storage for gyro readings across all states (for variation check)
        all_gyro_readings = []

        # Loop over randomized test states
        for state_idx in range(num_test_states):
            # Set qvel to small random values across all DOFs (real velocity)
            data.qvel[:] = np.random.uniform(-0.5, 0.5, size=model.nv)

            # Randomize base orientation (qpos[3:7] is base quaternion)
            # Generate a random unit quaternion
            q = np.random.randn(4)
            q /= np.linalg.norm(q)
            data.qpos[3:7] = q

            # Call mj_forward to compute REAL qacc from dynamics
            # (NOT forced-static like the previous test)
            mujoco.mj_forward(model, data)

            # Read gyro sensor value (3-element vector in site/local frame)
            gyro_reading = data.sensordata[gyro_sensor_adr : gyro_sensor_adr + 3]
            all_gyro_readings.append(gyro_reading.copy())

            # Compute ground truth gyro via mj_objectVelocity
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_SITE, imu_site_id, vel6, 1
            )
            ground_truth_gyro = vel6[:3]

            # Assert gyro matches ground truth (tight 1e-6 tolerance)
            gyro_match = np.allclose(gyro_reading, ground_truth_gyro, atol=1e-6)
            if not gyro_match:
                print(
                    f"FAIL at state {state_idx} - GYRO MISMATCH:\n"
                    f"  sensor_gyro:       {gyro_reading}\n"
                    f"  ground_truth_gyro: {ground_truth_gyro}\n"
                    f"  difference:        {gyro_reading - ground_truth_gyro}\n"
                    f"  max_abs_diff:      {np.max(np.abs(gyro_reading - ground_truth_gyro)):.16e}"
                )
            assert gyro_match, (
                f"Gyro sensor does not match ground truth at state {state_idx}: "
                f"sensor={gyro_reading}, ground_truth={ground_truth_gyro}"
            )

            # Read accelerometer sensor value
            accel_reading = data.sensordata[accel_sensor_adr : accel_sensor_adr + 3]

            # Compute ground truth accel via mj_objectAcceleration
            acc6 = np.zeros(6)
            mujoco.mj_objectAcceleration(
                model, data, mujoco.mjtObj.mjOBJ_SITE, imu_site_id, acc6, 1
            )
            ground_truth_accel = acc6[3:]

            # Assert accelerometer matches ground truth (tight 1e-6 tolerance)
            accel_match = np.allclose(accel_reading, ground_truth_accel, atol=1e-6)
            if not accel_match:
                print(
                    f"FAIL at state {state_idx} - ACCELEROMETER MISMATCH:\n"
                    f"  sensor_accel:       {accel_reading}\n"
                    f"  ground_truth_accel: {ground_truth_accel}\n"
                    f"  difference:         {accel_reading - ground_truth_accel}\n"
                    f"  max_abs_diff:       {np.max(np.abs(accel_reading - ground_truth_accel)):.16e}"
                )
            assert accel_match, (
                f"Accelerometer sensor does not match ground truth at state {state_idx}: "
                f"sensor={accel_reading}, ground_truth={ground_truth_accel}"
            )

            if (state_idx + 1) % 5 == 0:
                print(f"  Completed {state_idx + 1}/{num_test_states} states ✓")

        # Verify that test is not vacuous: gyro readings vary across states
        all_gyro_readings = np.array(all_gyro_readings)  # shape (num_test_states, 3)
        gyro_std = np.std(all_gyro_readings, axis=0)  # std along each axis
        max_gyro_std = np.max(gyro_std)

        print(f"\n--- Gyro Variation Check ---")
        print(f"Gyro std per axis: {gyro_std}")
        print(f"Max gyro std: {max_gyro_std:.16e}\n")

        assert (
            max_gyro_std > 1e-4
        ), (
            f"Gyro readings do not vary across states (all identical or near-identical). "
            f"Max std: {max_gyro_std:.16e} (expected > 1e-4). "
            f"Test is vacuous; randomized states were not sufficiently diverse."
        )

        print(f"--- Test PASSED ---\n")

    def test_touch_sensor_validation(self, model_path_jetson):
        """
        Validate touch sensor readings in airborne and contact scenarios.

        This test verifies the touch sensors (touch_l and touch_r) behavior in two scenarios:

        1. Airborne case (no ground contact):
           - Load model and create MjData
           - Reset to "stand" keyframe
           - Elevate the base well above ground: data.qpos[2] = 2.0
           - Zero all velocities: data.qvel[:] = 0
           - Call mujoco.mj_forward() once
           - Assert data.ncon == 0 (no contacts)
           - Assert both touch_l and touch_r sensors read exactly 0.0

        2. Settled/contact case:
           - Reset to "stand" keyframe (fresh)
           - Call mujoco.mj_forward()
           - Step 1000 times with mujoco.mj_step() to let the robot settle
           - Assert data.ncon > 0 (some contacts exist)
           - Assert both touch_l and touch_r sensors read > 0

        3. Cross-check consistency:
           - At the settled state from case 2
           - Verify that if data.ncon > 0, at least one of the two touch sensors is > 0
           - This ensures basic consistency between contact existence and sensor readings
        """
        # Load model and create data
        model = mujoco.MjModel.from_xml_path(model_path_jetson)
        data = mujoco.MjData(model)

        # Get touch sensor IDs and addresses
        touch_l_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_l"
        )
        touch_r_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_r"
        )

        assert touch_l_sensor_id >= 0, "Sensor 'touch_l' not found in model"
        assert touch_r_sensor_id >= 0, "Sensor 'touch_r' not found in model"

        touch_l_sensor_adr = model.sensor_adr[touch_l_sensor_id]
        touch_r_sensor_adr = model.sensor_adr[touch_r_sensor_id]

        print(f"\n--- Touch Sensor Validation Test ---")
        print(f"Touch_l sensor address: {touch_l_sensor_adr}")
        print(f"Touch_r sensor address: {touch_r_sensor_adr}\n")

        # ========== CASE 1: AIRBORNE (no ground contact) ==========
        print("Case 1: AIRBORNE (elevated, no contacts)")
        print("-" * 50)

        # Reset to "stand" keyframe
        keyframe_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        assert keyframe_id >= 0, "Keyframe 'stand' not found in model"
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

        # Elevate base well above ground
        data.qpos[2] = 2.0

        # Zero all velocities
        data.qvel[:] = 0

        # Call mj_forward once to compute sensor values
        mujoco.mj_forward(model, data)

        # Assert no contacts
        num_contacts_airborne = data.ncon
        print(f"  Number of contacts: {num_contacts_airborne}")
        assert (
            num_contacts_airborne == 0
        ), f"Expected 0 contacts in airborne state, got {num_contacts_airborne}"

        # Read touch sensor values
        touch_l_airborne = data.sensordata[touch_l_sensor_adr]
        touch_r_airborne = data.sensordata[touch_r_sensor_adr]

        print(f"  touch_l sensor reading: {touch_l_airborne}")
        print(f"  touch_r sensor reading: {touch_r_airborne}")

        # Assert both are exactly 0.0
        assert (
            touch_l_airborne == 0.0
        ), f"Expected touch_l = 0.0 in airborne state, got {touch_l_airborne}"
        assert (
            touch_r_airborne == 0.0
        ), f"Expected touch_r = 0.0 in airborne state, got {touch_r_airborne}"

        print("  ✓ Airborne case PASSED: ncon=0, both touch sensors = 0.0\n")

        # ========== CASE 2: SETTLED/CONTACT (1000 steps from stand) ==========
        print("Case 2: SETTLED/CONTACT (1000 steps from stand)")
        print("-" * 50)

        # Reset to "stand" keyframe fresh
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

        # Call mj_forward once
        mujoco.mj_forward(model, data)

        # Step 1000 times to let it settle
        for step_idx in range(1000):
            mujoco.mj_step(model, data)

        # Assert contacts exist
        num_contacts_settled = data.ncon
        print(f"  Number of contacts after 1000 steps: {num_contacts_settled}")
        assert (
            num_contacts_settled > 0
        ), f"Expected contacts in settled state, got {num_contacts_settled}"

        # Read touch sensor values
        touch_l_settled = data.sensordata[touch_l_sensor_adr]
        touch_r_settled = data.sensordata[touch_r_sensor_adr]

        print(f"  touch_l sensor reading: {touch_l_settled}")
        print(f"  touch_r sensor reading: {touch_r_settled}")

        # Assert both are > 0
        assert (
            touch_l_settled > 0
        ), f"Expected touch_l > 0 in settled state, got {touch_l_settled}"
        assert (
            touch_r_settled > 0
        ), f"Expected touch_r > 0 in settled state, got {touch_r_settled}"

        print(
            f"  ✓ Settled case PASSED: ncon={num_contacts_settled}, "
            f"touch_l={touch_l_settled:.6f}, touch_r={touch_r_settled:.6f}\n"
        )

        # ========== CASE 3: CROSS-CHECK CONSISTENCY ==========
        print("Case 3: CROSS-CHECK CONSISTENCY")
        print("-" * 50)

        # At settled state, verify consistency: if ncon > 0, at least one touch sensor > 0
        if num_contacts_settled > 0:
            at_least_one_touch = (touch_l_settled > 0) or (touch_r_settled > 0)
            print(
                f"  ncon={num_contacts_settled} > 0: at least one touch sensor should be > 0"
            )
            assert at_least_one_touch, (
                f"Expected at least one touch sensor > 0 when ncon={num_contacts_settled}, "
                f"but got touch_l={touch_l_settled}, touch_r={touch_r_settled}"
            )
            print(f"  ✓ Consistency check PASSED\n")

        print("--- Test PASSED ---\n")

    def test_integration_trajectory_all_sensors(self, model_path_jetson):
        """
        Integration test validating ALL sensor channels over a real dynamic trajectory.

        This test:
        - Loads the biped model and creates MjData
        - Resets to the "stand" keyframe, then elevates base to 0.5m height
        - Steps the simulation 1000 times to cover a real trajectory (fall, impact, settling)
        - At EVERY step, records:
          * All 8 jointpos/jointvel sensor readings vs ground truth (qpos/qvel)
          * torso_quat sensor vs data.xquat[torso_body_id]
          * torso_gyro sensor vs mj_objectVelocity ground truth (local frame)
          * torso_acc sensor vs mj_objectAcceleration ground truth (local frame)
          * touch_l/touch_r sensor values and data.ncon (qualitative check)
        - Computes RMS error for each channel across all 1000 steps
        - Asserts RMS errors < tight tolerances:
          * jointpos/jointvel RMS < 1e-10
          * torso_quat RMS (quaternion distance) < 1e-6
          * torso_gyro RMS < 1e-6
          * torso_acc RMS < 1e-6
        - Writes detailed CSV trace to docs/sensor_validation/integration_trajectory_trace.csv
        - Prints final RMS errors for each channel category
        """
        # Load model and create data
        model = mujoco.MjModel.from_xml_path(model_path_jetson)
        data = mujoco.MjData(model)

        # Reset to "stand" keyframe
        keyframe_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        assert keyframe_id >= 0, "Keyframe 'stand' not found in model"
        mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

        # Elevate base to 0.5m (moderate height for real airborne phase)
        data.qpos[2] = 0.5
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        # Define the 8 joint names
        joint_names = [
            "hip_roll_l",
            "hip_pitch_l",
            "knee_l",
            "hip_roll_r",
            "hip_pitch_r",
            "knee_r",
            "ankle_l",
            "ankle_r",
        ]

        # Build joint info dictionary
        joint_info = {}
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            assert joint_id >= 0, f"Joint '{joint_name}' not found"

            qposadr = model.jnt_qposadr[joint_id]
            dofadr = model.jnt_dofadr[joint_id]

            pos_sensor_name = f"pos_{joint_name}"
            vel_sensor_name = f"vel_{joint_name}"

            pos_sensor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, pos_sensor_name
            )
            vel_sensor_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, vel_sensor_name
            )

            assert pos_sensor_id >= 0, f"Sensor '{pos_sensor_name}' not found"
            assert vel_sensor_id >= 0, f"Sensor '{vel_sensor_name}' not found"

            pos_sensor_adr = model.sensor_adr[pos_sensor_id]
            vel_sensor_adr = model.sensor_adr[vel_sensor_id]

            joint_info[joint_name] = {
                "qposadr": qposadr,
                "dofadr": dofadr,
                "pos_sensor_adr": pos_sensor_adr,
                "vel_sensor_adr": vel_sensor_adr,
            }

        # Get torso body and IMU site IDs
        torso_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "composite_part_1__1_"
        )
        assert torso_body_id >= 0, "Torso body 'composite_part_1__1_' not found"

        imu_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        assert imu_site_id >= 0, "IMU site 'imu' not found"

        # Get sensor IDs and addresses
        torso_quat_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_quat"
        )
        torso_gyro_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_gyro"
        )
        torso_acc_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_acc"
        )
        touch_l_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_l"
        )
        touch_r_sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_r"
        )

        assert torso_quat_sensor_id >= 0, "Sensor 'torso_quat' not found"
        assert torso_gyro_sensor_id >= 0, "Sensor 'torso_gyro' not found"
        assert torso_acc_sensor_id >= 0, "Sensor 'torso_acc' not found"
        assert touch_l_sensor_id >= 0, "Sensor 'touch_l' not found"
        assert touch_r_sensor_id >= 0, "Sensor 'touch_r' not found"

        torso_quat_sensor_adr = model.sensor_adr[torso_quat_sensor_id]
        torso_gyro_sensor_adr = model.sensor_adr[torso_gyro_sensor_id]
        torso_acc_sensor_adr = model.sensor_adr[torso_acc_sensor_id]
        touch_l_sensor_adr = model.sensor_adr[touch_l_sensor_id]
        touch_r_sensor_adr = model.sensor_adr[touch_r_sensor_id]

        print("\n--- Integration Trajectory Test (All Sensors) ---")
        print(f"Stepping 1000 times from elevated state (0.5m)...\n")

        # Helper function: quaternion distance (sign-ambiguity-safe)
        def quat_distance(q1, q2):
            dist_same_sign = np.linalg.norm(q1 - q2)
            dist_opposite_sign = np.linalg.norm(q1 + q2)
            return min(dist_same_sign, dist_opposite_sign)

        # Storage for all errors across steps
        jointpos_errors = []  # Will collect 8*1000 values
        jointvel_errors = []  # Will collect 8*1000 values
        quat_dist_errors = []  # Will collect 1000 values
        gyro_errors = []  # Will collect 3*1000 values
        acc_errors = []  # Will collect 3*1000 values

        # Storage for CSV output
        csv_rows = []

        # Step 1000 times and record
        for step_idx in range(1000):
            # Step simulation
            mujoco.mj_step(model, data)
            # IMPORTANT: mj_step computes position/velocity-stage sensors (e.g.
            # jointpos/jointvel) at the START of the step, before the integrator
            # advances qpos/qvel - so immediately after mj_step returns,
            # data.sensordata for these channels is one full step STALE relative
            # to the just-updated data.qpos/data.qvel (verified: sensor reading
            # after mj_step exactly equals qpos from BEFORE that step, not after).
            # This is a real MuJoCo pipeline-timing characteristic, not a bug.
            # An extra mj_forward resyncs sensordata to the current state before
            # we read anything for comparison.
            mujoco.mj_forward(model, data)

            # Record time
            time = data.time

            # Dictionary for this step's CSV row
            csv_row = {
                "step": step_idx,
                "time": time,
                "ncon": data.ncon,
                "touch_l": data.sensordata[touch_l_sensor_adr],
                "touch_r": data.sensordata[touch_r_sensor_adr],
            }

            # Joint position and velocity errors
            for joint_name, info in joint_info.items():
                qposadr = info["qposadr"]
                dofadr = info["dofadr"]
                pos_sensor_adr = info["pos_sensor_adr"]
                vel_sensor_adr = info["vel_sensor_adr"]

                # Ground truth
                qpos_truth = data.qpos[qposadr]
                qvel_truth = data.qvel[dofadr]

                # Sensor readings
                pos_sensor = data.sensordata[pos_sensor_adr]
                vel_sensor = data.sensordata[vel_sensor_adr]

                # Errors
                pos_error = abs(pos_sensor - qpos_truth)
                vel_error = abs(vel_sensor - qvel_truth)

                jointpos_errors.append(pos_error)
                jointvel_errors.append(vel_error)

                # CSV columns
                csv_row[f"{joint_name}_pos_sensor"] = pos_sensor
                csv_row[f"{joint_name}_pos_truth"] = qpos_truth
                csv_row[f"{joint_name}_vel_sensor"] = vel_sensor
                csv_row[f"{joint_name}_vel_truth"] = qvel_truth

            # Torso quaternion error
            sensor_quat = data.sensordata[
                torso_quat_sensor_adr : torso_quat_sensor_adr + 4
            ]
            body_quat = data.xquat[torso_body_id]
            quat_dist = quat_distance(sensor_quat, body_quat)
            quat_dist_errors.append(quat_dist)

            csv_row["quat_dist_error"] = quat_dist

            # Torso gyro error
            sensor_gyro = data.sensordata[
                torso_gyro_sensor_adr : torso_gyro_sensor_adr + 3
            ]
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_SITE, imu_site_id, vel6, 1
            )
            truth_gyro = vel6[:3]
            gyro_error = np.abs(sensor_gyro - truth_gyro)
            gyro_errors.extend(gyro_error)

            csv_row["gyro_x_sensor"] = sensor_gyro[0]
            csv_row["gyro_x_truth"] = truth_gyro[0]
            csv_row["gyro_y_sensor"] = sensor_gyro[1]
            csv_row["gyro_y_truth"] = truth_gyro[1]
            csv_row["gyro_z_sensor"] = sensor_gyro[2]
            csv_row["gyro_z_truth"] = truth_gyro[2]

            # Torso accelerometer error
            sensor_acc = data.sensordata[
                torso_acc_sensor_adr : torso_acc_sensor_adr + 3
            ]
            acc6 = np.zeros(6)
            mujoco.mj_objectAcceleration(
                model, data, mujoco.mjtObj.mjOBJ_SITE, imu_site_id, acc6, 1
            )
            truth_acc = acc6[3:]
            acc_error = np.abs(sensor_acc - truth_acc)
            acc_errors.extend(acc_error)

            csv_row["acc_x_sensor"] = sensor_acc[0]
            csv_row["acc_x_truth"] = truth_acc[0]
            csv_row["acc_y_sensor"] = sensor_acc[1]
            csv_row["acc_y_truth"] = truth_acc[1]
            csv_row["acc_z_sensor"] = sensor_acc[2]
            csv_row["acc_z_truth"] = truth_acc[2]

            csv_rows.append(csv_row)

            if (step_idx + 1) % 100 == 0:
                print(f"  Completed {step_idx + 1}/1000 steps")

        # Convert errors to numpy arrays
        jointpos_errors = np.array(jointpos_errors)
        jointvel_errors = np.array(jointvel_errors)
        quat_dist_errors = np.array(quat_dist_errors)
        gyro_errors = np.array(gyro_errors)
        acc_errors = np.array(acc_errors)

        # Compute RMS errors
        jointpos_rms = np.sqrt(np.mean(jointpos_errors**2))
        jointvel_rms = np.sqrt(np.mean(jointvel_errors**2))
        quat_rms = np.sqrt(np.mean(quat_dist_errors**2))
        gyro_rms = np.sqrt(np.mean(gyro_errors**2))
        acc_rms = np.sqrt(np.mean(acc_errors**2))

        # Write CSV
        csv_path = "docs/sensor_validation/integration_trajectory_trace.csv"
        with open(csv_path, "w", newline="") as f:
            # Determine column order
            fieldnames = [
                "step",
                "time",
            ]
            # Add joint columns
            for joint_name in joint_names:
                fieldnames.append(f"{joint_name}_pos_sensor")
                fieldnames.append(f"{joint_name}_pos_truth")
                fieldnames.append(f"{joint_name}_vel_sensor")
                fieldnames.append(f"{joint_name}_vel_truth")
            # Add quaternion, gyro, accel columns
            fieldnames.extend(
                [
                    "quat_dist_error",
                    "gyro_x_sensor",
                    "gyro_x_truth",
                    "gyro_y_sensor",
                    "gyro_y_truth",
                    "gyro_z_sensor",
                    "gyro_z_truth",
                    "acc_x_sensor",
                    "acc_x_truth",
                    "acc_y_sensor",
                    "acc_y_truth",
                    "acc_z_sensor",
                    "acc_z_truth",
                    "ncon",
                    "touch_l",
                    "touch_r",
                ]
            )

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\nCSV written to: {csv_path}\n")

        # Print RMS errors
        print("--- RMS Errors (across 1000 steps) ---")
        print(f"Joint Position RMS:   {jointpos_rms:.16e} (tolerance: 1e-10)")
        print(f"Joint Velocity RMS:   {jointvel_rms:.16e} (tolerance: 1e-10)")
        print(f"Torso Quat RMS:       {quat_rms:.16e} (tolerance: 1e-6)")
        print(f"Torso Gyro RMS:       {gyro_rms:.16e} (tolerance: 1e-6)")
        print(f"Torso Accel RMS:      {acc_rms:.16e} (tolerance: 1e-6)\n")

        # Assert tight tolerances
        assert jointpos_rms < 1e-10, (
            f"Joint position RMS {jointpos_rms:.16e} exceeds tolerance 1e-10"
        )
        assert jointvel_rms < 1e-10, (
            f"Joint velocity RMS {jointvel_rms:.16e} exceeds tolerance 1e-10"
        )
        assert quat_rms < 1e-6, (
            f"Torso quat RMS {quat_rms:.16e} exceeds tolerance 1e-6"
        )
        assert gyro_rms < 1e-6, (
            f"Torso gyro RMS {gyro_rms:.16e} exceeds tolerance 1e-6"
        )
        assert acc_rms < 1e-6, (
            f"Torso accel RMS {acc_rms:.16e} exceeds tolerance 1e-6"
        )

        print("--- Test PASSED ---\n")
