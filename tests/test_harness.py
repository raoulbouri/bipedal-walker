import sys
import os
import glob
from pathlib import Path

import numpy as np
import pytest
import mujoco

# Add parent directory to path to import sim module
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim import BipedSim


class TestDeterminism:
    """Test determinism of reset and step."""

    def test_reset_bitwise_identical(self, model_path_jetson):
        """Reset twice should produce bitwise identical state."""
        sim = BipedSim(model_path_jetson, control_dt=0.02)

        # First reset
        sim.reset()
        qpos1 = sim.qpos()
        qvel1 = sim.qvel()

        # Second reset
        sim.reset()
        qpos2 = sim.qpos()
        qvel2 = sim.qvel()

        # Must be EXACTLY equal (bitwise)
        assert np.array_equal(qpos1, qpos2), "qpos not bitwise identical on reset"
        assert np.array_equal(qvel1, qvel2), "qvel not bitwise identical on reset"

    def test_reset_bitwise_identical_no_jetson(self, model_path_no_jetson):
        """Reset twice should produce bitwise identical state (no_jetson variant)."""
        sim = BipedSim(model_path_no_jetson, control_dt=0.02)

        sim.reset()
        qpos1 = sim.qpos()
        qvel1 = sim.qvel()

        sim.reset()
        qpos2 = sim.qpos()
        qvel2 = sim.qvel()

        assert np.array_equal(qpos1, qpos2), "qpos not bitwise identical on reset"
        assert np.array_equal(qvel1, qvel2), "qvel not bitwise identical on reset"

    def test_identical_sequences_independent_instances(self, model_path_jetson):
        """Two independent instances with identical ctrl sequences should match."""
        # Create two independent simulators
        sim1 = BipedSim(model_path_jetson, control_dt=0.02)
        sim2 = BipedSim(model_path_jetson, control_dt=0.02)

        # Reset both
        sim1.reset()
        sim2.reset()

        # Generate fixed control sequence (sinusoidal, small values)
        # Use small values safe for position actuators
        n_steps = 20
        ctrl_sequence = []
        for i in range(n_steps):
            # 0.1 * sin(i) should be within typical position actuator ranges
            ctrl = 0.1 * np.sin(np.linspace(0, 2 * np.pi, 6) + i)
            ctrl_sequence.append(ctrl)

        # Apply identical sequence to both
        for ctrl in ctrl_sequence:
            sim1.step(ctrl)
            sim2.step(ctrl)

        # Check EXACT match
        qpos1 = sim1.qpos()
        qpos2 = sim2.qpos()
        qvel1 = sim1.qvel()
        qvel2 = sim2.qvel()

        assert np.array_equal(qpos1, qpos2), "qpos diverged between instances"
        assert np.array_equal(qvel1, qvel2), "qvel diverged between instances"

    def test_identical_sequences_independent_instances_no_jetson(
        self, model_path_no_jetson
    ):
        """Two independent instances with identical ctrl sequences should match (no_jetson)."""
        sim1 = BipedSim(model_path_no_jetson, control_dt=0.02)
        sim2 = BipedSim(model_path_no_jetson, control_dt=0.02)

        sim1.reset()
        sim2.reset()

        n_steps = 20
        ctrl_sequence = []
        for i in range(n_steps):
            ctrl = 0.1 * np.sin(np.linspace(0, 2 * np.pi, 6) + i)
            ctrl_sequence.append(ctrl)

        for ctrl in ctrl_sequence:
            sim1.step(ctrl)
            sim2.step(ctrl)

        assert np.array_equal(sim1.qpos(), sim2.qpos()), "qpos diverged"
        assert np.array_equal(sim1.qvel(), sim2.qvel()), "qvel diverged"


class TestSubstepCount:
    """Test substep calculation and validation."""

    def test_substep_count_exact(self, model_path_jetson):
        """control_dt=0.02 should give n_substeps=10 for 2ms physics timestep."""
        sim = BipedSim(model_path_jetson, control_dt=0.02)
        assert sim.n_substeps == 10, f"Expected 10 substeps, got {sim.n_substeps}"

    def test_substep_count_non_multiple_raises(self, model_path_jetson):
        """control_dt that is not an exact multiple should raise ValueError."""
        with pytest.raises(ValueError):
            BipedSim(model_path_jetson, control_dt=0.0035)

    def test_substep_count_non_multiple_raises_no_jetson(self, model_path_no_jetson):
        """control_dt that is not an exact multiple should raise ValueError (no_jetson)."""
        with pytest.raises(ValueError):
            BipedSim(model_path_no_jetson, control_dt=0.0035)

    def test_substep_count_other_valid_multiples(self, model_path_jetson):
        """Test other valid control_dt values."""
        # 0.01 / 0.002 = 5 substeps
        sim1 = BipedSim(model_path_jetson, control_dt=0.01)
        assert sim1.n_substeps == 5

        # 0.004 / 0.002 = 2 substeps
        sim2 = BipedSim(model_path_jetson, control_dt=0.004)
        assert sim2.n_substeps == 2


class TestCtrlClipping:
    """Test control input clipping to actuator limits."""

    def test_clip_upper_bound(self, model_path_jetson):
        """Command above upper bound should be clipped (if limited)."""
        sim = BipedSim(model_path_jetson, control_dt=0.02)
        sim.reset()

        model = sim.model

        # Find a limited actuator
        limited_idx = None
        for i in range(model.nu):
            if model.actuator_ctrllimited[i]:
                limited_idx = i
                break

        if limited_idx is None:
            # No limited actuators - test that unlimited actuator passes value through
            limited_idx = 0
            ctrl = np.zeros(model.nu)
            ctrl[limited_idx] = 50.0  # Large value that would be clipped if limited
            sim.step(ctrl)
            actual_ctrl = sim.data.ctrl[limited_idx]
            assert actual_ctrl == 50.0, (
                f"Unlimited actuator should pass value through. "
                f"Expected 50.0, got {actual_ctrl}"
            )
            return

        lo, hi = model.actuator_ctrlrange[limited_idx]

        # Create control above the upper bound
        ctrl = np.zeros(model.nu)
        ctrl[limited_idx] = hi + 100.0

        # Step
        sim.step(ctrl)

        # Check that the control was clipped
        actual_ctrl = sim.data.ctrl[limited_idx]
        assert actual_ctrl == hi, (
            f"Control not clipped to upper bound. "
            f"Expected {hi}, got {actual_ctrl}"
        )

    def test_clip_lower_bound(self, model_path_jetson):
        """Command below lower bound should be clipped (if limited)."""
        sim = BipedSim(model_path_jetson, control_dt=0.02)
        sim.reset()

        model = sim.model

        # Find a limited actuator
        limited_idx = None
        for i in range(model.nu):
            if model.actuator_ctrllimited[i]:
                limited_idx = i
                break

        if limited_idx is None:
            # No limited actuators - test that unlimited actuator passes negative value through
            limited_idx = 0
            ctrl = np.zeros(model.nu)
            ctrl[limited_idx] = -50.0  # Large negative value that would be clipped if limited
            sim.step(ctrl)
            actual_ctrl = sim.data.ctrl[limited_idx]
            assert actual_ctrl == -50.0, (
                f"Unlimited actuator should pass value through. "
                f"Expected -50.0, got {actual_ctrl}"
            )
            return

        lo, hi = model.actuator_ctrlrange[limited_idx]

        # Create control below the lower bound
        ctrl = np.zeros(model.nu)
        ctrl[limited_idx] = lo - 100.0

        # Step
        sim.step(ctrl)

        # Check that the control was clipped
        actual_ctrl = sim.data.ctrl[limited_idx]
        assert actual_ctrl == lo, (
            f"Control not clipped to lower bound. "
            f"Expected {lo}, got {actual_ctrl}"
        )

    def test_clip_both_bounds_no_jetson(self, model_path_no_jetson):
        """Test clipping on both bounds (no_jetson model)."""
        sim = BipedSim(model_path_no_jetson, control_dt=0.02)
        sim.reset()

        model = sim.model

        # Find a limited actuator
        limited_idx = None
        for i in range(model.nu):
            if model.actuator_ctrllimited[i]:
                limited_idx = i
                break

        if limited_idx is None:
            # No limited actuators - test passthrough for both high and low values
            limited_idx = 0
            ctrl_high = np.zeros(model.nu)
            ctrl_high[limited_idx] = 100.0
            sim.step(ctrl_high)
            assert sim.data.ctrl[limited_idx] == 100.0

            # Reset and test with large negative value
            sim.reset()
            ctrl_low = np.zeros(model.nu)
            ctrl_low[limited_idx] = -100.0
            sim.step(ctrl_low)
            assert sim.data.ctrl[limited_idx] == -100.0
            return

        lo, hi = model.actuator_ctrlrange[limited_idx]

        # Test upper bound
        ctrl_high = np.zeros(model.nu)
        ctrl_high[limited_idx] = hi + 50.0
        sim.step(ctrl_high)
        assert sim.data.ctrl[limited_idx] == hi

        # Reset and test lower bound
        sim.reset()
        ctrl_low = np.zeros(model.nu)
        ctrl_low[limited_idx] = lo - 50.0
        sim.step(ctrl_low)
        assert sim.data.ctrl[limited_idx] == lo


class TestSensors:
    """Test sensor reading and output."""

    def test_sensors_dict_structure(self, model_path_jetson):
        """sensors() should return dict with correct structure."""
        sim = BipedSim(model_path_jetson, control_dt=0.02)
        sim.reset()

        sensors_dict = sim.sensors()

        # Should have exactly 24 sensors
        assert len(sensors_dict) == 24, (
            f"Expected 24 sensors, got {len(sensors_dict)}"
        )

        # Total size should match sum of sensor_dim
        total_size = sum(arr.size for arr in sensors_dict.values())
        expected_size = np.sum(sim.model.sensor_dim)
        assert total_size == expected_size, (
            f"Total sensor data size mismatch. "
            f"Expected {expected_size}, got {total_size}"
        )

    def test_sensors_dict_structure_no_jetson(self, model_path_no_jetson):
        """sensors() should return dict with correct structure (no_jetson)."""
        sim = BipedSim(model_path_no_jetson, control_dt=0.02)
        sim.reset()

        sensors_dict = sim.sensors()

        assert len(sensors_dict) == 24, (
            f"Expected 24 sensors, got {len(sensors_dict)}"
        )

        total_size = sum(arr.size for arr in sensors_dict.values())
        expected_size = np.sum(sim.model.sensor_dim)
        assert total_size == expected_size, (
            f"Total sensor data size mismatch. "
            f"Expected {expected_size}, got {total_size}"
        )

    def test_sensors_specific_shapes(self, model_path_jetson):
        """Check specific sensor shapes."""
        sim = BipedSim(model_path_jetson, control_dt=0.02)
        sim.reset()

        sensors_dict = sim.sensors()

        # Check that expected sensors exist and have reasonable shapes
        # torso_gyro should be 3D (angular velocity)
        if "torso_gyro" in sensors_dict:
            assert sensors_dict["torso_gyro"].size == 3, (
                f"torso_gyro should have 3 elements, "
                f"got {sensors_dict['torso_gyro'].size}"
            )

        # touch sensors are typically 1D
        if "touch_l" in sensors_dict:
            assert sensors_dict["touch_l"].size == 1, (
                f"touch_l should have 1 element, "
                f"got {sensors_dict['touch_l'].size}"
            )

    def test_sensors_are_copies(self, model_path_jetson):
        """sensors() should return copies, not views."""
        sim = BipedSim(model_path_jetson, control_dt=0.02)
        sim.reset()

        sensors_dict1 = sim.sensors()
        sensors_dict2 = sim.sensors()

        # Modifying dict1 should not affect dict2 or the sim's internal state
        for key in sensors_dict1:
            original_value = sensors_dict1[key].copy()
            sensors_dict1[key][:] = 999.0

            # dict2 should still have original values
            assert not np.array_equal(
                sensors_dict1[key], sensors_dict2[key]
            ), f"Sensor {key} appears to share memory"

            # Verify original is restored after reset
            sim.reset()
            sensors_dict3 = sim.sensors()
            assert np.array_equal(
                sensors_dict3[key], sensors_dict2[key]
            ), (
                f"Sensor {key} not restored to expected value after reset"
            )
            break  # Just test one to avoid excessive computation


class TestBasicFunctionality:
    """Test basic simulation functionality."""

    def test_qpos_qvel_copies(self, model_path_jetson):
        """qpos() and qvel() should return copies."""
        sim = BipedSim(model_path_jetson, control_dt=0.02)
        sim.reset()

        qpos1 = sim.qpos()
        qpos2 = sim.qpos()

        # They should be equal but not the same object
        assert np.array_equal(qpos1, qpos2)
        assert qpos1 is not qpos2

        # Modifying one should not affect the other
        qpos1[0] = 999.0
        assert qpos2[0] != 999.0

    def test_com_computation(self, model_path_jetson):
        """COM should return a 3-element array."""
        sim = BipedSim(model_path_jetson, control_dt=0.02)
        sim.reset()

        com = sim.com()

        assert com.shape == (3,), f"Expected COM shape (3,), got {com.shape}"
        assert np.all(np.isfinite(com)), "COM contains non-finite values"

    def test_contacts_returns_int(self, model_path_jetson):
        """contacts() should return an integer."""
        sim = BipedSim(model_path_jetson, control_dt=0.02)
        sim.reset()

        n_contacts = sim.contacts()

        assert isinstance(n_contacts, (int, np.integer)), (
            f"contacts() should return int, got {type(n_contacts)}"
        )

    def test_step_does_not_modify_input(self, model_path_jetson):
        """step() should not modify the input ctrl array."""
        sim = BipedSim(model_path_jetson, control_dt=0.02)
        sim.reset()

        ctrl = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ctrl_copy = ctrl.copy()

        sim.step(ctrl)

        # Input should not be modified
        assert np.array_equal(
            ctrl, ctrl_copy
        ), "step() modified the input ctrl array"


class TestSuspendedMode:
    """Test suspended mode: root body welded to world, no floating base."""

    def test_dof_reduction_jetson(self, model_path_jetson):
        """Suspended model should have nq=8, nv=8, njnt=8, nu=6."""
        # Create normal and suspended versions
        sim_normal = BipedSim(model_path_jetson, control_dt=0.02, suspended=False)
        sim_suspended = BipedSim(model_path_jetson, control_dt=0.02, suspended=True)

        # Normal model should have 15 DOF (7 floating base + 8 hinge)
        assert sim_normal.model.nq == 15, (
            f"Normal model: expected nq=15, got {sim_normal.model.nq}"
        )
        assert sim_normal.model.nv == 14, (
            f"Normal model: expected nv=14, got {sim_normal.model.nv}"
        )

        # Suspended model should have 8 DOF (only 8 hinge joints)
        assert sim_suspended.model.nq == 8, (
            f"Suspended model: expected nq=8, got {sim_suspended.model.nq}"
        )
        assert sim_suspended.model.nv == 8, (
            f"Suspended model: expected nv=8, got {sim_suspended.model.nv}"
        )
        assert sim_suspended.model.njnt == 8, (
            f"Suspended model: expected njnt=8, got {sim_suspended.model.njnt}"
        )
        assert sim_suspended.model.nu == 6, (
            f"Suspended model: expected nu=6, got {sim_suspended.model.nu}"
        )

    def test_dof_reduction_no_jetson(self, model_path_no_jetson):
        """Suspended model should have nq=8, nv=8, njnt=8, nu=6 (no_jetson variant)."""
        sim_normal = BipedSim(model_path_no_jetson, control_dt=0.02, suspended=False)
        sim_suspended = BipedSim(model_path_no_jetson, control_dt=0.02, suspended=True)

        # Normal model
        assert sim_normal.model.nq == 15
        assert sim_normal.model.nv == 14

        # Suspended model
        assert sim_suspended.model.nq == 8
        assert sim_suspended.model.nv == 8
        assert sim_suspended.model.njnt == 8
        assert sim_suspended.model.nu == 6

    def test_root_body_fixed_jetson(self, model_path_jetson):
        """Root body should remain fixed in world position after many steps."""
        sim = BipedSim(model_path_jetson, control_dt=0.02, suspended=True)
        sim.reset()

        # Record the root body position (body index 1, since 0 is world)
        root_pos_initial = sim.data.xpos[1].copy()

        # Step 50 times with zero control
        for _ in range(50):
            sim.step(np.zeros(sim.model.nu))

        # Check that position is EXACTLY unchanged
        root_pos_final = sim.data.xpos[1]
        assert np.array_equal(root_pos_initial, root_pos_final), (
            f"Root body moved! Initial {root_pos_initial}, Final {root_pos_final}"
        )

    def test_root_body_fixed_no_jetson(self, model_path_no_jetson):
        """Root body should remain fixed in world position (no_jetson variant)."""
        sim = BipedSim(model_path_no_jetson, control_dt=0.02, suspended=True)
        sim.reset()

        root_pos_initial = sim.data.xpos[1].copy()

        for _ in range(50):
            sim.step(np.zeros(sim.model.nu))

        root_pos_final = sim.data.xpos[1]
        assert np.array_equal(root_pos_initial, root_pos_final), (
            f"Root body moved! Initial {root_pos_initial}, Final {root_pos_final}"
        )

    def test_reset_works_on_suspended_jetson(self, model_path_jetson):
        """reset() should work correctly on suspended models."""
        sim = BipedSim(model_path_jetson, control_dt=0.02, suspended=True)

        # Reset should not raise
        sim.reset()

        # qpos should have length 8
        qpos = sim.qpos()
        assert len(qpos) == 8, (
            f"Expected qpos length 8, got {len(qpos)}"
        )

        # For stand keyframe, all hinge angles should be ~0
        # (the stand keyframe's last 8 values are all 0)
        assert np.allclose(qpos, 0.0, atol=1e-10), (
            f"Stand keyframe qpos not all zeros: {qpos}"
        )

    def test_reset_works_on_suspended_no_jetson(self, model_path_no_jetson):
        """reset() should work correctly on suspended models (no_jetson variant)."""
        sim = BipedSim(model_path_no_jetson, control_dt=0.02, suspended=True)

        sim.reset()

        qpos = sim.qpos()
        assert len(qpos) == 8
        assert np.allclose(qpos, 0.0, atol=1e-10), (
            f"Stand keyframe qpos not all zeros: {qpos}"
        )

    def test_no_leftover_temp_files_jetson(self, model_path_jetson):
        """Creating multiple suspended sims should not leave stray temp files."""
        models_dir = os.path.dirname(os.path.abspath(model_path_jetson))

        # Count temp files before
        temp_files_before = glob.glob(os.path.join(models_dir, '_tmp_suspended_*.xml'))

        # Create several suspended simulators
        for _ in range(5):
            sim = BipedSim(model_path_jetson, control_dt=0.02, suspended=True)
            sim.reset()

        # Count temp files after
        temp_files_after = glob.glob(os.path.join(models_dir, '_tmp_suspended_*.xml'))

        # Should be no stray files (only the ones that existed before)
        assert len(temp_files_after) == len(temp_files_before), (
            f"Found stray temp files: {set(temp_files_after) - set(temp_files_before)}"
        )

    def test_no_leftover_temp_files_no_jetson(self, model_path_no_jetson):
        """Creating multiple suspended sims should not leave stray temp files (no_jetson)."""
        models_dir = os.path.dirname(os.path.abspath(model_path_no_jetson))

        temp_files_before = glob.glob(os.path.join(models_dir, '_tmp_suspended_*.xml'))

        for _ in range(5):
            sim = BipedSim(model_path_no_jetson, control_dt=0.02, suspended=True)
            sim.reset()

        temp_files_after = glob.glob(os.path.join(models_dir, '_tmp_suspended_*.xml'))

        assert len(temp_files_after) == len(temp_files_before), (
            f"Found stray temp files: {set(temp_files_after) - set(temp_files_before)}"
        )


class TestPhysicsBaseline:
    """Integration test for physics baseline: zero-control settle over 5.0 seconds."""

    def test_settle_baseline_biped_jetson(self, model_path_jetson):
        """
        Test biped.xml settle baseline with zero control over 5.0 simulated seconds.

        This is an integration test that:
        1. Creates BipedSim with biped.xml and control_dt=0.02
        2. Calls reset() to stand keyframe
        3. Runs 250 steps (5.0 seconds) with zero control
        4. Tracks qpos/qvel finiteness, z-height min/max, and velocities every step
        5. Asserts stable physics behavior within documented bounds
        """
        # Create simulator with jetson model
        sim = BipedSim(model_path_jetson, control_dt=0.02)
        sim.reset(keyframe="stand")

        # Track metrics across all 250 steps
        n_steps = 250
        z_min = float("inf")
        z_max = float("-inf")
        max_qvel_at_step_200 = None
        max_qvel_final = None
        final_z = None
        final_contacts = None

        # Run 250 steps with zero control
        for step_idx in range(n_steps):
            # Step with zero control
            sim.step(np.zeros(sim.model.nu))

            # Get state
            qpos = sim.qpos()
            qvel = sim.qvel()

            # Check finiteness
            assert np.all(np.isfinite(qpos)), (
                f"Non-finite qpos at step {step_idx}: {qpos}"
            )
            assert np.all(np.isfinite(qvel)), (
                f"Non-finite qvel at step {step_idx}: {qvel}"
            )

            # Track z-height (qpos[2])
            z_current = qpos[2]
            z_min = min(z_min, z_current)
            z_max = max(z_max, z_current)

            # Track max velocity magnitude
            max_qvel_current = np.max(np.abs(qvel))

            # Record at step 200 (t=4.0s)
            if step_idx == 200:
                max_qvel_at_step_200 = max_qvel_current

            # Record at final step
            if step_idx == n_steps - 1:
                final_z = z_current
                max_qvel_final = max_qvel_current
                final_contacts = sim.contacts()

        # Assertions on final state
        assert max_qvel_at_step_200 is not None, "Failed to record max_qvel at step 200"
        assert max_qvel_final is not None, "Failed to record max_qvel at final step"
        assert final_z is not None, "Failed to record final z"
        assert final_contacts is not None, "Failed to record final contacts"

        # Assertion: max|qvel| at t=4.0s should be < 0.05
        assert max_qvel_at_step_200 < 0.05, (
            f"max|qvel| at step 200 (t=4.0s) should be <0.05, "
            f"got {max_qvel_at_step_200}"
        )

        # Assertion: z-range over entire trajectory
        assert z_min >= 0.0, (
            f"z should stay >= 0.0, but min was {z_min}"
        )
        assert z_max <= 0.25, (
            f"z should stay <= 0.25, but max was {z_max}"
        )

        # Assertion: final contacts should be sane (1-8)
        assert 1 <= final_contacts <= 8, (
            f"Final contact count should be 1-8, got {final_contacts}"
        )

        # Print measured values for Task 3
        print(f"\n=== Physics Baseline Measurements ===")
        print(f"Final z: {final_z:.6f}")
        print(f"Min z (over full trajectory): {z_min:.6f}")
        print(f"Max z (over full trajectory): {z_max:.6f}")
        print(f"Max|qvel| at step 200 (t=4.0s): {max_qvel_at_step_200:.6f}")
        print(f"Max|qvel| at final step (t=5.0s): {max_qvel_final:.6f}")
        print(f"Final contact count: {final_contacts}")
        print(f"====================================\n")
