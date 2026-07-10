"""
Physics validation tests for the biped robot.

This module contains tests for verifying correct physical behavior,
particularly free-fall dynamics.
"""

import numpy as np
import mujoco
import pytest


class TestFreefall:
    """Test suite for free-fall physics validation."""

    def test_freefall_biped_no_contacts(self):
        """
        Verify that a biped in free-fall exhibits correct physics.

        This test:
        - Elevates the base to z=2.0m (well above ground)
        - Zeros velocity and control
        - Steps for 150 physics steps (0.3 seconds at dt=0.002)
        - Verifies no contacts occur throughout the fall
        - Validates simulated z(t) matches analytical free-fall within 0.25% error
        """
        # Load the biped model directly
        model_path = "models/mjcf/biped.xml"
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)

        # Reset to the "stand" keyframe
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        assert key_id >= 0, "Keyframe 'stand' not found"
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Elevate the base to z=2.0m and zero velocity
        data.qpos[2] = 2.0
        data.qvel[:] = 0.0

        # Ensure control is zero (default, but being explicit)
        data.ctrl[:] = 0.0

        # Forward pass to update derived quantities
        mujoco.mj_forward(model, data)

        # Physics parameters
        dt = model.opt.timestep  # Should be 0.002
        g = 9.81
        z0 = 2.0
        num_steps = 150
        max_error = 0.0
        max_error_time = 0.0

        # Collect data from each step
        z_positions = []
        times = []
        ncons = []

        for step_idx in range(num_steps):
            # Step the physics
            mujoco.mj_step(model, data)

            # Record data
            t = data.time
            z_sim = data.qpos[2]
            ncon = data.ncon

            times.append(t)
            z_positions.append(z_sim)
            ncons.append(ncon)

            # Assert no contacts during fall
            assert ncon == 0, (
                f"Contact occurred at step {step_idx}, time {t:.4f}s. "
                f"ncon={ncon}. This indicates the elevated-drop setup failed."
            )

            # Compute analytical solution
            z_analytic = z0 - 0.5 * g * t**2

            # Compute relative error as percentage
            # Avoid division by zero by using absolute value of analytic solution
            if abs(z_analytic) > 1e-10:
                rel_error_pct = abs(z_sim - z_analytic) / abs(z_analytic) * 100
            else:
                rel_error_pct = 0.0

            # Track maximum error
            if rel_error_pct > max_error:
                max_error = rel_error_pct
                max_error_time = t

        # Assert max error is within tolerance (0.25%)
        tolerance = 0.25
        assert max_error < tolerance, (
            f"Maximum relative error {max_error:.4f}% at t={max_error_time:.4f}s "
            f"exceeds tolerance {tolerance}%"
        )

        # Print results for verification
        print(f"\n--- Free-Fall Test Results ---")
        print(f"Number of steps: {num_steps}")
        print(f"Total simulation time: {times[-1]:.4f}s")
        print(f"Physics timestep: {dt}s")
        print(f"Initial base height (z0): {z0}m")
        print(f"Final base height (z_f): {z_positions[-1]:.6f}m")
        print(f"Final analytical height: {z0 - 0.5 * g * times[-1]**2:.6f}m")
        print(f"Maximum relative error: {max_error:.6f}%")
        print(f"Time of maximum error: {max_error_time:.4f}s")
        print(f"Tolerance: {tolerance}%")
        print(f"Contact check: all {num_steps} steps had ncon=0 ✓")
        print(f"--- Test PASSED ---\n")


class TestAnklePendulum:
    """Test suite for ankle pendulum dynamics: energy conservation and oscillation period."""

    @staticmethod
    def build_ankle_pendulum_model(biped_model_path):
        """Build a standalone single-ankle-pendulum rig for energy/period validation."""
        import xml.etree.ElementTree as ET
        import os
        import copy

        # Step 1: get the REAL world pose of the ankle joint's immediate parent body ("motor")
        # at the stand keyframe, from the actual compiled biped model.
        m_full = mujoco.MjModel.from_xml_path(biped_model_path)
        d_full = mujoco.MjData(m_full)
        kid = mujoco.mj_name2id(m_full, mujoco.mjtObj.mjOBJ_KEY, "stand")
        mujoco.mj_resetDataKeyframe(m_full, d_full, kid)
        mujoco.mj_forward(m_full, d_full)
        motor_id = mujoco.mj_name2id(m_full, mujoco.mjtObj.mjOBJ_BODY, "motor")
        anchor_pos = d_full.xpos[motor_id].copy()
        anchor_quat = d_full.xquat[motor_id].copy()

        # Step 2: parse the original biped.xml, find the tibia_2 <body> element AS-IS
        tree = ET.parse(biped_model_path)
        root_elem = tree.getroot()
        wb = root_elem.find('worldbody')

        def find_body(elem, name):
            for b in elem.iter('body'):
                if b.get('name') == name:
                    return b
            return None

        tibia2 = find_body(wb, 'tibia_2')

        # Step 3: build a fresh minimal MJCF
        new_root = ET.Element('mujoco', {'model': 'ankle_pendulum'})
        new_root.append(root_elem.find('compiler'))
        new_root.append(root_elem.find('asset'))

        new_wb = ET.SubElement(new_root, 'worldbody')
        anchor = ET.SubElement(new_wb, 'body', {
            'name': 'anchor',
            'pos': f"{anchor_pos[0]} {anchor_pos[1]} {anchor_pos[2]}",
            'quat': f"{anchor_quat[0]} {anchor_quat[1]} {anchor_quat[2]} {anchor_quat[3]}",
        })
        anchor.append(copy.deepcopy(tibia2))

        xml_str = ET.tostring(new_root, encoding='unicode')
        tmp_path = os.path.join(
            os.path.dirname(os.path.abspath(biped_model_path)),
            '_tmp_ankle_pendulum_test.xml'
        )
        with open(tmp_path, 'w') as f:
            f.write(xml_str)
        try:
            m = mujoco.MjModel.from_xml_path(tmp_path)
        finally:
            os.remove(tmp_path)
        return m

    def test_ankle_pendulum_energy_and_period(self):
        """
        Test energy conservation and oscillation period of an isolated ankle pendulum.

        This test:
        - Builds a standalone pendulum rig containing only the left ankle and foot joints
        - Zeros all dissipation (damping, friction loss, armature)
        - Computes the analytical period using finite-difference gravitational stiffness
        - Simulates for 5 seconds with energy tracking enabled
        - Verifies energy conservation (drift < 2%)
        - Verifies oscillation period matches analytic prediction (within 5%)
        """
        model_path = "models/mjcf/biped.xml"
        model = self.build_ankle_pendulum_model(model_path)

        # Zero all dissipation in-memory
        model.dof_damping[:] = 0
        model.dof_frictionloss[:] = 0
        model.dof_armature[:] = 0

        # Find the ankle_l joint's qpos and qvel indices
        ankle_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_l")
        assert ankle_joint_id >= 0, "Joint 'ankle_l' not found in pendulum model"

        ankle_qpos_idx = model.jnt_qposadr[ankle_joint_id]
        ankle_qvel_idx = model.jnt_dofadr[ankle_joint_id]

        # Verify model structure
        print(f"\n--- Pendulum Model Structure ---")
        print(f"nq={model.nq}, nv={model.nv}, njnt={model.njnt}")
        print(f"ankle_l qpos index: {ankle_qpos_idx}")
        print(f"ankle_l qvel index: {ankle_qvel_idx}")

        # Create data and set initial condition
        data = mujoco.MjData(model)
        data.qpos[ankle_qpos_idx] = 0.3
        data.qvel[:] = 0

        # Forward pass
        mujoco.mj_forward(model, data)

        # ===== Compute analytic period =====
        # Compute effective inertia
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, M)
        I_eff = M[ankle_qvel_idx, ankle_qvel_idx]

        # Compute gravitational stiffness via finite differencing
        eps = 1e-3
        data_plus = mujoco.MjData(model)
        data_minus = mujoco.MjData(model)

        data_plus.qpos[ankle_qpos_idx] = eps
        data_plus.qvel[:] = 0
        mujoco.mj_forward(model, data_plus)

        data_minus.qpos[ankle_qpos_idx] = -eps
        data_minus.qvel[:] = 0
        mujoco.mj_forward(model, data_minus)

        tau_plus = data_plus.qfrc_bias[ankle_qvel_idx]
        tau_minus = data_minus.qfrc_bias[ankle_qvel_idx]
        k_est = -(tau_plus - tau_minus) / (2 * eps)

        omega = np.sqrt(abs(k_est) / I_eff)
        T_analytic = 2 * np.pi / omega

        print(f"--- Analytical Period Computation ---")
        print(f"I_eff (effective inertia): {I_eff:.6e}")
        print(f"k_est (gravitational stiffness): {k_est:.6e}")
        print(f"omega (natural frequency): {omega:.6f} rad/s")
        print(f"T_analytic: {T_analytic:.6f} s")

        # ===== Enable energy tracking and simulate =====
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
        data_sim = mujoco.MjData(model)
        data_sim.qpos[ankle_qpos_idx] = 0.3
        data_sim.qvel[:] = 0
        mujoco.mj_forward(model, data_sim)

        E0 = data_sim.energy[0] + data_sim.energy[1]

        # Simulate for 5 seconds
        num_steps = int(5.0 / model.opt.timestep)
        times = []
        ankle_angles = []
        energies = []

        for step_idx in range(num_steps):
            times.append(data_sim.time)
            ankle_angles.append(data_sim.qpos[ankle_qpos_idx])
            energies.append(data_sim.energy[0] + data_sim.energy[1])
            mujoco.mj_step(model, data_sim)

        # ===== Energy conservation assertion =====
        energies = np.array(energies)
        energy_drift_pct = (np.max(energies) - np.min(energies)) / abs(E0) * 100

        print(f"--- Energy Conservation Check ---")
        print(f"Initial energy E0: {E0:.6e} J")
        print(f"Energy min: {np.min(energies):.6e} J")
        print(f"Energy max: {np.max(energies):.6e} J")
        print(f"Energy drift: {energy_drift_pct:.4f}%")

        assert energy_drift_pct < 2.0, (
            f"Energy drift {energy_drift_pct:.4f}% exceeds tolerance 2.0%. "
            f"This indicates energy is not conserved."
        )

        # ===== Period assertion =====
        # Find local maxima (peaks) in the ankle-angle time series
        ankle_angles_arr = np.array(ankle_angles)
        peaks = []
        for i in range(1, len(ankle_angles_arr) - 1):
            if (ankle_angles_arr[i] > ankle_angles_arr[i-1] and
                ankle_angles_arr[i] > ankle_angles_arr[i+1]):
                peaks.append(i)

        assert len(peaks) >= 3, (
            f"Found only {len(peaks)} peaks in {num_steps} steps over 5 seconds. "
            f"Expected at least 3 peaks (real oscillation should occur)."
        )

        # Compute measured periods (time differences between consecutive peaks)
        peak_times = [times[i] for i in peaks]
        measured_periods = [peak_times[i+1] - peak_times[i] for i in range(len(peak_times) - 1)]
        measured_period_mean = np.mean(measured_periods)

        period_error_pct = abs(measured_period_mean - T_analytic) / T_analytic * 100

        print(f"--- Period Measurement ---")
        print(f"Number of peaks found: {len(peaks)}")
        print(f"Peak times (first 5): {peak_times[:5]}")
        print(f"Measured periods: {measured_periods}")
        print(f"Measured period (mean): {measured_period_mean:.6f} s")
        print(f"Period error: {period_error_pct:.4f}%")

        assert period_error_pct < 5.0, (
            f"Measured period {measured_period_mean:.6f}s differs from analytical "
            f"{T_analytic:.6f}s by {period_error_pct:.4f}%, exceeding tolerance 5.0%"
        )

        print(f"--- Test PASSED ---\n")


class TestInertia:
    """Test suite for inertia and mass property validation."""

    def test_body_masses_and_total(self, model_path_jetson):
        """
        Verify that all body masses are positive and the total mass is correct.

        Checks:
        - All bodies (except body 0, the implicit world body) have mass > 0,
          unless they are floating bases (bodies with freejoint)
        - Total mass of the biped is approximately 0.784 kg (tolerance ±0.001)
        """
        model = mujoco.MjModel.from_xml_path(model_path_jetson)

        # Expected total mass
        expected_total_mass = 0.784
        tolerance = 0.001

        # Check individual body masses
        for body_id in range(1, model.nbody):  # Skip body 0 (world)
            body_mass = model.body_mass[body_id]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            # Check if this body has a freejoint (floating base)
            jntadr = model.body_jntadr[body_id]
            jntnum = model.body_jntnum[body_id]
            has_freejoint = False
            if jntnum > 0:
                first_joint_id = jntadr
                if model.jnt_type[first_joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    has_freejoint = True

            # Floating bases (bodies with freejoint) may have 0 mass
            # All other bodies must have positive mass
            if not has_freejoint:
                assert body_mass > 0, (
                    f"Body '{body_name}' (id {body_id}) has invalid mass: {body_mass} kg. "
                    f"All non-floating bodies must have positive mass."
                )

        # Check total mass
        total_mass = np.sum(model.body_mass)
        assert abs(total_mass - expected_total_mass) <= tolerance, (
            f"Total mass {total_mass:.6f} kg differs from expected {expected_total_mass:.6f} kg "
            f"by {abs(total_mass - expected_total_mass):.6f} kg (tolerance: {tolerance} kg)"
        )

    def test_principal_inertia_triangle_inequality(self, model_path_jetson):
        """
        Verify that the principal inertia tensors satisfy the triangle inequality.

        For each body with mass > 1e-6, checks that the principal moments of inertia
        (Ixx, Iyy, Izz) satisfy: Ixx + Iyy >= Izz, Iyy + Izz >= Ixx, Izz + Ixx >= Iyy.
        """
        model = mujoco.MjModel.from_xml_path(model_path_jetson)

        # Numerical tolerance for floating-point comparisons
        tolerance = 1e-9

        for body_id in range(1, model.nbody):
            body_mass = model.body_mass[body_id]

            # Skip effectively-massless bodies
            if body_mass <= 1e-6:
                continue

            inertia = model.body_inertia[body_id]  # [Ixx, Iyy, Izz]
            Ixx, Iyy, Izz = inertia[0], inertia[1], inertia[2]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            # Check triangle inequality: sum of any two >= third
            assert Ixx + Iyy >= Izz - tolerance, (
                f"Body '{body_name}' (id {body_id}) violates triangle inequality: "
                f"Ixx + Iyy < Izz (Ixx={Ixx:.2e}, Iyy={Iyy:.2e}, Izz={Izz:.2e})"
            )
            assert Iyy + Izz >= Ixx - tolerance, (
                f"Body '{body_name}' (id {body_id}) violates triangle inequality: "
                f"Iyy + Izz < Ixx (Ixx={Ixx:.2e}, Iyy={Iyy:.2e}, Izz={Izz:.2e})"
            )
            assert Izz + Ixx >= Iyy - tolerance, (
                f"Body '{body_name}' (id {body_id}) violates triangle inequality: "
                f"Izz + Ixx < Iyy (Ixx={Ixx:.2e}, Iyy={Iyy:.2e}, Izz={Izz:.2e})"
            )

    def test_mass_matrix_positive_definite_at_stand(self, model_path_jetson):
        """
        Verify that the mass matrix is symmetric and positive-definite at the stand pose.

        Steps:
        - Load model and reset to 'stand' keyframe
        - Compute full mass matrix using mj_fullM
        - Assert symmetry and positive-definiteness
        """
        model = mujoco.MjModel.from_xml_path(model_path_jetson)
        data = mujoco.MjData(model)

        # Reset to stand keyframe
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        assert key_id >= 0, "Keyframe 'stand' not found"
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Forward pass to update derived quantities
        mujoco.mj_forward(model, data)

        # Compute full mass matrix
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, M)

        # Check symmetry
        assert np.allclose(M, M.T), (
            f"Mass matrix is not symmetric. "
            f"Max asymmetry: {np.max(np.abs(M - M.T)):.2e}"
        )

        # Check positive-definiteness via eigenvalues
        eigvals = np.linalg.eigvalsh(M)
        min_eigval = eigvals.min()
        assert min_eigval > 0, (
            f"Mass matrix is not positive-definite. "
            f"Minimum eigenvalue: {min_eigval:.2e} (should be > 0)"
        )


class TestContactQuality:
    """Test suite for contact quality: penetration depth, force balance, and stability."""

    def test_contact_quality_at_stand(self):
        """
        Test contact quality metrics for the biped standing on the floor.

        This test:
        - Loads biped.xml and resets to the 'stand' keyframe
        - Calls mj_forward to update derived quantities
        - Steps 1000 times to allow the robot to settle onto the floor
        - Validates penetration depth (abs(dist) < 2mm for each contact)
        - Validates force balance (sum of normal forces within 2% of expected weight)
        - Validates contact count stability over 500 additional steps (no wild chattering)
        """
        # Load the biped model
        model_path = "models/mjcf/biped.xml"
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)

        # Reset to the "stand" keyframe
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        assert key_id >= 0, "Keyframe 'stand' not found"
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Forward pass to update derived quantities
        mujoco.mj_forward(model, data)

        # ===== Step 1000 times to allow settling =====
        num_settle_steps = 1000
        for step_idx in range(num_settle_steps):
            mujoco.mj_step(model, data)

        print(f"\n--- Contact Quality Test (Biped at Stand) ---")
        print(f"After {num_settle_steps} settle steps:")
        print(f"Time: {data.time:.4f}s")
        print(f"Base position (z): {data.qpos[2]:.6f}m")
        print(f"Number of active contacts: {data.ncon}")

        # ===== Penetration depth check =====
        print(f"\n--- Penetration Depth Check ---")
        penetration_depths = []
        for i in range(data.ncon):
            dist = data.contact[i].dist
            penetration_depths.append(dist)
            contact_name_1 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1
            )
            contact_name_2 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2
            )
            print(f"Contact {i}: {contact_name_1} <-> {contact_name_2}: dist={dist:.6f}m")

            # Assert penetration within tolerance
            assert abs(dist) < 0.002, (
                f"Contact {i} penetration depth {abs(dist):.6f}m exceeds tolerance 0.002m"
            )

        print(f"Max penetration depth: {max(abs(d) for d in penetration_depths):.6f}m")
        print(f"All penetration depths within 2mm tolerance ✓")

        # ===== Force balance check =====
        print(f"\n--- Force Balance Check ---")

        # Compute total mass and expected weight
        total_mass = np.sum(model.body_mass)
        expected_weight = total_mass * 9.81

        print(f"Total mass: {total_mass:.6f} kg")
        print(f"Expected weight: {expected_weight:.6f} N")

        # Compute normal force sum
        normal_force_sum = 0.0
        for i in range(data.ncon):
            force6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force6)
            normal_force = force6[0]
            normal_force_sum += normal_force
            print(f"Contact {i} normal force: {normal_force:.6f} N")

        print(f"Sum of normal forces: {normal_force_sum:.6f} N")

        # Compute force balance error
        force_error = abs(normal_force_sum - expected_weight) / expected_weight
        force_error_pct = force_error * 100

        print(f"Force balance error: {force_error_pct:.4f}%")
        print(f"Tolerance: 2.0%")

        assert force_error_pct < 2.0, (
            f"Force balance error {force_error_pct:.4f}% exceeds tolerance 2.0%. "
            f"Sum of normal forces: {normal_force_sum:.6f}N, "
            f"expected weight: {expected_weight:.6f}N"
        )

        print(f"Force balance within 2% tolerance ✓")

        # ===== Contact count stability check =====
        print(f"\n--- Contact Count Stability Check ---")
        print(f"Stepping for 1.0 second ({int(1.0 / model.opt.timestep)} steps) to check stability...")

        # Step for 1.0 simulated second at dt=0.002s (500 steps)
        ncon_values = []
        num_stability_steps = int(1.0 / model.opt.timestep)

        for step_idx in range(num_stability_steps):
            mujoco.mj_step(model, data)
            ncon_values.append(data.ncon)

        # Analyze contact count
        unique_ncon = sorted(set(ncon_values))
        ncon_min = min(ncon_values)
        ncon_max = max(ncon_values)
        ncon_range = ncon_max - ncon_min

        print(f"Contact count over 1.0 second:")
        print(f"  Min: {ncon_min}, Max: {ncon_max}, Range: {ncon_range}")
        print(f"  Unique values: {unique_ncon}")
        print(f"  Number of distinct values: {len(unique_ncon)}")

        # Assert contact count doesn't chatter wildly
        # Allow up to 2 distinct values (acceptable for settling), or range <= 1
        assert len(unique_ncon) <= 2, (
            f"Contact count is chattering: {len(unique_ncon)} distinct values observed. "
            f"Expected <= 2 distinct values (e.g., transition from 4 to 4, or 4 to 5). "
            f"Unique values: {unique_ncon}"
        )

        print(f"Contact count stable (≤ 2 distinct values) ✓")
        print(f"--- Test PASSED ---\n")


class TestFrictionBreakaway:
    """Test suite for Coulomb friction breakaway behavior."""

    @staticmethod
    def build_foot_slider_model(biped_model_path):
        """Isolate the foot_1 body (with its real mesh/mass/friction) as a free-sliding
        block on a copy of the real floor plane, for a clean single-body friction test."""
        import xml.etree.ElementTree as ET
        import os
        import copy

        tree = ET.parse(biped_model_path)
        root_elem = tree.getroot()
        wb = root_elem.find('worldbody')

        def find_body(elem, name):
            for b in elem.iter('body'):
                if b.get('name') == name:
                    return b
            return None

        foot = find_body(wb, 'foot_1')
        floor = None
        for g in wb.findall('geom'):
            if g.get('name') == 'floor':
                floor = g

        new_root = ET.Element('mujoco', {'model': 'foot_slide_test'})
        new_root.append(root_elem.find('compiler'))
        new_root.append(root_elem.find('asset'))
        opt = ET.SubElement(new_root, 'option')
        orig_opt = root_elem.find('option')
        opt.set('timestep', orig_opt.get('timestep'))
        opt.set('integrator', orig_opt.get('integrator'))

        new_wb = ET.SubElement(new_root, 'worldbody')
        new_wb.append(copy.deepcopy(floor))

        foot_copy = copy.deepcopy(foot)
        foot_copy.set('pos', '0 0 0.02')  # small clearance above floor, let it settle
        foot_copy.set('quat', '1 0 0 0')
        # Remove the foot_l joint that foot_1 normally has, replace with a freejoint
        # so it's a free-sliding rigid body, not an articulated joint.
        j = foot_copy.find('joint')
        if j is not None:
            foot_copy.remove(j)
        fj = ET.Element('freejoint', {'name': 'foot_free'})
        foot_copy.insert(0, fj)
        new_wb.append(foot_copy)

        xml_str = ET.tostring(new_root, encoding='unicode')
        tmp_path = os.path.join(
            os.path.dirname(os.path.abspath(biped_model_path)),
            '_tmp_foot_slide_test.xml'
        )
        with open(tmp_path, 'w') as f:
            f.write(xml_str)
        try:
            m = mujoco.MjModel.from_xml_path(tmp_path)
        finally:
            os.remove(tmp_path)
        return m

    def measure_drift_at_force(self, model, force_magnitude):
        """
        Measure the drift distance of the foot when a horizontal force is applied.

        Steps:
        1. Create a fresh MjData
        2. Call mj_forward, then step 500 times to settle
        3. Record x0 = data.qpos[0] (foot's x-position)
        4. Apply force F to body index 1 (the foot)
        5. Step for 1.0 simulated second
        6. Record x1 = data.qpos[0]
        7. Return |x1 - x0| as the drift distance
        """
        data = mujoco.MjData(model)

        # Forward pass and settle
        mujoco.mj_forward(model, data)
        for _ in range(500):
            mujoco.mj_step(model, data)

        # Record initial position
        x0 = data.qpos[0]

        # Apply horizontal force to foot body (body index 1)
        data.xfrc_applied[1, 0] = force_magnitude

        # Step for 1.0 simulated second
        num_steps = int(1.0 / model.opt.timestep)
        for _ in range(num_steps):
            mujoco.mj_step(model, data)

        # Record final position
        x1 = data.qpos[0]

        # Return absolute drift distance
        return abs(x1 - x0)

    def test_friction_breakaway_signature(self):
        """
        Test the Coulomb friction breakaway behavior using an isolated foot-slider rig.

        This test:
        - Builds a standalone foot-slider model (foot_1 body as a free-sliding rigid body on floor)
        - Computes the normal force N = m * g and friction threshold mu * N
        - Measures drift at three force levels: 0.8x, 1.0x, 1.2x the friction threshold
        - Verifies a clear breakaway signature: drift increases monotonically and drift at 1.2x
          is at least 5x the drift at 0.8x (conservative floor for real Coulomb transition)
        """
        model_path = "models/mjcf/biped.xml"
        model = self.build_foot_slider_model(model_path)

        # Verify model structure
        print(f"\n--- Friction Breakaway Test (Foot-Slider Rig) ---")
        print(f"Model structure: nq={model.nq}, nv={model.nv}, nbody={model.nbody}")
        assert model.nbody == 2, "Expected 2 bodies (world + foot)"

        # Compute normal force and friction threshold
        # Body index 1 is the only non-world body (the foot)
        foot_mass = model.body_mass[1]
        N = foot_mass * 9.81
        mu = 1.0  # Friction coefficient from geom spec (1.0 0.02 0.001)
        friction_threshold = mu * N

        print(f"Foot mass: {foot_mass:.6f} kg")
        print(f"Normal force N: {N:.6f} N")
        print(f"Friction coefficient mu: {mu}")
        print(f"Friction threshold (mu*N): {friction_threshold:.6f} N")

        # Measure drift at three force levels
        force_0_8x = 0.8 * friction_threshold
        force_1_0x = 1.0 * friction_threshold
        force_1_2x = 1.2 * friction_threshold

        print(f"\n--- Measuring Drift at Force Levels ---")
        print(f"Force at 0.8x: {force_0_8x:.6f} N")
        drift_0_8x = self.measure_drift_at_force(model, force_0_8x)
        print(f"  Drift at 0.8x: {drift_0_8x:.6f} m ({drift_0_8x*1000:.2f} mm)")

        print(f"Force at 1.0x: {force_1_0x:.6f} N")
        drift_1_0x = self.measure_drift_at_force(model, force_1_0x)
        print(f"  Drift at 1.0x: {drift_1_0x:.6f} m ({drift_1_0x*1000:.2f} mm)")

        print(f"Force at 1.2x: {force_1_2x:.6f} N")
        drift_1_2x = self.measure_drift_at_force(model, force_1_2x)
        print(f"  Drift at 1.2x: {drift_1_2x:.6f} m ({drift_1_2x*1000:.2f} mm)")

        # Compute ratio
        ratio = drift_1_2x / drift_0_8x if drift_0_8x > 1e-10 else float('inf')
        print(f"\nRatio (drift_1.2x / drift_0.8x): {ratio:.2f}x")

        # Assertion 1: Monotonically increasing with force
        print(f"\n--- Monotonicity Check ---")
        assert drift_0_8x < drift_1_0x, (
            f"Drift not increasing from 0.8x to 1.0x: "
            f"drift_0.8x={drift_0_8x:.6f} m, drift_1.0x={drift_1_0x:.6f} m"
        )
        print(f"drift_0.8x < drift_1.0x ✓")

        assert drift_1_0x < drift_1_2x, (
            f"Drift not increasing from 1.0x to 1.2x: "
            f"drift_1.0x={drift_1_0x:.6f} m, drift_1.2x={drift_1_2x:.6f} m"
        )
        print(f"drift_1.0x < drift_1.2x ✓")

        # Assertion 2: Clear breakaway signature (5x ratio)
        print(f"\n--- Breakaway Signature Check ---")
        assert drift_1_2x >= 5 * drift_0_8x, (
            f"Breakaway signature insufficient: drift_1.2x ({drift_1_2x:.6f} m) "
            f"is only {ratio:.2f}x drift_0.8x ({drift_0_8x:.6f} m). "
            f"Expected at least 5x ratio (measured {ratio:.2f}x). "
            f"This indicates the friction model may not be functioning correctly."
        )
        print(f"drift_1.2x >= 5 * drift_0.8x ✓")

        print(f"\n--- Test PASSED ---\n")


class TestTimestepSensitivity:
    """Test suite for timestep sensitivity in physics integration."""

    def test_timestep_sensitivity_settled_height(self):
        """
        Test that settled base height is insensitive to timestep across dt values.

        This test:
        - Loads the biped model once via from_xml_path
        - For each dt in [0.002, 0.001, 0.0005]:
          - Sets model.opt.timestep = dt in-memory
          - Creates a fresh MjData, resets to 'stand' keyframe
          - Steps for 2.0 simulated seconds
          - Records settled base z-height (data.qpos[2])
          - Asserts no NaN/Inf occurred
        - Verifies all three settled z-values agree within 10% of each other
        - Prints all three measured z-values for the report
        """
        model_path = "models/mjcf/biped.xml"
        model = mujoco.MjModel.from_xml_path(model_path)

        timesteps = [0.002, 0.001, 0.0005]
        settled_heights = []

        print(f"\n--- Timestep Sensitivity Test (Settled Height) ---")
        print(f"Testing timesteps: {timesteps}")
        print(f"Settling duration: 2.0 seconds per timestep\n")

        for dt in timesteps:
            # Set timestep in-memory
            model.opt.timestep = dt

            # Create fresh data and reset to stand keyframe
            data = mujoco.MjData(model)
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
            assert key_id >= 0, "Keyframe 'stand' not found"
            mujoco.mj_resetDataKeyframe(model, data, key_id)

            # Forward pass
            mujoco.mj_forward(model, data)

            # Step for 2.0 simulated seconds
            num_steps = int(2.0 / dt)
            for step_idx in range(num_steps):
                mujoco.mj_step(model, data)

            # Check for NaN/Inf
            if not np.all(np.isfinite(data.qpos)):
                print(f"ERROR: NaN/Inf detected at dt={dt}s")
                print(f"qpos finite check failed")
                print(f"VERDICT: FAIL")
                raise AssertionError(
                    f"Detected NaN/Inf in qpos at dt={dt}s. "
                    f"Simulation numerics degraded at this timestep."
                )

            # Record settled height
            z_settled = data.qpos[2]
            settled_heights.append(z_settled)

            print(f"dt = {dt}s ({num_steps} steps): z_settled = {z_settled:.8f} m")

        # Verify all three values agree within 10%
        print(f"\n--- Height Agreement Check ---")
        z_0_002 = settled_heights[0]
        z_0_001 = settled_heights[1]
        z_0_0005 = settled_heights[2]

        # Pairwise relative error check
        def rel_error_pct(z_i, z_j):
            max_abs = max(abs(z_i), abs(z_j))
            if max_abs < 1e-10:
                return 0.0
            return abs(z_i - z_j) / max_abs * 100

        error_0_002_vs_0_001 = rel_error_pct(z_0_002, z_0_001)
        error_0_002_vs_0_0005 = rel_error_pct(z_0_002, z_0_0005)
        error_0_001_vs_0_0005 = rel_error_pct(z_0_001, z_0_0005)

        tolerance_pct = 10.0

        print(f"z(dt=0.002) = {z_0_002:.8f} m")
        print(f"z(dt=0.001) = {z_0_001:.8f} m")
        print(f"z(dt=0.0005) = {z_0_0005:.8f} m")
        print(f"\nRelative errors:")
        print(f"  |z_0.002 - z_0.001| / max = {error_0_002_vs_0_001:.6f}%")
        print(f"  |z_0.002 - z_0.0005| / max = {error_0_002_vs_0_0005:.6f}%")
        print(f"  |z_0.001 - z_0.0005| / max = {error_0_001_vs_0_0005:.6f}%")
        print(f"Tolerance: {tolerance_pct}%")

        assert error_0_002_vs_0_001 < tolerance_pct, (
            f"dt=0.002 and dt=0.001 settled heights disagree by {error_0_002_vs_0_001:.6f}%, "
            f"exceeding tolerance {tolerance_pct}%. "
            f"Values: {z_0_002:.8f}m vs {z_0_001:.8f}m"
        )
        assert error_0_002_vs_0_0005 < tolerance_pct, (
            f"dt=0.002 and dt=0.0005 settled heights disagree by {error_0_002_vs_0_0005:.6f}%, "
            f"exceeding tolerance {tolerance_pct}%. "
            f"Values: {z_0_002:.8f}m vs {z_0_0005:.8f}m"
        )
        assert error_0_001_vs_0_0005 < tolerance_pct, (
            f"dt=0.001 and dt=0.0005 settled heights disagree by {error_0_001_vs_0_0005:.6f}%, "
            f"exceeding tolerance {tolerance_pct}%. "
            f"Values: {z_0_001:.8f}m vs {z_0_0005:.8f}m"
        )

        print(f"\nAll pairwise comparisons within {tolerance_pct}% tolerance ✓")
        print(f"--- Test PASSED ---\n")


class TestLongHorizonSanity:
    """Test suite for long-horizon stability and energy behavior."""

    def test_30_second_passive_settle(self):
        """
        Test 30-second passive settle with energy tracking and stability checks.

        This test:
        - Loads biped.xml and enables energy tracking
        - Resets to the 'stand' keyframe
        - Steps for 30.0 simulated seconds (15000 steps at dt=0.002)
        - Records energy at every step and checks for NaN/Inf immediately (fail on first occurrence)
        - Verifies max single-step energy increase < 0.01 (empirically ~0.0026 is normal)
        - Verifies windowed-max energy envelope is non-increasing (60 windows of 250 steps each)
        - Prints energy values at key timepoints: t=0, t=1s, t=15s, t=30s
        """
        # Load model and enable energy tracking
        model_path = "models/mjcf/biped.xml"
        model = mujoco.MjModel.from_xml_path(model_path)
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY

        # Create data and reset to stand keyframe
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        assert key_id >= 0, "Keyframe 'stand' not found"
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Forward pass to initialize derived quantities
        mujoco.mj_forward(model, data)

        # Physics parameters
        dt = model.opt.timestep  # Expected to be 0.002
        num_steps = int(30.0 / dt)  # 15000 steps for 30 seconds

        # Arrays to record data
        times = []
        qpos_z = []
        energies = []

        print(f"\n--- Long-Horizon Sanity Test (30-second passive settle) ---")
        print(f"Timestep: {dt}s")
        print(f"Total steps: {num_steps}")
        print(f"Total simulation time: 30.0s")
        print(f"Energy tracking enabled: True\n")

        # Main simulation loop with per-step NaN/Inf checking
        for step_idx in range(num_steps):
            # Record data BEFORE stepping
            times.append(data.time)
            qpos_z.append(data.qpos[2])
            energies.append(data.energy[0] + data.energy[1])

            # Step physics
            mujoco.mj_step(model, data)

            # Immediate NaN/Inf check (fail on first occurrence)
            if not np.all(np.isfinite(data.qpos)):
                print(f"\nERROR: Non-finite qpos detected at step {step_idx}, time {data.time:.4f}s")
                print(f"qpos: {data.qpos}")
                print(f"VERDICT: FAIL")
                raise AssertionError(
                    f"Non-finite qpos at step {step_idx}, time {data.time:.4f}s. "
                    f"Simulation numerics have diverged."
                )

            if not np.all(np.isfinite(data.qvel)):
                print(f"\nERROR: Non-finite qvel detected at step {step_idx}, time {data.time:.4f}s")
                print(f"qvel: {data.qvel}")
                print(f"VERDICT: FAIL")
                raise AssertionError(
                    f"Non-finite qvel at step {step_idx}, time {data.time:.4f}s. "
                    f"Simulation numerics have diverged."
                )

        # Record final energy
        times.append(data.time)
        qpos_z.append(data.qpos[2])
        energies.append(data.energy[0] + data.energy[1])

        # Convert to numpy arrays
        times = np.array(times)
        energies = np.array(energies)

        # ===== Energy increase check =====
        print(f"--- Single-Step Energy Increase Check ---")
        energy_diffs = np.diff(energies)
        max_energy_increase = np.max(energy_diffs)

        print(f"Max single-step energy increase: {max_energy_increase:.6e}")
        print(f"Tolerance: 0.01")

        assert max_energy_increase < 0.01, (
            f"Max single-step energy increase {max_energy_increase:.6e} exceeds tolerance 0.01. "
            f"This may indicate numerical instability."
        )
        print(f"Single-step energy increase within tolerance ✓\n")

        # ===== Windowed-max envelope check =====
        print(f"--- Windowed-Max Energy Envelope Check ---")
        window_size = 250  # 0.5 seconds at dt=0.002
        num_windows = len(energies) // window_size
        window_maxes = []

        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_max = np.max(energies[start_idx:end_idx])
            window_maxes.append(window_max)

        print(f"Number of windows: {num_windows} (window size: {window_size} steps = 0.5s)")

        # Check that each window's max is <= previous window's max + 1e-3 tolerance
        envelope_ok = True
        for i in range(len(window_maxes) - 1):
            if window_maxes[i+1] > window_maxes[i] + 1e-3:
                print(f"Envelope violation: window {i+1} max {window_maxes[i+1]:.6e} > "
                      f"window {i} max {window_maxes[i]:.6e} + 1e-3")
                envelope_ok = False

        assert envelope_ok, (
            f"Windowed-max energy envelope is not non-increasing. "
            f"This indicates energy is not properly dissipating."
        )
        print(f"Windowed-max envelope is non-increasing ✓\n")

        # ===== Key timepoint energy values =====
        print(f"--- Energy at Key Timepoints ---")

        # Find indices closest to t=0, t=1, t=15, t=30
        key_times = [0.0, 1.0, 15.0, 30.0]
        for t_target in key_times:
            idx = np.argmin(np.abs(times - t_target))
            t_actual = times[idx]
            e_actual = energies[idx]
            print(f"t ≈ {t_target:5.1f}s (actual t={t_actual:.4f}s): energy = {e_actual:.6e} J")

        print(f"\n--- Test PASSED ---\n")
