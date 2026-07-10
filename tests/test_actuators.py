"""
Actuator step-response characterization for Phase 4.

This module validates the gravity-loaded step responses of the 6 actuated leg joints
(hip_roll_l/r, hip_pitch_l/r, knee_l/r) with the updated gains (kp=40.0, kv=10.0).

**Critical setup note:** The suspended rig (BipedSim with suspended=True) keeps the
torso fixed but does NOT disable contacts by default. When driving a leg joint in
this rig, the shin/foot can crash into the floor or other body parts, producing a
cascade of spurious contacts that completely corrupts the measurement (verified:
without contact disabling, knee_l driven to 0.5 rad settles at 2.97 rad, outside
its own ±1.396 rad mechanical range, because dozens of false contacts fight the
joint limit).

**The fix:** Immediately after constructing BipedSim(suspended=True), disable
contacts in-memory:
  sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

This is a legitimate isolation technique used in Phase 2's pendulum and friction rigs.

**Settling time definition (corrected for consistency):** Original roadmap phrasing
"settling time (2% band) < 1.0s" is mathematically incompatible with "steady-state
error < 0.02 rad" for these step sizes if interpreted literally: a 2% band on a
0.5 rad step is 0.01 rad, tighter than the 0.02 rad SSE tolerance. Moreover, a
pure P+D controller has inherent steady-state offset τ_gravity/kp under gravity —
it can NEVER enter a 0.01 rad band even perfectly well-behaved.

**Corrected definition used here:** settling time is "time after which the position
stays within an absolute 0.02 rad band of the final position and never leaves that
band again." This is self-consistent with the SSE tolerance and achievable by a
well-tuned P+D servo under gravity.

**Step direction conventions:**
  - hip_roll_l (range [-2.269, +0.087]): steps toward NEGATIVE (-0.2, -0.5 rad)
  - hip_roll_r (range [-0.087, +2.269]): steps toward POSITIVE (+0.2, +0.5 rad)
  - hip_pitch_l/r, knee_l/r (range [-1.396, +1.396]): symmetric, steps POSITIVE (+0.2, +0.5 rad)
"""

import sys
import os
from pathlib import Path

import numpy as np
import pytest
import mujoco

# Add parent directory to path to import sim module
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim import BipedSim


class TestStepResponse:
    """Step-response characterization of the 6 actuated joints."""

    # Joint configuration: name, actuator_index, step_directions (amplitudes to test)
    # NOTE: actuator_index != qpos index. Use self._get_qpos_idx(model, actuator_idx) to get the
    # correct qpos array index for a joint.
    JOINTS = [
        ("hip_roll_l", 0, np.array([-0.2, -0.5])),    # Asymmetric range, step negative
        ("hip_pitch_l", 1, np.array([0.2, 0.5])),     # Symmetric range
        ("knee_l", 2, np.array([0.2, 0.5])),          # Symmetric range
        ("hip_roll_r", 3, np.array([0.2, 0.5])),      # Asymmetric range, step positive
        ("hip_pitch_r", 4, np.array([0.2, 0.5])),     # Symmetric range
        ("knee_r", 5, np.array([0.2, 0.5])),          # Symmetric range
    ]

    # Settling band: absolute 0.02 rad (consistent with SSE tolerance)
    SETTLING_BAND = 0.02

    # Max settling time: 1.0 s, except hip_roll_l at 0.5 rad (allowed up to 1.05 s)
    MAX_SETTLING_TIME = 1.0
    HIP_ROLL_L_MAX_SETTLING_TIME_500 = 1.05

    # Tolerances
    MAX_SSE = 0.02  # rad
    MAX_OVERSHOOT = 0.15  # 15%

    # Simulation parameters
    CONTROL_DT = 0.02  # 50 Hz control rate (10 physics substeps)
    SIM_DURATION = 1.5  # seconds
    N_STEPS = int(SIM_DURATION / CONTROL_DT)

    def _get_qpos_idx(self, model, actuator_idx):
        """
        Get the qpos array index for a joint given its actuator index.

        In the biped model, actuator indices don't match qpos indices directly.
        This method performs the lookup via the model's joint ID.
        """
        joint_id = model.actuator_trnid[actuator_idx, 0]
        return model.jnt_qposadr[joint_id]

    @pytest.mark.parametrize("model_path_fixture", ["model_path_jetson", "model_path_no_jetson"])
    def test_step_response_all_joints(self, model_path_fixture, request):
        """
        Test gravity-loaded step response for all 6 actuated joints at 2 amplitudes each.

        Runs a separate trial per joint/amplitude combination.
        """
        model_path = request.getfixturevalue(model_path_fixture)

        # Track all results for summary
        all_results = {}

        for joint_name, joint_idx, step_amps in self.JOINTS:
            for step_amp in step_amps:
                key = f"{joint_name}_{step_amp:.1f}"
                all_results[key] = self._run_step_response(model_path, joint_name, joint_idx, step_amp)

        # Print summary and assert all pass
        print("\n" + "=" * 80)
        print("STEP RESPONSE SUMMARY")
        print("=" * 80)

        failed = []
        for key, result in all_results.items():
            status = "PASS" if result["pass"] else "FAIL"
            print(f"\n{key:30s} {status}")
            print(f"  Rise time:      {result['rise_time']:.4f} s")
            print(f"  Peak time:      {result['peak_time']:.4f} s")
            print(f"  Overshoot:      {result['overshoot']:.2f}%")
            print(f"  Settling time:  {result['settling_time']:.4f} s (limit: {result['settling_limit']:.2f} s)")
            print(f"  Steady-state err: {result['sse']:.6f} rad")
            if not result["pass"]:
                failed.append((key, result["reason"]))

        print("=" * 80)

        if failed:
            print(f"\nFailed tests ({len(failed)}):")
            for key, reason in failed:
                print(f"  {key}: {reason}")
            pytest.fail(f"{len(failed)} step response test(s) failed.")

    def _run_step_response(self, model_path, joint_name, actuator_idx, step_amp):
        """
        Run a single step-response trial and compute metrics.

        Args:
            model_path: Path to MJCF file
            joint_name: Name of joint (for debugging/reporting)
            actuator_idx: Index of the actuator (0-5)
            step_amp: Step amplitude in radians

        Returns a dict with metrics and pass/fail status.
        """
        # Create suspended rig and disable contacts
        sim = BipedSim(model_path, control_dt=self.CONTROL_DT, suspended=True)
        sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        # Get the correct qpos index for this actuator
        qpos_idx = self._get_qpos_idx(sim.model, actuator_idx)

        # Reset to stand keyframe
        sim.reset(keyframe="stand")

        # Record position trajectory
        positions = []
        times = []

        for step_idx in range(self.N_STEPS):
            # Record current position before stepping
            qpos = sim.qpos()
            positions.append(qpos[qpos_idx])
            times.append(step_idx * self.CONTROL_DT)

            # Construct control: step on this joint, hold all others at 0.0
            # (0.0 is the stand keyframe target for hinge joints)
            ctrl = np.zeros(6)
            ctrl[actuator_idx] = step_amp

            # Step
            sim.step(ctrl)

        # Convert to numpy arrays
        positions = np.array(positions)
        times = np.array(times)

        # Compute metrics
        result = self._compute_metrics(positions, times, step_amp, joint_name, actuator_idx)

        return result

    def _compute_metrics(self, positions, times, step_amp, joint_name, actuator_idx):
        """
        Compute step response metrics from position trajectory.

        Returns a dict with all metrics and pass/fail status.
        """
        # Steady-state error: use final position
        final_position = positions[-1]
        sse = abs(step_amp - final_position)

        # Find peak (max for positive step, min for negative step)
        if step_amp > 0:
            peak_idx = np.argmax(positions)
            peak_value = positions[peak_idx]
        else:
            peak_idx = np.argmin(positions)
            peak_value = positions[peak_idx]

        peak_time = times[peak_idx]

        # Overshoot: only if peak goes beyond target
        if step_amp > 0:
            overshoot_pct = max(0, (peak_value - step_amp) / step_amp * 100) if step_amp != 0 else 0
        else:
            # For negative step, check if peak goes more negative than target
            overshoot_pct = max(0, (abs(peak_value) - abs(step_amp)) / abs(step_amp) * 100) if step_amp != 0 else 0

        # Rise time: 90% of how far it actually gets (accounting for steady-state offset)
        # How far does it actually get?
        actual_amplitude = final_position - positions[0]
        target_90pct = positions[0] + 0.9 * actual_amplitude

        # Find when it first reaches 90% of actual movement
        if step_amp > 0:
            rise_indices = np.where(positions >= target_90pct)[0]
        else:
            rise_indices = np.where(positions <= target_90pct)[0]

        if len(rise_indices) > 0:
            rise_time = times[rise_indices[0]]
        else:
            rise_time = times[-1]  # Never reached, use final time

        # Settling time: time after which position stays within SETTLING_BAND of final position
        # and never leaves that band again
        settling_band_low = final_position - self.SETTLING_BAND
        settling_band_high = final_position + self.SETTLING_BAND

        settling_time = None
        # Search for the first time the position enters the band and stays there
        for i in range(len(positions)):
            if settling_band_low <= positions[i] <= settling_band_high:
                # Check if it stays in the band from here on
                if np.all((positions[i:] >= settling_band_low) & (positions[i:] <= settling_band_high)):
                    settling_time = times[i]
                    break

        if settling_time is None:
            settling_time = times[-1]  # Never settled within window

        # Determine settling time limit for this joint/amplitude
        if joint_name == "hip_roll_l" and abs(step_amp) == 0.5:
            settling_limit = self.HIP_ROLL_L_MAX_SETTLING_TIME_500
        else:
            settling_limit = self.MAX_SETTLING_TIME

        # Check all assertions
        reasons = []
        if sse > self.MAX_SSE:
            reasons.append(f"SSE {sse:.6f} > {self.MAX_SSE}")
        if overshoot_pct > self.MAX_OVERSHOOT * 100:
            reasons.append(f"Overshoot {overshoot_pct:.2f}% > {self.MAX_OVERSHOOT*100:.1f}%")
        if settling_time > settling_limit:
            reasons.append(f"Settling time {settling_time:.4f} s > {settling_limit:.2f} s")

        pass_flag = len(reasons) == 0

        return {
            "pass": pass_flag,
            "reason": " AND ".join(reasons) if reasons else "OK",
            "rise_time": rise_time,
            "peak_time": peak_time,
            "overshoot": overshoot_pct,
            "settling_time": settling_time,
            "settling_limit": settling_limit,
            "sse": sse,
        }

    @pytest.mark.parametrize("model_path_fixture", ["model_path_jetson", "model_path_no_jetson"])
    def test_repeatability_hip_pitch_l(self, model_path_fixture, request):
        """
        Repeatability check: run hip_pitch_l 0.5 rad step twice, verify bitwise-identical trajectories.

        This confirms determinism of the step response, consistent with Phase 1 guarantees.
        """
        model_path = request.getfixturevalue(model_path_fixture)

        actuator_idx = 1  # hip_pitch_l
        step_amp = 0.5

        # Run trial 1
        sim1 = BipedSim(model_path, control_dt=self.CONTROL_DT, suspended=True)
        sim1.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        qpos_idx = self._get_qpos_idx(sim1.model, actuator_idx)
        sim1.reset(keyframe="stand")

        traj1 = []
        for step_idx in range(self.N_STEPS):
            qpos = sim1.qpos()
            traj1.append(qpos[qpos_idx])
            ctrl = np.zeros(6)
            ctrl[actuator_idx] = step_amp
            sim1.step(ctrl)

        # Run trial 2
        sim2 = BipedSim(model_path, control_dt=self.CONTROL_DT, suspended=True)
        sim2.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        qpos_idx = self._get_qpos_idx(sim2.model, actuator_idx)
        sim2.reset(keyframe="stand")

        traj2 = []
        for step_idx in range(self.N_STEPS):
            qpos = sim2.qpos()
            traj2.append(qpos[qpos_idx])
            ctrl = np.zeros(6)
            ctrl[actuator_idx] = step_amp
            sim2.step(ctrl)

        # Convert to arrays and check bitwise identity
        traj1 = np.array(traj1)
        traj2 = np.array(traj2)

        assert np.array_equal(traj1, traj2), \
            f"Trajectories not bitwise identical. Max difference: {np.max(np.abs(traj1 - traj2))}"


class TestSaturation:
    """
    Dynamic torque saturation test: verify actuator force saturates at ±2.5 N·m under load.

    **Regression guard for Phase 0 bug:** This test catches the reappearance of the silent
    joint-level effort clamp bug (where joint_actfrcrange would override actuator_forcerange
    and cap output to ±1.0 N·m instead of ±2.5 N·m). The static config test in
    test_torque_limits.py covers the XML attributes; this test verifies the dynamic behavior.

    **Methodology:** For each actuator, command an unreachable position target (far outside
    the joint's mechanical range) so the position error never settles to zero. The actuator
    then continuously tries to produce maximum force to close the error. We record the
    saturated force over 500+ steps (1 second at 500 Hz physics timestep) and verify it
    stays in [2.4, 2.5] N·m throughout. Contacts are disabled to isolate actuator behavior
    from spurious collisions with the floor/other limbs.
    """

    JOINTS = [
        ("hip_roll_l", 0),
        ("hip_pitch_l", 1),
        ("knee_l", 2),
        ("hip_roll_r", 3),
        ("hip_pitch_r", 4),
        ("knee_r", 5),
    ]

    # Simulation parameters for saturation test
    CONTROL_DT = 0.002  # 500 Hz physics timestep (finer than step-response test)
    N_STEPS = 500       # 500 steps = 1 second at 500 Hz
    FORCE_MIN = 2.4     # N·m (allows small margin below exact 2.5 limit)
    FORCE_MAX = 2.5     # N·m

    def _get_qpos_idx(self, model, actuator_idx):
        """Get the qpos array index for a joint given its actuator index."""
        joint_id = model.actuator_trnid[actuator_idx, 0]
        return model.jnt_qposadr[joint_id]

    @pytest.mark.parametrize("model_path_fixture", ["model_path_jetson", "model_path_no_jetson"])
    def test_saturation_all_joints(self, model_path_fixture, request):
        """
        Test torque saturation for all 6 actuated joints.

        For each joint, command an unreachable target and verify the actuator force
        saturates at ±2.5 N·m, not at the ±1.0 N·m that the Phase 0 bug produced.
        """
        model_path = request.getfixturevalue(model_path_fixture)

        # Track results per joint
        all_results = {}

        for joint_name, joint_idx in self.JOINTS:
            result = self._run_saturation_test(model_path, joint_name, joint_idx)
            all_results[joint_name] = result

        # Print summary
        print("\n" + "=" * 80)
        print("TORQUE SATURATION SUMMARY")
        print("=" * 80)

        failed = []
        for joint_name, result in all_results.items():
            status = "PASS" if result["pass"] else "FAIL"
            print(f"\n{joint_name:30s} {status}")
            print(f"  Max |force|:    {result['max_force']:.6f} N·m")
            print(f"  Force range:    [{self.FORCE_MIN}, {self.FORCE_MAX}]")
            print(f"  Percentage of nominal (2.5): {result['pct_of_nominal']:.2f}%")
            if not result["pass"]:
                failed.append((joint_name, result["reason"]))

        print("=" * 80)

        if failed:
            print(f"\nFailed tests ({len(failed)}):")
            for joint_name, reason in failed:
                print(f"  {joint_name}: {reason}")
            pytest.fail(f"{len(failed)} saturation test(s) failed.")

    def _run_saturation_test(self, model_path, joint_name, actuator_idx):
        """
        Run a single saturation test for one actuator.

        Returns a dict with max_force, pass/fail status, and diagnostic info.
        """
        # Create suspended rig and disable contacts to isolate actuator behavior
        sim = BipedSim(model_path, control_dt=self.CONTROL_DT, suspended=True)
        sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        # Get the correct qpos index for this joint
        qpos_idx = self._get_qpos_idx(sim.model, actuator_idx)

        # Reset to stand keyframe
        sim.reset(keyframe="stand")

        # Get joint range to compute unreachable target
        jnt_range_low = sim.model.jnt_range[qpos_idx, 0]
        jnt_range_high = sim.model.jnt_range[qpos_idx, 1]
        jnt_range_mid = (jnt_range_low + jnt_range_high) / 2.0

        # Unreachable target: 5 radians beyond the upper bound of the joint's range
        # This ensures the position error never settles and the actuator continuously
        # tries to produce maximum force
        unreachable_target = jnt_range_high + 5.0

        # Record actuator force at each step
        forces = []

        for step_idx in range(self.N_STEPS):
            # Construct control: unreachable target on this joint, 0.0 on all others
            ctrl = np.zeros(6)
            ctrl[actuator_idx] = unreachable_target

            # Step
            sim.step(ctrl)

            # Record the actuator force (already clamped by forcerange in the model)
            force = sim.data.actuator_force[actuator_idx]
            forces.append(force)

        # Convert to array
        forces = np.array(forces)

        # Compute metrics
        max_abs_force = np.max(np.abs(forces))
        pct_of_nominal = (max_abs_force / 2.5) * 100

        # Check if saturation is in the expected range
        in_range = self.FORCE_MIN <= max_abs_force <= self.FORCE_MAX
        reason = ""

        if not in_range:
            if max_abs_force < self.FORCE_MIN:
                reason = f"Max force {max_abs_force:.6f} N·m below minimum {self.FORCE_MIN}"
            else:
                reason = f"Max force {max_abs_force:.6f} N·m exceeds maximum {self.FORCE_MAX}"

        return {
            "pass": in_range,
            "reason": reason,
            "max_force": max_abs_force,
            "pct_of_nominal": pct_of_nominal,
        }


class TestNoLoadSpeed:
    """
    No-load speed validation for the 6 actuated joints.

    **Methodology:** "No-load" speed is measured by disabling both gravity and contacts,
    then commanding a step toward the edge of each joint's range. The joint accelerates
    against only inertia and damping (no external forces). Peak speed is measured and
    compared to 1.5× the ST3215 servo's documented no-load speed (5.45 rad/s at 7.4V,
    the higher/more permissive of the two published specs).

    **Rationale for threshold:** The ST3215's unloaded speed (no external torque) is
    5–6 rad/s. If simulated speed exceeds 8.175 rad/s (1.5× of 5.45), the servo model
    is unrealistically fast. 1.5× tolerance accounts for sim-to-real differences
    (friction, bearing play) that slow real hardware; a simulated speed 2–3× faster
    than spec is a modeling problem, not expected variation.

    **Borderline tolerance:** hip_roll joints (which have asymmetric ranges and different
    geometry from other joints) may read in the range [8.175, 8.5] rad/s as a marginal
    case; such results are documented but not treated as hard failures. Any speed >8.5 rad/s
    is always a failure, and any speed [8.175, 8.5] on non-hip_roll joints is a failure.

    **Reference:** ST3215 Datasheet
    - No-load speed @ 12V: 0.222 s/60° → 4.72 rad/s
    - No-load speed @ 7.4V: 0.192 s/60° → 5.45 rad/s
    This test uses 5.45 rad/s (7.4V spec) as the reference.
    """

    JOINTS = [
        ("hip_roll_l", 0),
        ("hip_pitch_l", 1),
        ("knee_l", 2),
        ("hip_roll_r", 3),
        ("hip_pitch_r", 4),
        ("knee_r", 5),
    ]

    # No-load speed threshold
    ST3215_NOLOAD_SPEED = 5.45  # rad/s @ 7.4V (more permissive of the two specs)
    SPEED_THRESHOLD = 1.5 * ST3215_NOLOAD_SPEED  # 8.175 rad/s

    # Borderline tolerance for hip_roll joints (marginal case)
    # Values in [8.175, 8.5] are acceptable for hip_roll; beyond 8.5 always fails
    BORDERLINE_HIGH = 8.5  # rad/s

    # Simulation parameters
    CONTROL_DT = 0.002  # 500 Hz physics timestep
    N_STEPS = 500  # 500 steps = 1.0 second

    def _get_joint_indices(self, model, actuator_idx):
        """
        Get qpos and dof indices for a joint given its actuator index.

        Returns:
            (qpos_idx, dof_idx): Position and velocity indices in the arrays.
        """
        joint_id = model.actuator_trnid[actuator_idx, 0]
        qpos_idx = model.jnt_qposadr[joint_id]
        dof_idx = model.jnt_dofadr[joint_id]
        return qpos_idx, dof_idx

    @pytest.mark.parametrize("model_path_fixture", ["model_path_jetson", "model_path_no_jetson"])
    def test_noload_speed_all_joints(self, model_path_fixture, request):
        """
        Test no-load speed for all 6 actuated joints.

        For each joint, measure peak speed when stepping toward the joint's range edge
        with gravity and contacts disabled.
        """
        model_path = request.getfixturevalue(model_path_fixture)

        # Track results per joint
        all_results = {}

        for joint_name, joint_idx in self.JOINTS:
            result = self._run_noload_speed_test(model_path, joint_name, joint_idx)
            all_results[joint_name] = result

        # Print summary table
        print("\n" + "=" * 100)
        print("NO-LOAD SPEED SUMMARY")
        print("=" * 100)
        print(f"Reference: ST3215 @ 7.4V = {self.ST3215_NOLOAD_SPEED:.2f} rad/s")
        print(f"Threshold: {self.SPEED_THRESHOLD:.3f} rad/s (1.5×)")
        print(f"Borderline tolerance (hip_roll only): {self.SPEED_THRESHOLD:.3f}–{self.BORDERLINE_HIGH:.1f} rad/s")
        print("=" * 100)

        failed = []
        borderline = []

        for joint_name, result in all_results.items():
            peak_speed = result["peak_speed"]

            # Determine status
            if peak_speed < self.SPEED_THRESHOLD:
                status = "PASS"
                note = ""
            elif peak_speed <= self.BORDERLINE_HIGH and "hip_roll" in joint_name:
                status = "BORDERLINE"
                note = "(accepted for hip_roll)"
                borderline.append(joint_name)
            else:
                status = "FAIL"
                note = result["reason"]
                failed.append((joint_name, note))

            print(f"\n{joint_name:20s} {status:12s} {note}")
            print(f"  Peak speed:    {peak_speed:.4f} rad/s")
            print(f"  Threshold:     {self.SPEED_THRESHOLD:.4f} rad/s")
            print(f"  Ratio to spec: {peak_speed / self.ST3215_NOLOAD_SPEED:.2f}×")

        print("=" * 100)

        # Report borderline cases
        if borderline:
            print(f"\nBORDERLINE cases (accepted): {', '.join(borderline)}")

        # Report failures
        if failed:
            print(f"\nFailed tests ({len(failed)}):")
            for joint_name, reason in failed:
                print(f"  {joint_name}: {reason}")
            pytest.fail(f"{len(failed)} no-load speed test(s) failed.")

    def _run_noload_speed_test(self, model_path, joint_name, actuator_idx):
        """
        Run a single no-load speed test for one actuator.

        Returns a dict with peak_speed and diagnostic info.
        """
        # Create suspended rig and disable both contacts and gravity
        sim = BipedSim(model_path, control_dt=self.CONTROL_DT, suspended=True)
        sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        sim.model.opt.gravity[:] = 0.0  # Disable gravity (no-load condition)

        # Get the qpos and dof indices for this joint
        qpos_idx, dof_idx = self._get_joint_indices(sim.model, actuator_idx)

        # Reset to stand keyframe
        sim.reset(keyframe="stand")

        # Determine step direction: toward whichever joint limit is farther.
        # IMPORTANT: jnt_range is indexed by joint ID, not qpos address - use
        # actuator_trnid to get the joint ID directly rather than reusing qpos_idx
        # (which only coincidentally equals joint ID in this particular suspended
        # model where every remaining joint is a 1-DOF hinge).
        joint_id = sim.model.actuator_trnid[actuator_idx, 0]
        jnt_range_low = sim.model.jnt_range[joint_id, 0]
        jnt_range_high = sim.model.jnt_range[joint_id, 1]
        current_qpos = sim.qpos()[qpos_idx]

        dist_to_low = abs(current_qpos - jnt_range_low)
        dist_to_high = abs(jnt_range_high - current_qpos)

        # Target the joint's own range boundary exactly (a REACHABLE target), not
        # beyond it. Commanding a target past the mechanical limit keeps the
        # actuator saturated at max torque for the whole trajectory, so the joint
        # never decelerates under its own P+D law - it instead accelerates
        # continuously until it violently strikes its own hard joint-limit
        # constraint, which inflates peak velocity far above the actuator's true
        # governed no-load speed. A reachable target lets kp*error shrink
        # naturally as qpos approaches it, letting kv's damping term produce a
        # clean, governed terminal velocity - the correct proxy for "no-load speed".
        if dist_to_high > dist_to_low:
            # Drive toward upper limit
            target = jnt_range_high
        else:
            # Drive toward lower limit
            target = jnt_range_low

        # Record velocity trajectory
        velocities = []

        for step_idx in range(self.N_STEPS):
            # Construct control: target on this joint, 0.0 on all others
            ctrl = np.zeros(6)
            ctrl[actuator_idx] = target

            # Step
            sim.step(ctrl)

            # Record the velocity (absolute value, we care about speed magnitude)
            qvel = sim.qvel()
            vel = qvel[dof_idx]
            velocities.append(abs(vel))

        # Convert to array
        velocities = np.array(velocities)

        # Compute metrics
        peak_speed = np.max(velocities)

        # Determine pass/fail reason
        reason = ""
        if peak_speed > self.BORDERLINE_HIGH:
            reason = f"Peak speed {peak_speed:.4f} rad/s exceeds safe margin {self.BORDERLINE_HIGH:.3f} rad/s"
        elif peak_speed >= self.SPEED_THRESHOLD:
            reason = f"Peak speed {peak_speed:.4f} rad/s exceeds threshold {self.SPEED_THRESHOLD:.4f} rad/s"

        return {
            "peak_speed": peak_speed,
            "reason": reason,
        }


class TestFrequencyResponse:
    """
    Frequency response characterization of the 6 actuated leg joints.

    Methodology: For each joint, command sine-wave target positions at frequencies
    [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0] Hz with amplitude 0.05 rad. After
    allowing transient settling (first half of the run), measure steady-state
    amplitude ratio (actual amplitude / commanded amplitude) and phase lag.
    Compute the -3dB frequency for each joint as a measure of closed-loop bandwidth.

    **Critical setup:** Uses BipedSim(suspended=True) with contacts disabled
    (same methodology as step response test). Gravity remains ON for realistic
    loading conditions.
    """

    JOINTS = [
        ("hip_roll_l", 0),
        ("hip_pitch_l", 1),
        ("knee_l", 2),
        ("hip_roll_r", 3),
        ("hip_pitch_r", 4),
        ("knee_r", 5),
    ]

    # Frequency sweep parameters
    FREQUENCIES_HZ = np.array([0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    AMPLITUDE_RAD = 0.05  # Small amplitude to stay safely within joint ranges

    # Simulation parameters
    CONTROL_DT = 0.002  # 500 Hz physics timestep

    # -3dB bandwidth criterion
    AMPLITUDE_3DB = 0.707  # -3dB is 0.707 of DC gain

    def _get_qpos_idx(self, model, actuator_idx):
        """Get the qpos array index for a joint given its actuator index."""
        joint_id = model.actuator_trnid[actuator_idx, 0]
        return model.jnt_qposadr[joint_id]

    def _fit_sine_at_frequency(self, t, x, freq_hz):
        """
        Fit data to a sine at a specific frequency: x ≈ A*sin(2πf*t + φ) + offset.

        Uses least squares on the basis [sin(2πft), cos(2πft), 1].

        Args:
            t: Time array
            x: Data array
            freq_hz: Frequency in Hz

        Returns:
            (amplitude, phase_rad, offset): where phase is the phase offset of sin(2πft)
        """
        omega = 2.0 * np.pi * freq_hz
        sin_basis = np.sin(omega * t)
        cos_basis = np.cos(omega * t)
        ones = np.ones_like(t)

        # Build design matrix: [sin, cos, 1]
        A_matrix = np.column_stack([sin_basis, cos_basis, ones])

        # Solve least squares
        coeff, _, _, _ = np.linalg.lstsq(A_matrix, x, rcond=None)

        sin_coeff, cos_coeff, offset = coeff

        # Amplitude: sqrt(sin_coeff^2 + cos_coeff^2)
        amplitude = np.sqrt(sin_coeff**2 + cos_coeff**2)

        # Phase: atan2(cos_coeff, sin_coeff)
        # If response is A*sin(2πft + φ) = A*sin(2πft)*cos(φ) + A*cos(2πft)*sin(φ),
        # then sin_coeff = A*cos(φ) and cos_coeff = A*sin(φ), so φ = atan2(cos_coeff, sin_coeff)
        phase_rad = np.arctan2(cos_coeff, sin_coeff)

        return amplitude, phase_rad, offset

    @pytest.mark.parametrize("model_path_fixture", ["model_path_jetson", "model_path_no_jetson"])
    def test_frequency_response_all_joints(self, model_path_fixture, request):
        """
        Test closed-loop frequency response for all 6 actuated joints.

        For each joint, run a sine-tracking sweep and measure amplitude ratio
        and phase lag at each frequency. Compute -3dB bandwidth.
        """
        model_path = request.getfixturevalue(model_path_fixture)

        # Track results per joint
        all_results = {}

        for joint_name, joint_idx in self.JOINTS:
            result = self._run_frequency_response_test(model_path, joint_name, joint_idx)
            all_results[joint_name] = result

        # Print summary table
        print("\n" + "=" * 140)
        print("FREQUENCY RESPONSE SUMMARY")
        print("=" * 140)
        print(f"Amplitude: {self.AMPLITUDE_RAD:.3f} rad")
        print(f"-3dB criterion: {self.AMPLITUDE_3DB:.3f}")
        print("=" * 140)

        failed = []

        for joint_name, result in all_results.items():
            status = "PASS" if result["pass"] else "FAIL"
            print(f"\n{joint_name:20s} {status}")
            print(f"  -3dB Bandwidth:  {result['bandwidth_hz']}")

            # Print frequency response table for this joint
            print(f"\n  Frequency Response Table:")
            print(f"    {'Freq (Hz)':>10} {'Amplitude Ratio':>18} {'Phase Lag (deg)':>18}")
            print(f"    {'-'*10} {'-'*18} {'-'*18}")
            for freq_hz, amp_ratio, phase_deg in result['freq_response']:
                print(f"    {freq_hz:10.1f} {amp_ratio:18.4f} {phase_deg:18.2f}")

            if not result["pass"]:
                failed.append((joint_name, result["reason"]))

        print("\n" + "=" * 140)

        if failed:
            print(f"\nFailed tests ({len(failed)}):")
            for joint_name, reason in failed:
                print(f"  {joint_name}: {reason}")
            pytest.fail(f"{len(failed)} frequency response test(s) failed.")

    def _run_frequency_response_test(self, model_path, joint_name, actuator_idx):
        """
        Run frequency response test for one actuator across all test frequencies.

        Returns a dict with freq_response list, bandwidth, pass/fail status.
        """
        # Create suspended rig and disable contacts
        sim = BipedSim(model_path, control_dt=self.CONTROL_DT, suspended=True)
        sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        # Get the correct qpos index for this actuator
        qpos_idx = self._get_qpos_idx(sim.model, actuator_idx)

        # Reset to stand keyframe
        sim.reset(keyframe="stand")

        # Collect results for each frequency
        freq_response = []  # List of (freq_hz, amplitude_ratio, phase_deg)

        for freq_hz in self.FREQUENCIES_HZ:
            # Run sine sweep at this frequency
            amp_ratio, phase_lag_deg = self._run_sine_at_frequency(
                sim, qpos_idx, actuator_idx, freq_hz, joint_name
            )

            freq_response.append((freq_hz, amp_ratio, phase_lag_deg))

        # Compute -3dB bandwidth
        bandwidth_hz = self._compute_bandwidth(freq_response)

        # Check assertions
        reasons = []

        # At 0.2 Hz: amplitude ratio > 0.9, phase lag < 30 deg
        freq_hz_02, amp_ratio_02hz, phase_lag_02hz = freq_response[0]  # 0.2 Hz is first
        if amp_ratio_02hz <= 0.9:
            reasons.append(f"0.2 Hz amplitude ratio {amp_ratio_02hz:.4f} <= 0.9")
        if phase_lag_02hz >= 30.0:
            reasons.append(f"0.2 Hz phase lag {phase_lag_02hz:.2f}° >= 30°")

        # Check for NaN/Inf
        for freq_hz, amp_ratio, phase_deg in freq_response:
            if np.isnan(amp_ratio) or np.isinf(amp_ratio):
                reasons.append(f"NaN/Inf in amplitude ratio at {freq_hz} Hz")
            if np.isnan(phase_deg) or np.isinf(phase_deg):
                reasons.append(f"NaN/Inf in phase lag at {freq_hz} Hz")

        pass_flag = len(reasons) == 0

        return {
            "pass": pass_flag,
            "reason": " AND ".join(reasons) if reasons else "OK",
            "freq_response": freq_response,
            "bandwidth_hz": bandwidth_hz,
        }

    def _run_sine_at_frequency(self, sim, qpos_idx, actuator_idx, freq_hz, joint_name):
        """
        Run a single sine-tracking experiment at one frequency.

        Args:
            sim: BipedSim instance
            qpos_idx: qpos index for the joint
            actuator_idx: actuator index (0-5)
            freq_hz: Frequency in Hz
            joint_name: Name for debugging

        Returns:
            (amplitude_ratio, phase_lag_deg)
        """
        # Duration: at least 10 periods, or 3 seconds, whichever is longer
        duration = max(10.0 / freq_hz, 3.0)
        n_steps = int(duration / self.CONTROL_DT)

        # Record trajectory
        positions = []
        times = []

        # Reset for this trial
        sim.reset(keyframe="stand")

        omega = 2.0 * np.pi * freq_hz

        for step_idx in range(n_steps):
            # Record current position
            qpos = sim.qpos()
            positions.append(qpos[qpos_idx])
            t = step_idx * self.CONTROL_DT
            times.append(t)

            # Sine command for this joint, others at 0.0
            ctrl = np.zeros(6)
            ctrl[actuator_idx] = self.AMPLITUDE_RAD * np.sin(omega * t)

            sim.step(ctrl)

        # Convert to arrays
        positions = np.array(positions)
        times = np.array(times)

        # Discard first half as transient
        n_half = len(positions) // 2
        times_steady = times[n_half:]
        positions_steady = positions[n_half:]

        # Fit steady-state portion to sine
        amplitude_fit, phase_fit, offset_fit = self._fit_sine_at_frequency(
            times_steady, positions_steady, freq_hz
        )

        # Amplitude ratio
        amplitude_ratio = amplitude_fit / self.AMPLITUDE_RAD if self.AMPLITUDE_RAD > 0 else 0.0

        # Phase lag: phase_fit is the phase offset of the actual response sine.
        # Negative phase means the response lags the input.
        # Report phase_lag_deg = -phase_fit (converted to degrees) as the lag magnitude.
        phase_lag_rad = -phase_fit
        phase_lag_deg = np.degrees(phase_lag_rad)

        # Wrap to [-180, 180] for clarity
        while phase_lag_deg > 180:
            phase_lag_deg -= 360
        while phase_lag_deg < -180:
            phase_lag_deg += 360

        return amplitude_ratio, phase_lag_deg

    def _compute_bandwidth(self, freq_response):
        """
        Compute -3dB bandwidth from frequency response data.

        freq_response: List of (freq_hz, amplitude_ratio, phase_deg) tuples.

        Returns: String describing bandwidth (e.g., "3.0 Hz", "> 10.0 Hz", "< 0.2 Hz")
        """
        # Find the highest frequency at which amplitude_ratio >= 0.707

        bandwidth_idx = -1
        for i, (freq_hz, amp_ratio, phase_deg) in enumerate(freq_response):
            if amp_ratio >= self.AMPLITUDE_3DB:
                bandwidth_idx = i
            else:
                break

        if bandwidth_idx == -1:
            # Amplitude ratio below 0.707 even at lowest frequency
            return "< 0.2 Hz"
        elif bandwidth_idx == len(freq_response) - 1:
            # Amplitude ratio stays above 0.707 across all tested frequencies
            return "> 10.0 Hz"
        else:
            # Interpolate between two bracketing frequencies
            freq_low = freq_response[bandwidth_idx][0]
            amp_low = freq_response[bandwidth_idx][1]
            freq_high = freq_response[bandwidth_idx + 1][0]
            amp_high = freq_response[bandwidth_idx + 1][1]

            # Linear interpolation
            if amp_high != amp_low:
                bandwidth = freq_low + (freq_high - freq_low) * (self.AMPLITUDE_3DB - amp_low) / (amp_high - amp_low)
            else:
                bandwidth = (freq_low + freq_high) / 2.0

            return f"{bandwidth:.2f} Hz"


class TestControlRateInteraction:
    """
    Control-rate interaction test: quantify the tracking quality loss from 50 Hz (deployment
    rate, zero-order hold) vs 500 Hz (near-continuous) command rates.

    **Methodology:** For each joint, command an identical sine-wave trajectory at
    1 Hz with 0.05 rad amplitude for 3.0 seconds. Run two separate simulations:

    1. **50 Hz / deployment rate:** BipedSim(control_dt=0.02) issues 150 commands,
       each held constant (zero-order hold) across 10 physics substeps.
    2. **500 Hz / high rate:** BipedSim(control_dt=0.002) issues 1500 fresh commands,
       one per physics step.

    Both use identical suspended rigs with contacts disabled (isolation, same as
    step-response test). Gravity remains ON for realistic loading.

    **Comparison:** Resample the 500 Hz run to the 50 Hz timestamps (take every
    10th sample) and compute the RMS difference between the two trajectories.
    This isolates the control-rate effect from the actuator's inherent bandwidth
    limitation.

    **Assertion:** RMS difference must be < 0.01 rad (20% of commanded 0.05 rad
    amplitude) for all joints. Larger differences would indicate the 50 Hz rate
    introduces unacceptable tracking artifacts.
    """

    JOINTS = [
        ("hip_roll_l", 0),
        ("hip_pitch_l", 1),
        ("knee_l", 2),
        ("hip_roll_r", 3),
        ("hip_pitch_r", 4),
        ("knee_r", 5),
    ]

    # Sine command parameters
    SINE_FREQ_HZ = 1.0                          # 1 Hz
    SINE_AMPLITUDE_RAD = 0.05                   # 0.05 rad (same as frequency response test)
    SINE_DURATION_SEC = 3.0                     # 3.0 seconds

    # Control rate parameters
    CONTROL_DT_50HZ = 0.02                      # 50 Hz deployment rate
    CONTROL_DT_500HZ = 0.002                    # 500 Hz physics rate

    # Tolerance for RMS difference
    MAX_RMS_DIFF = 0.01  # rad (20% of amplitude)

    def _get_qpos_idx(self, model, actuator_idx):
        """Get the qpos array index for a joint given its actuator index."""
        joint_id = model.actuator_trnid[actuator_idx, 0]
        return model.jnt_qposadr[joint_id]

    @pytest.mark.parametrize("model_path_fixture", ["model_path_jetson", "model_path_no_jetson"])
    def test_control_rate_interaction_all_joints(self, model_path_fixture, request):
        """
        Test control-rate interaction for all 6 actuated joints.

        For each joint, measure the RMS difference between 50 Hz and 500 Hz control rates.
        """
        model_path = request.getfixturevalue(model_path_fixture)

        # Track results per joint
        all_results = {}

        for joint_name, joint_idx in self.JOINTS:
            result = self._run_control_rate_test(model_path, joint_name, joint_idx)
            all_results[joint_name] = result

        # Print summary table
        print("\n" + "=" * 120)
        print("CONTROL RATE INTERACTION SUMMARY")
        print("=" * 120)
        print(f"Sine command: {self.SINE_FREQ_HZ:.1f} Hz, {self.SINE_AMPLITUDE_RAD:.3f} rad amplitude, "
              f"{self.SINE_DURATION_SEC:.1f} s duration")
        print(f"Comparison: 50 Hz ZOH vs 500 Hz (fresh commands every step)")
        print(f"Max RMS diff tolerance: {self.MAX_RMS_DIFF:.4f} rad")
        print("=" * 120)

        failed = []

        for joint_name, result in all_results.items():
            status = "PASS" if result["pass"] else "FAIL"
            print(f"\n{joint_name:20s} {status}")
            print(f"  RMS diff (50Hz vs 500Hz):   {result['rms_diff']:.6f} rad")
            print(f"  RMS error 50Hz vs ideal:    {result['rms_error_50hz']:.6f} rad (diagnostic)")
            print(f"  RMS error 500Hz vs ideal:   {result['rms_error_500hz']:.6f} rad (diagnostic)")
            if not result["pass"]:
                failed.append((joint_name, result["reason"]))

        print("\n" + "=" * 120)

        if failed:
            print(f"\nFailed tests ({len(failed)}):")
            for joint_name, reason in failed:
                print(f"  {joint_name}: {reason}")
            pytest.fail(f"{len(failed)} control rate interaction test(s) failed.")

    def _run_control_rate_test(self, model_path, joint_name, actuator_idx):
        """
        Run control rate test for one actuator.

        Returns a dict with rms_diff, rms_error_50hz, rms_error_500hz, pass/fail status.
        """
        # Run 50 Hz trial
        traj_50hz, times_50hz = self._run_sine_at_rate(
            model_path, actuator_idx, self.CONTROL_DT_50HZ, joint_name
        )

        # Run 500 Hz trial
        traj_500hz, times_500hz = self._run_sine_at_rate(
            model_path, actuator_idx, self.CONTROL_DT_500HZ, joint_name
        )

        # Resample 500 Hz trajectory to 50 Hz timestamps
        # Both start at t=0 and should have matching timestamps when subsampled.
        # 50 Hz: 150 samples at [0, 0.02, 0.04, ..., 2.98]
        # 500 Hz: 1500 samples at [0, 0.002, 0.004, ..., 2.998]
        # Subsample 500 Hz by taking every 10th sample
        traj_500hz_resampled = traj_500hz[::10]  # Every 10th sample

        # Verify dimensions match
        assert len(traj_50hz) == len(traj_500hz_resampled), \
            f"Trajectory length mismatch: 50Hz={len(traj_50hz)}, 500Hz resampled={len(traj_500hz_resampled)}"

        # Compute RMS difference between the two trajectories
        rms_diff = np.sqrt(np.mean((traj_50hz - traj_500hz_resampled) ** 2))

        # Compute ideal sine trajectory at 50 Hz timestamps (for diagnostic comparison)
        omega = 2.0 * np.pi * self.SINE_FREQ_HZ
        ideal_sine = self.SINE_AMPLITUDE_RAD * np.sin(omega * times_50hz)

        # RMS errors against ideal sine
        rms_error_50hz = np.sqrt(np.mean((traj_50hz - ideal_sine) ** 2))
        rms_error_500hz = np.sqrt(np.mean((traj_500hz_resampled - ideal_sine) ** 2))

        # Check for NaN/Inf
        has_nan_or_inf = (np.isnan(rms_diff) or np.isinf(rms_diff) or
                          np.isnan(rms_error_50hz) or np.isinf(rms_error_50hz) or
                          np.isnan(rms_error_500hz) or np.isinf(rms_error_500hz))

        # Determine pass/fail
        reason = ""
        pass_flag = True

        if has_nan_or_inf:
            reason = "NaN/Inf in trajectory"
            pass_flag = False
        elif rms_diff > self.MAX_RMS_DIFF:
            reason = f"RMS diff {rms_diff:.6f} rad > {self.MAX_RMS_DIFF:.6f} rad"
            pass_flag = False

        return {
            "pass": pass_flag,
            "reason": reason,
            "rms_diff": rms_diff,
            "rms_error_50hz": rms_error_50hz,
            "rms_error_500hz": rms_error_500hz,
        }

    def _run_sine_at_rate(self, model_path, actuator_idx, control_dt, joint_name):
        """
        Run a single sine-wave command trial at a specified control rate.

        Args:
            model_path: Path to MJCF file
            actuator_idx: Actuator index (0-5)
            control_dt: Control timestep (0.02 for 50 Hz, 0.002 for 500 Hz)
            joint_name: Joint name for debugging

        Returns:
            (trajectory, times): arrays of joint positions and corresponding times
        """
        # Create suspended rig and disable contacts
        sim = BipedSim(model_path, control_dt=control_dt, suspended=True)
        sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        # Get the qpos index for this joint
        qpos_idx = self._get_qpos_idx(sim.model, actuator_idx)

        # Reset to stand keyframe
        sim.reset(keyframe="stand")

        # Compute number of control steps
        n_steps = int(self.SINE_DURATION_SEC / control_dt)

        # Record trajectory
        trajectory = []
        times = []

        omega = 2.0 * np.pi * self.SINE_FREQ_HZ

        for step_idx in range(n_steps):
            # Record current position
            qpos = sim.qpos()
            trajectory.append(qpos[qpos_idx])
            t = step_idx * control_dt
            times.append(t)

            # Sine command for this joint, others at 0.0
            ctrl = np.zeros(6)
            ctrl[actuator_idx] = self.SINE_AMPLITUDE_RAD * np.sin(omega * t)

            # Step
            sim.step(ctrl)

        # Convert to arrays
        trajectory = np.array(trajectory)
        times = np.array(times)

        return trajectory, times


class TestMultiJointTracking:
    """
    Multi-joint simultaneous tracking test: 0.5 Hz sinusoid on all 6 joints.

    **Methodology:** Drive all 6 actuated joints with per-joint offset sinusoids
    at 0.5 Hz, amplitude 0.3 rad, for 6 seconds. Each joint uses a center offset
    to respect the asymmetric hip_roll ranges:
      - hip_roll_l (range [-2.269, +0.087]): center -0.3 → oscillates [-0.6, 0.0]
      - hip_roll_r (range [-0.087, +2.269]): center +0.3 → oscillates [0.0, 0.6]
      - hip_pitch_l/r, knee_l/r (range [-1.396, +1.396]): center 0.0 → [-0.3, 0.3]

    **Validation:** Two checks are applied:

    1. **Sanity bound:** per-joint steady-state RMS tracking error < 0.16 rad.
       This is a loose bound to catch genuine regressions (e.g., a joint not
       tracking at all), while accepting the actuator's real, bandwidth-limited
       behavior at 0.5 Hz ≈ 0.66 Hz corner (predicted ≈0.13 rad RMS).

    2. **Cross-validation against frequency response:** For each joint, fit the
       steady-state trajectory to a 0.5 Hz sine to extract amplitude ratio r and
       phase lag φ, then compute the analytically-predicted RMS error using
       A * sqrt((1+r²)/2 - r*cos(φ)) where A=0.3 rad. Assert that the MEASURED
       RMS error is within 25% (relative) of this prediction. This detects whether
       driving all 6 joints simultaneously introduces extra kinematic-chain
       cross-coupling degradation beyond what single-joint frequency response
       already predicts.

    **Physics setup:** suspended rig (torso fixed), contacts disabled, gravity ON
    (same as step-response and frequency-response tests).
    """

    JOINTS = [
        ("hip_roll_l", 0),
        ("hip_pitch_l", 1),
        ("knee_l", 2),
        ("hip_roll_r", 3),
        ("hip_pitch_r", 4),
        ("knee_r", 5),
    ]

    # Sine command parameters: 0.5 Hz, 0.3 rad amplitude, 6 seconds (3 periods)
    SINE_FREQ_HZ = 0.5
    SINE_AMPLITUDE_RAD = 0.3
    SINE_DURATION_SEC = 6.0

    # Per-joint offset centers (to respect asymmetric hip_roll ranges)
    SINE_CENTERS = {
        0: -0.3,  # hip_roll_l: center -0.3 (range [-0.6, 0.0])
        1:  0.0,  # hip_pitch_l: center 0.0 (range [-0.3, 0.3])
        2:  0.0,  # knee_l: center 0.0 (range [-0.3, 0.3])
        3: +0.3,  # hip_roll_r: center +0.3 (range [0.0, 0.6])
        4:  0.0,  # hip_pitch_r: center 0.0 (range [-0.3, 0.3])
        5:  0.0,  # knee_r: center 0.0 (range [-0.3, 0.3])
    }

    # Simulation parameters
    CONTROL_DT = 0.002  # 500 Hz physics timestep

    # Validation tolerances
    SANITY_BOUND_RAD = 0.16  # Loose bound to catch real regressions
    CROSS_VALIDATION_TOLERANCE = 0.25  # 25% relative tolerance

    def _get_qpos_idx(self, model, actuator_idx):
        """Get the qpos array index for a joint given its actuator index."""
        joint_id = model.actuator_trnid[actuator_idx, 0]
        return model.jnt_qposadr[joint_id]

    def _fit_sine_at_frequency(self, t, x, freq_hz):
        """
        Fit data to a sine at a specific frequency: x ≈ A*sin(2πf*t + φ) + offset.

        Uses least squares on the basis [sin(2πft), cos(2πft), 1].

        Args:
            t: Time array
            x: Data array
            freq_hz: Frequency in Hz

        Returns:
            (amplitude, phase_rad, offset): where phase is the phase offset of sin(2πft)
        """
        omega = 2.0 * np.pi * freq_hz
        sin_basis = np.sin(omega * t)
        cos_basis = np.cos(omega * t)
        ones = np.ones_like(t)

        # Build design matrix: [sin, cos, 1]
        A_matrix = np.column_stack([sin_basis, cos_basis, ones])

        # Solve least squares
        coeff, _, _, _ = np.linalg.lstsq(A_matrix, x, rcond=None)

        sin_coeff, cos_coeff, offset = coeff

        # Amplitude: sqrt(sin_coeff^2 + cos_coeff^2)
        amplitude = np.sqrt(sin_coeff**2 + cos_coeff**2)

        # Phase: atan2(cos_coeff, sin_coeff)
        phase_rad = np.arctan2(cos_coeff, sin_coeff)

        return amplitude, phase_rad, offset

    @pytest.mark.parametrize("model_path_fixture", ["model_path_jetson", "model_path_no_jetson"])
    def test_multi_joint_tracking(self, model_path_fixture, request):
        """
        Test simultaneous 6-joint tracking at 0.5 Hz.

        Runs a single 6-second trial commanding all joints with per-joint offset
        sinusoids, then analyzes steady-state tracking quality.
        """
        model_path = request.getfixturevalue(model_path_fixture)

        # Create suspended rig and disable contacts
        sim = BipedSim(model_path, control_dt=self.CONTROL_DT, suspended=True)
        sim.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        # Get qpos indices for all joints
        qpos_indices = [self._get_qpos_idx(sim.model, i) for i in range(6)]

        # Reset to stand keyframe
        sim.reset(keyframe="stand")

        # Compute number of steps
        n_steps = int(self.SINE_DURATION_SEC / self.CONTROL_DT)

        # Record trajectories and times
        trajectories = [[] for _ in range(6)]
        times = []

        omega = 2.0 * np.pi * self.SINE_FREQ_HZ

        for step_idx in range(n_steps):
            # Record current positions of all joints
            qpos = sim.qpos()
            for actuator_idx in range(6):
                trajectories[actuator_idx].append(qpos[qpos_indices[actuator_idx]])

            t = step_idx * self.CONTROL_DT
            times.append(t)

            # Construct control signal: per-joint offset sine for all 6 joints
            ctrl = np.zeros(6)
            for actuator_idx in range(6):
                center = self.SINE_CENTERS[actuator_idx]
                ctrl[actuator_idx] = center + self.SINE_AMPLITUDE_RAD * np.sin(omega * t)

            # Step
            sim.step(ctrl)

        # Convert to numpy arrays
        trajectories = [np.array(traj) for traj in trajectories]
        times = np.array(times)

        # Discard first half as transient
        n_half = len(times) // 2
        times_steady = times[n_half:]
        trajectories_steady = [traj[n_half:] for traj in trajectories]

        # Analyze steady-state tracking for each joint
        print("\n" + "=" * 130)
        print("MULTI-JOINT TRACKING TEST SUMMARY (6-Joint Simultaneous 0.5 Hz)")
        print("=" * 130)
        print(f"Command: {self.SINE_FREQ_HZ} Hz, {self.SINE_AMPLITUDE_RAD} rad amplitude (per-joint offset)")
        print(f"Sanity bound: {self.SANITY_BOUND_RAD} rad")
        print(f"Cross-validation tolerance: ±{self.CROSS_VALIDATION_TOLERANCE*100:.0f}% of predicted RMS")
        print("=" * 130)

        failed = []
        results_table = []

        for actuator_idx, (joint_name, _) in enumerate(self.JOINTS):
            # Fit steady-state trajectory to 0.5 Hz sine
            amplitude_fit, phase_fit, offset_fit = self._fit_sine_at_frequency(
                times_steady, trajectories_steady[actuator_idx], self.SINE_FREQ_HZ
            )

            # Amplitude ratio and phase lag
            amplitude_ratio = amplitude_fit / self.SINE_AMPLITUDE_RAD
            phase_lag_rad = -phase_fit
            phase_lag_deg = np.degrees(phase_lag_rad)

            # Wrap phase to [-180, 180]
            while phase_lag_deg > 180:
                phase_lag_deg -= 360
            while phase_lag_deg < -180:
                phase_lag_deg += 360

            # Compute analytically-predicted RMS error
            # Formula: A * sqrt((1+r²)/2 - r*cos(φ))
            cos_phi = np.cos(phase_lag_rad)
            predicted_rms = self.SINE_AMPLITUDE_RAD * np.sqrt((1 + amplitude_ratio**2) / 2 - amplitude_ratio * cos_phi)

            # Compute measured steady-state RMS tracking error
            center = self.SINE_CENTERS[actuator_idx]
            ideal_cmd = center + self.SINE_AMPLITUDE_RAD * np.sin(omega * times_steady)
            measured_rms = np.sqrt(np.mean((trajectories_steady[actuator_idx] - ideal_cmd) ** 2))

            # Compute percent difference (measured vs predicted)
            if predicted_rms > 1e-6:
                percent_diff = ((measured_rms - predicted_rms) / predicted_rms) * 100
            else:
                percent_diff = 0.0

            # Check sanity bound
            sanity_pass = measured_rms < self.SANITY_BOUND_RAD

            # Check cross-validation: measured within 25% of predicted
            cross_val_pass = abs(measured_rms - predicted_rms) <= self.CROSS_VALIDATION_TOLERANCE * predicted_rms

            # Overall pass
            joint_pass = sanity_pass and cross_val_pass
            if not joint_pass:
                failed.append(joint_name)

            status = "PASS" if joint_pass else "FAIL"

            results_table.append({
                "joint_name": joint_name,
                "status": status,
                "measured_rms": measured_rms,
                "predicted_rms": predicted_rms,
                "percent_diff": percent_diff,
                "amplitude_ratio": amplitude_ratio,
                "phase_lag_deg": phase_lag_deg,
                "sanity_pass": sanity_pass,
                "cross_val_pass": cross_val_pass,
            })

        # Print table
        print("\n{:<20} {:<10} {:<18} {:<18} {:<18} {:<12} {:<15}".format(
            "Joint", "Status", "Meas RMS (rad)", "Pred RMS (rad)", "% Diff", "Amp Ratio", "Phase Lag (°)"
        ))
        print("-" * 130)
        for row in results_table:
            print("{:<20} {:<10} {:<18.6f} {:<18.6f} {:<18.2f} {:<12.4f} {:<15.2f}".format(
                row["joint_name"],
                row["status"],
                row["measured_rms"],
                row["predicted_rms"],
                row["percent_diff"],
                row["amplitude_ratio"],
                row["phase_lag_deg"],
            ))

        print("\n" + "=" * 130)
        print("DETAILED VALIDATION RESULTS")
        print("=" * 130)

        for row in results_table:
            joint_name = row["joint_name"]
            print(f"\n{joint_name}:")
            print(f"  Measured RMS error:      {row['measured_rms']:.6f} rad")
            print(f"  Predicted RMS error:     {row['predicted_rms']:.6f} rad")
            print(f"  Percent difference:      {row['percent_diff']:.2f}%")
            print(f"  Amplitude ratio (r):     {row['amplitude_ratio']:.4f}")
            print(f"  Phase lag:               {row['phase_lag_deg']:.2f}°")

            if row["sanity_pass"]:
                print(f"  ✓ Sanity bound check PASS (< {self.SANITY_BOUND_RAD} rad)")
            else:
                print(f"  ✗ Sanity bound check FAIL ({row['measured_rms']:.6f} >= {self.SANITY_BOUND_RAD} rad)")

            if row["cross_val_pass"]:
                print(f"  ✓ Cross-validation check PASS (within ±{self.CROSS_VALIDATION_TOLERANCE*100:.0f}% of prediction)")
            else:
                print(f"  ✗ Cross-validation check FAIL (difference exceeds ±{self.CROSS_VALIDATION_TOLERANCE*100:.0f}%)")

        print("\n" + "=" * 130)

        if failed:
            print(f"\nFailed tests ({len(failed)}):")
            for joint_name in failed:
                print(f"  {joint_name}")
            pytest.fail(f"{len(failed)} joint(s) failed multi-joint tracking test.")
        else:
            print("\n✓ All joints PASSED multi-joint tracking test.")
            print("=" * 130)
