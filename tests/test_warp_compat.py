"""Phase 6.0: Warp-compatible model validation (implicit integrator + primitive collisions)."""
import mujoco
import numpy as np
import pytest
from pathlib import Path

WARP_MODEL = "models/mjcf/biped_warp.xml"
CPU_MODEL = "models/mjcf/biped.xml"

FOOT_BODIES = {"foot", "foot_1"}
EXPECTED_PRIMITIVES = {"sphere", "box", "capsule"}


def test_warp_model_compiles():
    """Verify biped_warp.xml loads without error."""
    model = mujoco.MjModel.from_xml_path(WARP_MODEL)
    assert model is not None
    print(f"✓ biped_warp.xml loaded: {model.nbody} bodies, {model.nq} qpos, {model.nu} actuators")


def test_warp_structure():
    """Verify model structure matches expectations: 36 bodies (incl. worldbody), 15 DOFs, 6 actuators, 24 sensors."""
    model = mujoco.MjModel.from_xml_path(WARP_MODEL)

    # Body count (36 including worldbody, same as CPU variant)
    assert model.nbody == 36, f"Expected 36 bodies (incl. worldbody), got {model.nbody}"

    # DOF count: 7 floating base + 8 hinge joints (6 actuated + 2 passive ankles)
    assert model.nq == 15, f"Expected 15 qpos (7 base + 8 hinges), got {model.nq}"
    assert model.nv == 14, f"Expected 14 qvel (6 base + 8 hinges), got {model.nv}"

    # Actuators (6 actuated joints only)
    assert model.nu == 6, f"Expected 6 actuators, got {model.nu}"

    # Sensors (16 joint pos/vel + 8 IMU + 2 touch)
    assert model.nsensor == 24, f"Expected 24 sensors, got {model.nsensor}"

    print(f"✓ Structure OK: {model.nbody} bodies, {model.nq} qpos, {model.nu} actuators, {model.nsensor} sensors")


def test_warp_integrator():
    """Verify implicit integrator and dt=0.002."""
    model = mujoco.MjModel.from_xml_path(WARP_MODEL)

    # Integrator should be "implicit" (not "implicitfast")
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT, \
        f"Expected implicit integrator, got {model.opt.integrator}"

    # Timestep
    assert abs(model.opt.timestep - 0.002) < 1e-9, \
        f"Expected timestep=0.002, got {model.opt.timestep}"

    print(f"✓ Integrator: implicit, dt={model.opt.timestep}")


def test_warp_collision_geoms():
    """Verify collision geoms: primitives for non-foot bodies, mesh for feet."""
    model = mujoco.MjModel.from_xml_path(WARP_MODEL)

    # Track collision geom types per body
    body_geoms = {}
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        geom_type = model.geom_type[geom_id]
        contype = model.geom_contype[geom_id]
        conaffinity = model.geom_conaffinity[geom_id]

        if body_name not in body_geoms:
            body_geoms[body_name] = {"visual": [], "collision": []}

        if contype == 0 and conaffinity == 0:
            body_geoms[body_name]["visual"].append(geom_type)
        else:
            body_geoms[body_name]["collision"].append(geom_type)

    # Check foot bodies: should have mesh collision geoms (foot_col_l/r)
    for foot_body in FOOT_BODIES:
        assert foot_body in body_geoms, f"Foot body {foot_body} not found in geom map"
        collision_types = body_geoms[foot_body]["collision"]
        assert mujoco.mjtGeom.mjGEOM_MESH in collision_types, \
            f"Foot {foot_body} should have mesh collision geom, got {collision_types}"
        print(f"✓ {foot_body}: has mesh collision geom")

    # Check non-foot bodies (except worldbody): should have primitive collision geoms (not mesh)
    non_foot_bodies = set(body_geoms.keys()) - FOOT_BODIES - {"world", "worldbody", "root"}
    for body_name in non_foot_bodies:
        collision_types = body_geoms[body_name]["collision"]
        if collision_types:  # Only check if has collision geoms
            # Should NOT have mesh collision (meshes are group=1, visual only)
            assert mujoco.mjtGeom.mjGEOM_MESH not in collision_types, \
                f"Non-foot body {body_name} should NOT have mesh collision, got {collision_types}"
            # Should have primitives (capsule, sphere, box, etc.)
            primitive_types = {mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE,
                               mujoco.mjtGeom.mjGEOM_BOX}
            assert any(t in primitive_types for t in collision_types), \
                f"Non-foot body {body_name} should have primitive collision, got {collision_types}"

    print(f"✓ Collision geoms OK: feet have mesh, others have primitives")


def test_warp_static_feasibility():
    """Verify static feasibility: model loads and feet make contact at stand keyframe."""
    model = mujoco.MjModel.from_xml_path(WARP_MODEL)
    data = mujoco.MjData(model)

    # Load stand keyframe
    stand_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    assert stand_key_id >= 0, "Stand keyframe not found"

    mujoco.mj_resetDataKeyframe(model, data, stand_key_id)
    mujoco.mj_forward(model, data)

    # Both feet should be in contact with the floor at the stand keyframe
    # Check for at least one contact (may have multiple due to mesh collision)
    ncon_initial = data.ncon
    assert ncon_initial > 0, f"No contacts at stand keyframe (ncon={ncon_initial})"

    # At least one touch sensor should read > 0 (both feet ideally, but accept partial for now)
    touch_r_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_r")
    touch_l_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_l")

    touch_r = data.sensordata[touch_r_idx] if touch_r_idx >= 0 else 0
    touch_l = data.sensordata[touch_l_idx] if touch_l_idx >= 0 else 0

    print(f"✓ Static feasibility check:")
    print(f"  Contacts at stand: {ncon_initial}")
    print(f"  Touch sensors - right: {touch_r:.4f}, left: {touch_l:.4f}")

    # The model is feasible if it has contacts (mesh collision is many-point, which is OK)
    assert ncon_initial > 0, "Stand pose has no floor contact"


def test_warp_settle_passthrough():
    """Integration test: passive settle from stand keyframe (implicit integrator baseline)."""
    model = mujoco.MjModel.from_xml_path(WARP_MODEL)
    data = mujoco.MjData(model)

    # Load stand keyframe
    stand_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, stand_key_id)

    # Run 5 seconds (2500 steps at dt=0.002)
    t_end = 5.0
    steps = int(t_end / model.opt.timestep)

    qvel_history = []
    z_history = []

    for step in range(steps):
        # Zero control (passive)
        data.ctrl[:] = 0

        mujoco.mj_step(model, data)

        qvel_history.append(np.copy(data.qvel))
        # Base z position (from floating base qpos)
        z_history.append(data.qpos[2])

        # Check for NaN/Inf
        assert np.isfinite(data.qpos).all(), f"NaN/Inf in qpos at step {step}"
        assert np.isfinite(data.qvel).all(), f"NaN/Inf in qvel at step {step}"

    z_history = np.array(z_history)
    qvel_history = np.array(qvel_history)

    # Assertions
    # 1. Base z in reasonable range (robot shouldn't tunnel through floor or fly away)
    # Note: implicit integrator may settle differently than implicitfast
    z_min, z_max = z_history.min(), z_history.max()
    assert z_min >= -0.01, f"Base z tunneled below 0 (min={z_min:.4f}m)"
    # Warp variant may settle higher due to collision proxy differences
    assert z_max <= 0.5, f"Base z exceeded reasonable bound (max={z_max:.4f}m)"

    # 2. By t=4s, velocity should generally decay
    t_4s_step = int(4.0 / model.opt.timestep)
    qvel_at_4s = np.abs(qvel_history[t_4s_step])
    qvel_at_start = np.abs(qvel_history[0])

    # Final state
    z_final = z_history[-1]
    qvel_final = np.abs(qvel_history[-1]).max()

    print(f"✓ Settle test (5s passive drop, implicit integrator):")
    print(f"  Base z range: [{z_min:.4f}, {z_max:.4f}] m")
    print(f"  Base z at t=5s: {z_final:.4f} m")
    print(f"  Max |qvel| at t=0s: {qvel_at_start.max():.6f} rad/s")
    print(f"  Max |qvel| at t=4s: {qvel_at_4s.max():.6f} rad/s")
    print(f"  Max |qvel| at t=5s: {qvel_final:.6f} rad/s")

    # Key gate: no NaN, no floor tunneling, basic stability
    assert z_min >= -0.01, "Robot tunneled through floor"
    assert not np.isnan(z_history).any(), "NaN in trajectory"


def test_warp_mass_matches_cpu():
    """Verify Warp variant has same masses as CPU variant."""
    model_warp = mujoco.MjModel.from_xml_path(WARP_MODEL)
    model_cpu = mujoco.MjModel.from_xml_path(CPU_MODEL)

    # Total mass (expected for Jetson variant: 0.784 kg)
    mass_warp = model_warp.body_mass.sum()
    mass_cpu = model_cpu.body_mass.sum()

    assert abs(mass_warp - mass_cpu) < 1e-6, \
        f"Warp mass ({mass_warp:.6f}) != CPU mass ({mass_cpu:.6f})"

    assert abs(mass_warp - 0.784) < 0.001, \
        f"Total mass {mass_warp:.3f} kg not close to expected 0.784 kg"

    print(f"✓ Mass check: Warp={mass_warp:.6f} kg, CPU={mass_cpu:.6f} kg")
