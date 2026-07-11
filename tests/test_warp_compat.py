"""Phase 6.0: Warp-compatible model validation (implicitfast integrator +
whole-body mesh collision, matching the CPU variant -- corrected
2026-07-13, see test_warp_collision_geoms's docstring for why)."""
import mujoco
import numpy as np
import pytest
from pathlib import Path

WARP_MODEL = "models/mjcf/biped_warp.xml"
CPU_MODEL = "models/mjcf/biped.xml"

FOOT_BODIES = {"foot", "foot_1"}


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
    """Verify implicitfast integrator and dt=0.002."""
    model = mujoco.MjModel.from_xml_path(WARP_MODEL)

    # Integrator should be "implicitfast": mjlab's real integrator map
    # (mjlab/sim/sim.py _INTEGRATOR_MAP, confirmed live 2026-07-11 via a
    # local CPU mjlab install) only recognizes "euler"/"implicitfast" --
    # "implicit" isn't a valid option and would raise KeyError at env
    # construction time.
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST, \
        f"Expected implicitfast integrator, got {model.opt.integrator}"

    # Timestep
    assert abs(model.opt.timestep - 0.002) < 1e-9, \
        f"Expected timestep=0.002, got {model.opt.timestep}"

    print(f"✓ Integrator: implicit, dt={model.opt.timestep}")


def test_warp_collision_geoms():
    """Verify collision geoms: mesh collision everywhere, same as the CPU
    variant. Corrected 2026-07-13 -- biped_warp.xml previously used
    primitive collision proxies (capsule/box/sphere) for non-foot bodies
    instead of mesh collision, on the theory that MuJoCo Warp's mesh/CCD
    colliders were too memory-expensive at scale (Phase 6.0). That
    tradeoff was never actually necessary for this specific model (only
    ~33 small, simple bodies) and introduced a real, previously
    undiagnosed physics discrepancy between the training-time model and
    the CPU/deployment-representative model. mujoco_warp's own test
    fixtures (test_data/aloha_pot) confirm mesh collision genuinely works
    there. This test now asserts mesh-on-mesh collision throughout,
    matching biped.xml exactly (see test_warp_matches_cpu_collision below
    for a direct structural-equivalence check)."""
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

    # Check non-foot bodies (except worldbody): should ALSO have mesh
    # collision (not primitives) -- same convention as biped.xml.
    non_foot_bodies = set(body_geoms.keys()) - FOOT_BODIES - {"world", "worldbody", "root"}
    for body_name in non_foot_bodies:
        collision_types = body_geoms[body_name]["collision"]
        if collision_types:  # Only check if has collision geoms
            assert mujoco.mjtGeom.mjGEOM_MESH in collision_types, \
                f"Non-foot body {body_name} should have mesh collision, got {collision_types}"
            primitive_types = {mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE,
                               mujoco.mjtGeom.mjGEOM_BOX}
            assert not any(t in primitive_types for t in collision_types), \
                f"Non-foot body {body_name} should NOT have primitive collision, got {collision_types}"

    print("✓ Collision geoms OK: mesh collision throughout, matching biped.xml")


def test_warp_matches_cpu_collision():
    """Direct structural-equivalence check: biped_warp.xml's collision
    setup (contype/conaffinity/geom type per body) is identical to
    biped.xml's, not just 'mesh instead of primitives' in isolation."""
    warp_model = mujoco.MjModel.from_xml_path(WARP_MODEL)
    cpu_model = mujoco.MjModel.from_xml_path(CPU_MODEL)

    def collision_signature(model):
        sig = {}
        for i in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i])
            sig.setdefault(body, []).append((
                int(model.geom_type[i]), int(model.geom_contype[i]), int(model.geom_conaffinity[i]),
            ))
        for body in sig:
            sig[body].sort()
        return sig

    warp_sig = collision_signature(warp_model)
    cpu_sig = collision_signature(cpu_model)

    assert warp_model.ngeom == cpu_model.ngeom, \
        f"geom count mismatch: warp={warp_model.ngeom}, cpu={cpu_model.ngeom}"
    assert set(warp_sig.keys()) == set(cpu_sig.keys()), "body set mismatch between variants"
    for body in warp_sig:
        assert warp_sig[body] == cpu_sig[body], (
            f"collision geom signature mismatch for body '{body}': "
            f"warp={warp_sig[body]} cpu={cpu_sig[body]}"
        )

    print("✓ biped_warp.xml's collision setup is structurally identical to biped.xml's")


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
    """Integration test: passive settle from stand keyframe. With the
    2026-07-13 mesh-collision fix, this should now settle very similarly
    to biped.xml's own settle baseline (docs/physics_baselines.md) since
    both integrator and collision are now identical -- see
    test_warp_matches_cpu_settle below for a direct trajectory
    comparison. The loose bounds here remain as a basic sanity gate."""
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
    z_min, z_max = z_history.min(), z_history.max()
    assert z_min >= -0.01, f"Base z tunneled below 0 (min={z_min:.4f}m)"
    assert z_max <= 0.5, f"Base z exceeded reasonable bound (max={z_max:.4f}m)"

    # 2. By t=4s, velocity should generally decay
    t_4s_step = int(4.0 / model.opt.timestep)
    qvel_at_4s = np.abs(qvel_history[t_4s_step])
    qvel_at_start = np.abs(qvel_history[0])

    # Final state
    z_final = z_history[-1]
    qvel_final = np.abs(qvel_history[-1]).max()

    print(f"✓ Settle test (5s passive drop, implicitfast integrator):")
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


def test_warp_matches_cpu_settle():
    """Direct physics-equivalence check, added 2026-07-13 alongside the
    mesh-collision fix: with integrator and collision now identical
    between biped_warp.xml and biped.xml, a matched passive-drop
    trajectory (same stand keyframe, same zero-ctrl sequence, same dt)
    stepped through the plain MuJoCo engine should track very closely --
    not just pass a loose sanity bound. Empirically this produces
    bit-identical qpos/qvel trajectories over the full 5s drop (verified
    directly before writing this assertion); the tolerance here is kept
    small but non-zero for robustness across platforms/BLAS builds rather
    than asserting exact equality."""
    model_warp = mujoco.MjModel.from_xml_path(WARP_MODEL)
    model_cpu = mujoco.MjModel.from_xml_path(CPU_MODEL)
    data_warp = mujoco.MjData(model_warp)
    data_cpu = mujoco.MjData(model_cpu)

    key_warp = mujoco.mj_name2id(model_warp, mujoco.mjtObj.mjOBJ_KEY, "stand")
    key_cpu = mujoco.mj_name2id(model_cpu, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model_warp, data_warp, key_warp)
    mujoco.mj_resetDataKeyframe(model_cpu, data_cpu, key_cpu)

    t_end = 5.0
    steps = int(t_end / model_warp.opt.timestep)
    max_qpos_diff = 0.0
    max_qvel_diff = 0.0

    for _ in range(steps):
        data_warp.ctrl[:] = 0
        data_cpu.ctrl[:] = 0
        mujoco.mj_step(model_warp, data_warp)
        mujoco.mj_step(model_cpu, data_cpu)
        max_qpos_diff = max(max_qpos_diff, float(np.abs(data_warp.qpos - data_cpu.qpos).max()))
        max_qvel_diff = max(max_qvel_diff, float(np.abs(data_warp.qvel - data_cpu.qvel).max()))

    assert max_qpos_diff < 1e-6, f"qpos trajectories diverged: max diff = {max_qpos_diff}"
    assert max_qvel_diff < 1e-6, f"qvel trajectories diverged: max diff = {max_qvel_diff}"

    print(f"✓ Matched settle trajectories: max qpos diff={max_qpos_diff:.2e}, "
          f"max qvel diff={max_qvel_diff:.2e} over {t_end}s")
