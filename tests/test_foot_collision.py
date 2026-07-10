"""Tests for the passive foot-tilt joint's limits and whole-body collision configuration."""
import math
import mujoco
import numpy as np
import pytest


# Foot (passive hardstop) joint limit constants (in radians)
FOOT_LO = math.radians(12 - 30)  # radians(-18) ≈ -0.31416 rad
FOOT_HI = math.radians(47 - 30)  # radians(17) ≈ 0.29671 rad


@pytest.mark.parametrize(
    "model_fixture",
    [
        "model_path_jetson",
        "model_path_no_jetson",
    ],
)
def test_foot_joint_limits(request, model_fixture):
    """
    Test that both passive foot-tilt joints have correct limit ranges.

    The two foot joints (foot_l and foot_r) are passive hinge joints with
    hardstop limits that should match the computed FOOT_LO and FOOT_HI values.
    Named "foot" (not "ankle") to avoid colliding with the actuated ankle_l/r
    joints, which match the motor controller / joint encoder firmware naming
    on the Jetson (see ~/Work/biped python_st3215 package) for the lowest
    actuated joint.
    """
    model_path = request.getfixturevalue(model_fixture)
    model = mujoco.MjModel.from_xml_path(model_path)

    # Expected range for both foot joints
    expected_range = np.array([FOOT_LO, FOOT_HI])
    tolerance = 0.001

    # Test both foot joints
    for foot_name in ["foot_l", "foot_r"]:
        # Look up joint id
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, foot_name)
        assert joint_id >= 0, f"Joint '{foot_name}' not found in model"

        # Check that joint limits are enabled
        is_limited = model.jnt_limited[joint_id]
        assert is_limited, f"Joint '{foot_name}' (id {joint_id}) does not have limits enabled"

        # Check the range
        actual_range = model.jnt_range[joint_id]
        assert np.allclose(actual_range, expected_range, atol=tolerance), \
            f"Joint '{foot_name}' (id {joint_id}): range {actual_range} not close to {expected_range}"


@pytest.mark.parametrize(
    "model_fixture",
    [
        "model_path_jetson",
        "model_path_no_jetson",
    ],
)
def test_floor_collision_flags(request, model_fixture):
    """Test that the floor geom has collision type and affinity both set to 1."""
    model_path = request.getfixturevalue(model_fixture)
    model = mujoco.MjModel.from_xml_path(model_path)

    # Look up floor geom
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert geom_id >= 0, "Floor geom not found in model"

    # Check collision type and affinity
    contype = model.geom_contype[geom_id]
    conaffinity = model.geom_conaffinity[geom_id]

    assert contype == 1, f"Floor geom: contype {contype} != 1"
    assert conaffinity == 1, f"Floor geom: conaffinity {conaffinity} != 1"


@pytest.mark.parametrize(
    "model_fixture",
    [
        "model_path_jetson",
        "model_path_no_jetson",
    ],
)
def test_wholebody_mesh_collision_flags(request, model_fixture):
    """
    Test that all mesh-type geoms in group 1 have collision enabled, EXCEPT
    geoms belonging to the foot bodies ("foot", "foot_1"). Those bodies' own
    visual mesh geom is deliberately left at contype=0/conaffinity=0 by
    postprocess.py's whole-body-collision step, because each foot body
    already has a separate purpose-built foot_col_l/foot_col_r collision
    geom (group 3) added earlier in the pipeline — see postprocess.py's own
    `skip_bodies = set(FOOT_BODIES.keys())` logic. The exclusion is by
    parent BODY name, not by geom name (the visual geom being excluded is
    unnamed; only the separately-added collision geom is named
    "foot_col_*"). Note: these BODY names ("foot", "foot_1") are a separate
    MuJoCo namespace from the passive foot_l/foot_r JOINT names above - they
    come from Onshape mesh/part naming and are unrelated to the joint rename.

    This guards against regression where the robot fell through the floor
    because only the feet had collision enabled.
    """
    model_path = request.getfixturevalue(model_fixture)
    model = mujoco.MjModel.from_xml_path(model_path)

    skip_body_names = {"foot", "foot_1"}
    skip_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in skip_body_names
    }
    assert all(bid >= 0 for bid in skip_body_ids), \
        f"Expected foot bodies {skip_body_names} not found in model"

    checked_count = 0
    for geom_id in range(model.ngeom):
        geom_type = model.geom_type[geom_id]
        geom_group = model.geom_group[geom_id]

        if geom_type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if geom_group != 1:
            continue

        parent_body_id = model.geom_bodyid[geom_id]
        if parent_body_id in skip_body_ids:
            continue

        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        geom_name_str = geom_name if geom_name else f"<unnamed geom {geom_id}>"

        contype = model.geom_contype[geom_id]
        conaffinity = model.geom_conaffinity[geom_id]

        assert contype == 1, \
            f"Mesh geom '{geom_name_str}' (id {geom_id}, group {geom_group}): contype {contype} != 1"
        assert conaffinity == 1, \
            f"Mesh geom '{geom_name_str}' (id {geom_id}, group {geom_group}): conaffinity {conaffinity} != 1"
        checked_count += 1

    # Sanity check so this test can't silently pass by checking nothing.
    assert checked_count >= 20, \
        f"Expected to check at least 20 whole-body mesh geoms, only checked {checked_count}"
