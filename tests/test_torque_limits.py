"""Regression test for torque limit fix: ensure joint & actuator limits are at 2.5 N·m."""
import pytest
import mujoco
import numpy as np


@pytest.mark.parametrize("model_path_fixture", ["model_path_jetson", "model_path_no_jetson"])
def test_torque_limits(model_path_fixture, request):
    """
    Verify that both joint effort limits and actuator forceranges are set to 2.5 N·m.

    This test catches the original bug where joint limit effort was clamped to 1.0 N·m
    even though the actuator forcerange was set to 3.0 N·m.
    """
    # Load the fixture value
    model_path = request.getfixturevalue(model_path_fixture)

    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path(model_path)

    # Test 1: Exactly 6 actuators
    assert model.nu == 6, f"Expected 6 actuators, got {model.nu}"

    # Test 2: All actuators have forcerange of [-2.5, 2.5]
    tol = 1e-6
    for i in range(model.nu):
        forcerange = model.actuator_forcerange[i]
        assert np.allclose(forcerange[0], -2.5, atol=tol), \
            f"Actuator {i}: lower forcerange {forcerange[0]} != -2.5"
        assert np.allclose(forcerange[1], 2.5, atol=tol), \
            f"Actuator {i}: upper forcerange {forcerange[1]} != 2.5"

    # Test 3: Each joint that has an actuator should either be unlimited or have
    # a limit that covers at least [-2.5, 2.5]
    # First, collect all joint ids that have actuators
    actuated_joint_ids = set()
    for i in range(model.nu):
        # Get the transmission type
        trntype = model.actuator_trntype[i]
        # For JOINT transmissions (trntype == mjtTrn.mjTRN_JOINT)
        if trntype == mujoco.mjtTrn.mjTRN_JOINT:
            # Get the joint id that this actuator acts on
            joint_id = model.actuator_trnid[i, 0]
            actuated_joint_ids.add(joint_id)

    # Now verify each actuated joint
    for joint_id in actuated_joint_ids:
        is_limited = model.jnt_actfrclimited[joint_id]
        if not is_limited:
            # Unlimited is always fine
            continue

        # Limited: check that the range covers at least [-2.5, 2.5]
        actfrcrange = model.jnt_actfrcrange[joint_id]
        lower = actfrcrange[0]
        upper = actfrcrange[1]

        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        assert lower <= -2.5 + tol, \
            f"Joint {joint_name} (id {joint_id}): lower actfrcrange {lower} < -2.5"
        assert upper >= 2.5 - tol, \
            f"Joint {joint_name} (id {joint_id}): upper actfrcrange {upper} < 2.5"
