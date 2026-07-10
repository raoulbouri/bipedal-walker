import mujoco
import numpy as np
import pytest


@pytest.mark.parametrize(
    "model_fixture,expected_total_mass,expected_torso_mass",
    [
        ("model_path_jetson", 0.784, 0.279),
        ("model_path_no_jetson", 0.604, 0.099),
    ],
)
def test_model_mass(request, model_fixture, expected_total_mass, expected_torso_mass):
    """Test that model total mass and torso body mass are within tolerance."""
    model_path = request.getfixturevalue(model_fixture)
    model = mujoco.MjModel.from_xml_path(model_path)

    # Test total mass
    total_mass = np.sum(model.body_mass)
    assert np.isclose(total_mass, expected_total_mass, atol=0.001), \
        f"Total mass {total_mass} not close to {expected_total_mass}"

    # Test torso body mass
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "composite_part_1__1_")
    assert body_id >= 0, "Torso body 'composite_part_1__1_' not found"

    torso_mass = model.body_mass[body_id]
    assert np.isclose(torso_mass, expected_torso_mass, atol=0.001), \
        f"Torso mass {torso_mass} not close to {expected_torso_mass}"


@pytest.mark.parametrize(
    "model_fixture",
    [
        "model_path_jetson",
        "model_path_no_jetson",
    ],
)
def test_actuator_count_and_names(request, model_fixture):
    """Test that model has 6 actuators with the expected names."""
    model_path = request.getfixturevalue(model_fixture)
    model = mujoco.MjModel.from_xml_path(model_path)

    # Test actuator count
    assert model.nu == 6, f"Expected 6 actuators, got {model.nu}"

    # Test actuator names
    expected_names = {
        "act_hip_roll_l",
        "act_knee_l",
        "act_ankle_l",
        "act_hip_roll_r",
        "act_knee_r",
        "act_ankle_r",
    }

    actuator_names = set()
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        actuator_names.add(name)

    assert actuator_names == expected_names, \
        f"Actuator names {actuator_names} do not match expected {expected_names}"


@pytest.mark.parametrize(
    "model_fixture",
    [
        "model_path_jetson",
        "model_path_no_jetson",
    ],
)
def test_actuator_kp_gain(request, model_fixture):
    """Test that all actuators have kp gain close to 40.0 (Phase 4 retune, 2026-07-10)."""
    model_path = request.getfixturevalue(model_fixture)
    model = mujoco.MjModel.from_xml_path(model_path)

    expected_kp = 40.0

    for i in range(model.nu):
        kp = model.actuator_gainprm[i, 0]
        assert np.isclose(kp, expected_kp, atol=0.001), \
            f"Actuator {i} kp {kp} not close to {expected_kp}"
