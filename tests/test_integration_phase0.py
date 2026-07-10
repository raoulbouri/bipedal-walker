import pytest
import numpy as np
import mujoco


def find_stand_keyframe_id(model):
    """Find the index of the 'stand' keyframe."""
    for i in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
        if name == "stand":
            return i
    raise ValueError("Stand keyframe not found")


def get_sensor_value(model, data, sensor_name):
    """Get the value of a sensor by name."""
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        raise ValueError(f"Sensor '{sensor_name}' not found")
    start_idx = model.sensor_adr[sensor_id]
    return data.sensordata[start_idx]


@pytest.mark.parametrize("model_path", ["models/mjcf/biped.xml", "models/mjcf/biped_no_jetson.xml"])
def test_stability_500_steps(model_path):
    """Test 500-step stability from the stand keyframe.

    Loads the model, initializes to stand keyframe, and runs 500 physics steps
    with passive dynamics (zero controls). Asserts that:
    - All qpos and qvel remain finite (no NaN/Inf)
    - Base height (qpos[2]) never sinks below -0.05 (guards against floor tunneling)
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Find stand keyframe
    stand_key_id = find_stand_keyframe_id(model)

    # Initialize to stand pose
    data.qpos[:] = model.key_qpos[stand_key_id]
    mujoco.mj_forward(model, data)

    # Run 500 steps with checks
    for step in range(500):
        mujoco.mj_step(model, data)

        # Check for NaN/Inf
        assert np.all(np.isfinite(data.qpos)), f"Step {step}: qpos contains non-finite values"
        assert np.all(np.isfinite(data.qvel)), f"Step {step}: qvel contains non-finite values"

        # Check base height (z-coordinate is qpos[2])
        assert data.qpos[2] > -0.05, f"Step {step}: base height {data.qpos[2]} is below -0.05"


@pytest.mark.parametrize("model_path", ["models/mjcf/biped.xml", "models/mjcf/biped_no_jetson.xml"])
def test_foot_contact_at_rest(model_path):
    """Test that both feet are in contact at the stand keyframe.

    Initializes to stand keyframe and immediately checks that both touch_l
    and touch_r sensors have positive readings without stepping.
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Find stand keyframe
    stand_key_id = find_stand_keyframe_id(model)

    # Initialize to stand pose
    data.qpos[:] = model.key_qpos[stand_key_id]
    mujoco.mj_forward(model, data)

    # Check foot contact
    touch_l = get_sensor_value(model, data, "touch_l")
    touch_r = get_sensor_value(model, data, "touch_r")

    assert touch_l > 0, f"Left foot touch sensor reading is {touch_l}, expected > 0"
    assert touch_r > 0, f"Right foot touch sensor reading is {touch_r}, expected > 0"


@pytest.mark.parametrize("model_path", ["models/mjcf/biped.xml", "models/mjcf/biped_no_jetson.xml"])
def test_torque_saturation_probe(model_path):
    """Test that actuator force saturates at 2.5 N·m.

    Sets act_knee_l control to an extreme value (100.0) to force saturation,
    runs 20 steps, and verifies that the absolute actuator force settles
    in the range [2.4, 2.5] N·m (allowing small margin for the exact limit).
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Find stand keyframe
    stand_key_id = find_stand_keyframe_id(model)

    # Initialize to stand pose
    data.qpos[:] = model.key_qpos[stand_key_id]
    mujoco.mj_forward(model, data)

    # Find actuator
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_knee_l")
    assert actuator_id >= 0, "Actuator 'act_knee_l' not found"

    # Set extreme control target to force saturation
    data.ctrl[actuator_id] = 100.0

    # Run 20 steps to let force settle
    for _ in range(20):
        mujoco.mj_step(model, data)

    # Check force saturation
    force_magnitude = abs(data.actuator_force[actuator_id])
    assert 2.4 <= force_magnitude <= 2.5, f"Actuator force magnitude {force_magnitude} is not in [2.4, 2.5]"
