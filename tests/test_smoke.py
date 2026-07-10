import mujoco


def test_biped_jetson_loads(model_path_jetson):
    model = mujoco.MjModel.from_xml_path(model_path_jetson)
    assert model is not None
    assert model.nbody > 0


def test_biped_no_jetson_loads(model_path_no_jetson):
    model = mujoco.MjModel.from_xml_path(model_path_no_jetson)
    assert model is not None
    assert model.nbody > 0
