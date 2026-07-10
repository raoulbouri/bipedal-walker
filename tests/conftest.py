import pytest


@pytest.fixture
def model_path_jetson():
    return "models/mjcf/biped.xml"


@pytest.fixture
def model_path_no_jetson():
    return "models/mjcf/biped_no_jetson.xml"
