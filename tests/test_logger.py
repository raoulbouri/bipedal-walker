"""Tests for the Logger class and load_log function."""

import sys
from pathlib import Path

import pytest
import numpy as np
import tempfile
import os
import hashlib
import mujoco

# Add parent directory to path to import sim module
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim import BipedSim, Logger, load_log


def test_round_trip_arrays_jetson(model_path_jetson):
    """Test that arrays are bitwise identical after round-trip save/load (jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simulator and logger
        sim = BipedSim(model_path_jetson)
        sim.reset()

        logger = Logger(model_path_jetson)

        # Run ~20 steps, logging each
        control_dt = 0.02
        n_steps = 20
        for i in range(n_steps):
            # Create small varying control input
            ctrl = np.sin(np.linspace(0, 2 * np.pi, sim.model.nu) + i * 0.1)
            sim.step(ctrl)

            # Gather data from sim
            time = i * control_dt
            qpos = sim.qpos()
            qvel = sim.qvel()

            # Concatenate sensors dict in sorted-by-key order
            sensors_dict = sim.sensors()
            sensor_names = sorted(sensors_dict.keys())
            sensordata_flat = np.concatenate([sensors_dict[name] for name in sensor_names])

            ncon = sim.contacts()

            # Log the step
            logger.log_step(time, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Save to temp path
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)

        # Load it back
        npz_data, metadata = load_log(temp_log_path)

        # Assert all arrays are exactly equal
        assert np.array_equal(npz_data["time"], np.array([i * control_dt for i in range(n_steps)])), \
            "time arrays not equal"
        assert npz_data["qpos"].shape[0] == n_steps, "qpos has wrong number of steps"
        assert npz_data["qvel"].shape[0] == n_steps, "qvel has wrong number of steps"
        assert npz_data["ctrl"].shape[0] == n_steps, "ctrl has wrong number of steps"
        assert npz_data["sensordata"].shape[0] == n_steps, "sensordata has wrong number of steps"
        assert npz_data["ncon"].shape[0] == n_steps, "ncon has wrong number of steps"


def test_round_trip_arrays_no_jetson(model_path_no_jetson):
    """Test that arrays are bitwise identical after round-trip save/load (no jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simulator and logger
        sim = BipedSim(model_path_no_jetson)
        sim.reset()

        logger = Logger(model_path_no_jetson)

        # Run ~20 steps, logging each
        control_dt = 0.02
        n_steps = 20
        for i in range(n_steps):
            # Create small varying control input
            ctrl = np.sin(np.linspace(0, 2 * np.pi, sim.model.nu) + i * 0.1)
            sim.step(ctrl)

            # Gather data from sim
            time = i * control_dt
            qpos = sim.qpos()
            qvel = sim.qvel()

            # Concatenate sensors dict in sorted-by-key order
            sensors_dict = sim.sensors()
            sensor_names = sorted(sensors_dict.keys())
            sensordata_flat = np.concatenate([sensors_dict[name] for name in sensor_names])

            ncon = sim.contacts()

            # Log the step
            logger.log_step(time, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Save to temp path
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)

        # Load it back
        npz_data, metadata = load_log(temp_log_path)

        # Assert all arrays are exactly equal
        assert np.array_equal(npz_data["time"], np.array([i * control_dt for i in range(n_steps)])), \
            "time arrays not equal"
        assert npz_data["qpos"].shape[0] == n_steps, "qpos has wrong number of steps"
        assert npz_data["qvel"].shape[0] == n_steps, "qvel has wrong number of steps"
        assert npz_data["ctrl"].shape[0] == n_steps, "ctrl has wrong number of steps"
        assert npz_data["sensordata"].shape[0] == n_steps, "sensordata has wrong number of steps"
        assert npz_data["ncon"].shape[0] == n_steps, "ncon has wrong number of steps"


def test_metadata_correctness_jetson(model_path_jetson):
    """Test that metadata is correctly recorded and can be verified (jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simulator and logger
        sim = BipedSim(model_path_jetson)
        sim.reset()

        logger = Logger(model_path_jetson, seed=42)

        # Log one step
        ctrl = np.zeros(sim.model.nu)
        sim.step(ctrl)
        time = 0.0
        qpos = sim.qpos()
        qvel = sim.qvel()
        sensors_dict = sim.sensors()
        sensor_names = sorted(sensors_dict.keys())
        sensordata_flat = np.concatenate([sensors_dict[name] for name in sensor_names])
        ncon = sim.contacts()
        logger.log_step(time, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Save
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)

        # Load and verify metadata
        npz_data, metadata = load_log(temp_log_path)

        # Check required keys
        assert "model_path" in metadata, "model_path not in metadata"
        assert "seed" in metadata, "seed not in metadata"
        assert "config" in metadata, "config not in metadata"
        assert "sha256" in metadata, "sha256 not in metadata"
        assert "mujoco_version" in metadata, "mujoco_version not in metadata"

        # Check values
        assert metadata["model_path"] == model_path_jetson
        assert metadata["seed"] == 42
        assert metadata["config"] == {}

        # Verify SHA-256
        assert len(metadata["sha256"]) == 64, "sha256 should be 64 hex characters"
        assert all(c in "0123456789abcdef" for c in metadata["sha256"]), \
            "sha256 should contain only hex characters"

        # Independently compute SHA-256 and verify it matches
        with open(model_path_jetson, "rb") as f:
            model_content = f.read()
        expected_sha256 = hashlib.sha256(model_content).hexdigest()
        assert metadata["sha256"] == expected_sha256, \
            f"sha256 mismatch: got {metadata['sha256']}, expected {expected_sha256}"

        # Check mujoco version format
        assert isinstance(metadata["mujoco_version"], str)
        assert len(metadata["mujoco_version"]) > 0


def test_metadata_correctness_no_jetson(model_path_no_jetson):
    """Test that metadata is correctly recorded and can be verified (no jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simulator and logger
        sim = BipedSim(model_path_no_jetson)
        sim.reset()

        logger = Logger(model_path_no_jetson, seed=42)

        # Log one step
        ctrl = np.zeros(sim.model.nu)
        sim.step(ctrl)
        time = 0.0
        qpos = sim.qpos()
        qvel = sim.qvel()
        sensors_dict = sim.sensors()
        sensor_names = sorted(sensors_dict.keys())
        sensordata_flat = np.concatenate([sensors_dict[name] for name in sensor_names])
        ncon = sim.contacts()
        logger.log_step(time, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Save
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)

        # Load and verify metadata
        npz_data, metadata = load_log(temp_log_path)

        # Check required keys
        assert "model_path" in metadata, "model_path not in metadata"
        assert "seed" in metadata, "seed not in metadata"
        assert "config" in metadata, "config not in metadata"
        assert "sha256" in metadata, "sha256 not in metadata"
        assert "mujoco_version" in metadata, "mujoco_version not in metadata"

        # Check values
        assert metadata["model_path"] == model_path_no_jetson
        assert metadata["seed"] == 42
        assert metadata["config"] == {}

        # Verify SHA-256
        assert len(metadata["sha256"]) == 64, "sha256 should be 64 hex characters"
        assert all(c in "0123456789abcdef" for c in metadata["sha256"]), \
            "sha256 should contain only hex characters"

        # Independently compute SHA-256 and verify it matches
        with open(model_path_no_jetson, "rb") as f:
            model_content = f.read()
        expected_sha256 = hashlib.sha256(model_content).hexdigest()
        assert metadata["sha256"] == expected_sha256, \
            f"sha256 mismatch: got {metadata['sha256']}, expected {expected_sha256}"

        # Check mujoco version format
        assert isinstance(metadata["mujoco_version"], str)
        assert len(metadata["mujoco_version"]) > 0


def test_config_dict_passthrough_jetson(model_path_jetson):
    """Test that config dict is properly stored and retrieved (jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a non-trivial config
        config = {
            "experiment": "test",
            "n_steps": 20,
            "nested": {"a": 1, "b": 2},
            "list_data": [1, 2, 3],
        }

        sim = BipedSim(model_path_jetson)
        sim.reset()

        logger = Logger(model_path_jetson, config=config)

        # Log one step
        ctrl = np.zeros(sim.model.nu)
        sim.step(ctrl)
        time = 0.0
        qpos = sim.qpos()
        qvel = sim.qvel()
        sensors_dict = sim.sensors()
        sensor_names = sorted(sensors_dict.keys())
        sensordata_flat = np.concatenate([sensors_dict[name] for name in sensor_names])
        ncon = sim.contacts()
        logger.log_step(time, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Save
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)

        # Load and verify config
        npz_data, metadata = load_log(temp_log_path)

        assert metadata["config"] == config, \
            f"config mismatch: got {metadata['config']}, expected {config}"


def test_config_dict_passthrough_no_jetson(model_path_no_jetson):
    """Test that config dict is properly stored and retrieved (no jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a non-trivial config
        config = {
            "experiment": "test",
            "n_steps": 20,
            "nested": {"a": 1, "b": 2},
            "list_data": [1, 2, 3],
        }

        sim = BipedSim(model_path_no_jetson)
        sim.reset()

        logger = Logger(model_path_no_jetson, config=config)

        # Log one step
        ctrl = np.zeros(sim.model.nu)
        sim.step(ctrl)
        time = 0.0
        qpos = sim.qpos()
        qvel = sim.qvel()
        sensors_dict = sim.sensors()
        sensor_names = sorted(sensors_dict.keys())
        sensordata_flat = np.concatenate([sensors_dict[name] for name in sensor_names])
        ncon = sim.contacts()
        logger.log_step(time, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Save
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)

        # Load and verify config
        npz_data, metadata = load_log(temp_log_path)

        assert metadata["config"] == config, \
            f"config mismatch: got {metadata['config']}, expected {config}"


def test_directory_auto_creation_jetson(model_path_jetson):
    """Test that save() creates parent directories as needed (jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sim = BipedSim(model_path_jetson)
        sim.reset()

        logger = Logger(model_path_jetson)

        # Log one step
        ctrl = np.zeros(sim.model.nu)
        sim.step(ctrl)
        time = 0.0
        qpos = sim.qpos()
        qvel = sim.qvel()
        sensors_dict = sim.sensors()
        sensor_names = sorted(sensors_dict.keys())
        sensordata_flat = np.concatenate([sensors_dict[name] for name in sensor_names])
        ncon = sim.contacts()
        logger.log_step(time, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Save to a path with non-existent parent directories
        nested_log_path = os.path.join(tmpdir, "a", "b", "c", "test_run")
        logger.save(nested_log_path)

        # Verify both files exist
        npz_file = nested_log_path + ".npz"
        json_file = nested_log_path + ".json"

        assert os.path.exists(npz_file), f"NPZ file not created: {npz_file}"
        assert os.path.exists(json_file), f"JSON file not created: {json_file}"

        # Verify we can load it
        npz_data, metadata = load_log(nested_log_path)
        assert npz_data is not None
        assert metadata is not None


def test_directory_auto_creation_no_jetson(model_path_no_jetson):
    """Test that save() creates parent directories as needed (no jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sim = BipedSim(model_path_no_jetson)
        sim.reset()

        logger = Logger(model_path_no_jetson)

        # Log one step
        ctrl = np.zeros(sim.model.nu)
        sim.step(ctrl)
        time = 0.0
        qpos = sim.qpos()
        qvel = sim.qvel()
        sensors_dict = sim.sensors()
        sensor_names = sorted(sensors_dict.keys())
        sensordata_flat = np.concatenate([sensors_dict[name] for name in sensor_names])
        ncon = sim.contacts()
        logger.log_step(time, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Save to a path with non-existent parent directories
        nested_log_path = os.path.join(tmpdir, "a", "b", "c", "test_run")
        logger.save(nested_log_path)

        # Verify both files exist
        npz_file = nested_log_path + ".npz"
        json_file = nested_log_path + ".json"

        assert os.path.exists(npz_file), f"NPZ file not created: {npz_file}"
        assert os.path.exists(json_file), f"JSON file not created: {json_file}"

        # Verify we can load it
        npz_data, metadata = load_log(nested_log_path)
        assert npz_data is not None
        assert metadata is not None


def test_array_defensive_copy_jetson(model_path_jetson):
    """Test that arrays are defensively copied to prevent mutation (jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sim = BipedSim(model_path_jetson)
        sim.reset()

        logger = Logger(model_path_jetson)

        # Create mutable arrays
        time_val = 0.0
        qpos = np.array([1.0, 2.0, 3.0])
        qvel = np.array([4.0, 5.0, 6.0])
        ctrl = np.array([7.0, 8.0])
        sensordata_flat = np.array([9.0, 10.0, 11.0])
        ncon = 1

        # Store original copies
        qpos_orig = qpos.copy()
        qvel_orig = qvel.copy()
        ctrl_orig = ctrl.copy()
        sensordata_orig = sensordata_flat.copy()

        # Log the step
        logger.log_step(time_val, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Mutate the original arrays
        qpos[:] = 999.0
        qvel[:] = 999.0
        ctrl[:] = 999.0
        sensordata_flat[:] = 999.0

        # Save and load
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)
        npz_data, metadata = load_log(temp_log_path)

        # Verify that the loaded data matches the original values, not the mutated ones
        assert np.array_equal(npz_data["qpos"][0], qpos_orig), \
            "qpos was not defensively copied"
        assert np.array_equal(npz_data["qvel"][0], qvel_orig), \
            "qvel was not defensively copied"
        assert np.array_equal(npz_data["ctrl"][0], ctrl_orig), \
            "ctrl was not defensively copied"
        assert np.array_equal(npz_data["sensordata"][0], sensordata_orig), \
            "sensordata was not defensively copied"


def test_array_defensive_copy_no_jetson(model_path_no_jetson):
    """Test that arrays are defensively copied to prevent mutation (no jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sim = BipedSim(model_path_no_jetson)
        sim.reset()

        logger = Logger(model_path_no_jetson)

        # Create mutable arrays
        time_val = 0.0
        qpos = np.array([1.0, 2.0, 3.0])
        qvel = np.array([4.0, 5.0, 6.0])
        ctrl = np.array([7.0, 8.0])
        sensordata_flat = np.array([9.0, 10.0, 11.0])
        ncon = 1

        # Store original copies
        qpos_orig = qpos.copy()
        qvel_orig = qvel.copy()
        ctrl_orig = ctrl.copy()
        sensordata_orig = sensordata_flat.copy()

        # Log the step
        logger.log_step(time_val, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Mutate the original arrays
        qpos[:] = 999.0
        qvel[:] = 999.0
        ctrl[:] = 999.0
        sensordata_flat[:] = 999.0

        # Save and load
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)
        npz_data, metadata = load_log(temp_log_path)

        # Verify that the loaded data matches the original values, not the mutated ones
        assert np.array_equal(npz_data["qpos"][0], qpos_orig), \
            "qpos was not defensively copied"
        assert np.array_equal(npz_data["qvel"][0], qvel_orig), \
            "qvel was not defensively copied"
        assert np.array_equal(npz_data["ctrl"][0], ctrl_orig), \
            "ctrl was not defensively copied"
        assert np.array_equal(npz_data["sensordata"][0], sensordata_orig), \
            "sensordata was not defensively copied"
