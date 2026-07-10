"""Tests for the replay function."""

import sys
from pathlib import Path

import pytest
import numpy as np
import tempfile
import os

# Add parent directory to path to import sim module
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim import BipedSim, Logger, load_log, replay


def test_replay_reproduces_log_jetson(model_path_jetson):
    """Test that replay() reproduces the logged trajectory bitwise-exactly (jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simulator and logger
        sim = BipedSim(model_path_jetson)
        sim.reset()

        logger = Logger(model_path_jetson)

        # Run ~30 steps with deterministic control input
        control_dt = 0.02
        n_steps = 30
        for i in range(n_steps):
            # Create deterministic control: small sinusoidal variation per actuator
            ctrl = 0.05 * np.sin(0.1 * i + np.linspace(0, 2 * np.pi, sim.model.nu))
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

        # Save the log
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)

        # Load the log
        npz_data, metadata = load_log(temp_log_path)
        logged_qpos = npz_data["qpos"]
        logged_qvel = npz_data["qvel"]

        # Replay the log
        replayed = replay(temp_log_path, model_path_jetson, control_dt=control_dt)
        replayed_qpos = replayed["qpos"]
        replayed_qvel = replayed["qvel"]

        # Assert bitwise equality
        assert np.array_equal(replayed_qpos, logged_qpos), \
            "Replayed qpos does not match logged qpos bitwise-exactly"
        assert np.array_equal(replayed_qvel, logged_qvel), \
            "Replayed qvel does not match logged qvel bitwise-exactly"


def test_replay_reproduces_log_no_jetson(model_path_no_jetson):
    """Test that replay() reproduces the logged trajectory bitwise-exactly (no jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simulator and logger
        sim = BipedSim(model_path_no_jetson)
        sim.reset()

        logger = Logger(model_path_no_jetson)

        # Run ~30 steps with deterministic control input
        control_dt = 0.02
        n_steps = 30
        for i in range(n_steps):
            # Create deterministic control: small sinusoidal variation per actuator
            ctrl = 0.05 * np.sin(0.1 * i + np.linspace(0, 2 * np.pi, sim.model.nu))
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

        # Save the log
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)

        # Load the log
        npz_data, metadata = load_log(temp_log_path)
        logged_qpos = npz_data["qpos"]
        logged_qvel = npz_data["qvel"]

        # Replay the log
        replayed = replay(temp_log_path, model_path_no_jetson, control_dt=control_dt)
        replayed_qpos = replayed["qpos"]
        replayed_qvel = replayed["qvel"]

        # Assert bitwise equality
        assert np.array_equal(replayed_qpos, logged_qpos), \
            "Replayed qpos does not match logged qpos bitwise-exactly"
        assert np.array_equal(replayed_qvel, logged_qvel), \
            "Replayed qvel does not match logged qvel bitwise-exactly"


def test_replay_is_reproducible_jetson(model_path_jetson):
    """Test that replay() is itself reproducible (jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simulator and logger
        sim = BipedSim(model_path_jetson)
        sim.reset()

        logger = Logger(model_path_jetson)

        # Run ~30 steps with deterministic control input
        control_dt = 0.02
        n_steps = 30
        for i in range(n_steps):
            # Create deterministic control
            ctrl = 0.05 * np.sin(0.1 * i + np.linspace(0, 2 * np.pi, sim.model.nu))
            sim.step(ctrl)

            # Gather data
            time = i * control_dt
            qpos = sim.qpos()
            qvel = sim.qvel()
            sensors_dict = sim.sensors()
            sensor_names = sorted(sensors_dict.keys())
            sensordata_flat = np.concatenate([sensors_dict[name] for name in sensor_names])
            ncon = sim.contacts()

            logger.log_step(time, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Save the log
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)

        # Load the log
        npz_data, metadata = load_log(temp_log_path)
        logged_qpos = npz_data["qpos"]
        logged_qvel = npz_data["qvel"]

        # Replay twice
        replay1 = replay(temp_log_path, model_path_jetson, control_dt=control_dt)
        replay2 = replay(temp_log_path, model_path_jetson, control_dt=control_dt)

        # Assert three-way bitwise equality
        assert np.array_equal(replay1["qpos"], replay2["qpos"]), \
            "Replay #1 and #2 qpos are not bitwise-equal"
        assert np.array_equal(replay1["qvel"], replay2["qvel"]), \
            "Replay #1 and #2 qvel are not bitwise-equal"
        assert np.array_equal(replay1["qpos"], logged_qpos), \
            "Replay #1 qpos does not match logged qpos"
        assert np.array_equal(replay1["qvel"], logged_qvel), \
            "Replay #1 qvel does not match logged qvel"
        assert np.array_equal(replay2["qpos"], logged_qpos), \
            "Replay #2 qpos does not match logged qpos"
        assert np.array_equal(replay2["qvel"], logged_qvel), \
            "Replay #2 qvel does not match logged qvel"


def test_replay_is_reproducible_no_jetson(model_path_no_jetson):
    """Test that replay() is itself reproducible (no jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simulator and logger
        sim = BipedSim(model_path_no_jetson)
        sim.reset()

        logger = Logger(model_path_no_jetson)

        # Run ~30 steps with deterministic control input
        control_dt = 0.02
        n_steps = 30
        for i in range(n_steps):
            # Create deterministic control
            ctrl = 0.05 * np.sin(0.1 * i + np.linspace(0, 2 * np.pi, sim.model.nu))
            sim.step(ctrl)

            # Gather data
            time = i * control_dt
            qpos = sim.qpos()
            qvel = sim.qvel()
            sensors_dict = sim.sensors()
            sensor_names = sorted(sensors_dict.keys())
            sensordata_flat = np.concatenate([sensors_dict[name] for name in sensor_names])
            ncon = sim.contacts()

            logger.log_step(time, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Save the log
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)

        # Load the log
        npz_data, metadata = load_log(temp_log_path)
        logged_qpos = npz_data["qpos"]
        logged_qvel = npz_data["qvel"]

        # Replay twice
        replay1 = replay(temp_log_path, model_path_no_jetson, control_dt=control_dt)
        replay2 = replay(temp_log_path, model_path_no_jetson, control_dt=control_dt)

        # Assert three-way bitwise equality
        assert np.array_equal(replay1["qpos"], replay2["qpos"]), \
            "Replay #1 and #2 qpos are not bitwise-equal"
        assert np.array_equal(replay1["qvel"], replay2["qvel"]), \
            "Replay #1 and #2 qvel are not bitwise-equal"
        assert np.array_equal(replay1["qpos"], logged_qpos), \
            "Replay #1 qpos does not match logged qpos"
        assert np.array_equal(replay1["qvel"], logged_qvel), \
            "Replay #1 qvel does not match logged qvel"
        assert np.array_equal(replay2["qpos"], logged_qpos), \
            "Replay #2 qpos does not match logged qpos"
        assert np.array_equal(replay2["qvel"], logged_qvel), \
            "Replay #2 qvel does not match logged qvel"


def test_replay_motion_is_not_trivial_jetson(model_path_jetson):
    """Test that the logged run actually produces non-trivial motion (jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simulator and logger
        sim = BipedSim(model_path_jetson)
        sim.reset()

        logger = Logger(model_path_jetson)

        # Run ~30 steps with deterministic control input
        control_dt = 0.02
        n_steps = 30
        for i in range(n_steps):
            # Create deterministic control
            ctrl = 0.05 * np.sin(0.1 * i + np.linspace(0, 2 * np.pi, sim.model.nu))
            sim.step(ctrl)

            # Gather data
            time = i * control_dt
            qpos = sim.qpos()
            qvel = sim.qvel()
            sensors_dict = sim.sensors()
            sensor_names = sorted(sensors_dict.keys())
            sensordata_flat = np.concatenate([sensors_dict[name] for name in sensor_names])
            ncon = sim.contacts()

            logger.log_step(time, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Save the log
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)

        # Load the log
        npz_data, metadata = load_log(temp_log_path)
        logged_qpos = npz_data["qpos"]

        # Sanity check: qpos should actually change
        assert not np.array_equal(logged_qpos[0], logged_qpos[-1]), \
            "qpos did not change over the run; control sequence is trivial"


def test_replay_motion_is_not_trivial_no_jetson(model_path_no_jetson):
    """Test that the logged run actually produces non-trivial motion (no jetson model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create simulator and logger
        sim = BipedSim(model_path_no_jetson)
        sim.reset()

        logger = Logger(model_path_no_jetson)

        # Run ~30 steps with deterministic control input
        control_dt = 0.02
        n_steps = 30
        for i in range(n_steps):
            # Create deterministic control
            ctrl = 0.05 * np.sin(0.1 * i + np.linspace(0, 2 * np.pi, sim.model.nu))
            sim.step(ctrl)

            # Gather data
            time = i * control_dt
            qpos = sim.qpos()
            qvel = sim.qvel()
            sensors_dict = sim.sensors()
            sensor_names = sorted(sensors_dict.keys())
            sensordata_flat = np.concatenate([sensors_dict[name] for name in sensor_names])
            ncon = sim.contacts()

            logger.log_step(time, qpos, qvel, ctrl, sensordata_flat, ncon)

        # Save the log
        temp_log_path = os.path.join(tmpdir, "test_run")
        logger.save(temp_log_path)

        # Load the log
        npz_data, metadata = load_log(temp_log_path)
        logged_qpos = npz_data["qpos"]

        # Sanity check: qpos should actually change
        assert not np.array_equal(logged_qpos[0], logged_qpos[-1]), \
            "qpos did not change over the run; control sequence is trivial"
