"""Logger for recording and replaying simulation runs."""

import numpy as np
import json
import os
import hashlib
import mujoco


class Logger:
    """Records simulation runs to disk for later replay and inspection."""

    def __init__(self, model_path, seed=None, config=None):
        """
        Initialize a logger.

        Args:
            model_path: Path to the MJCF model file.
            seed: Optional random seed for reproducibility.
            config: Optional dict of experiment configuration (default {}).
        """
        self.model_path = model_path
        self.seed = seed
        self.config = config if config is not None else {}

        # Initialize lists to accumulate per-step data
        self.times = []
        self.qpos_list = []
        self.qvel_list = []
        self.ctrl_list = []
        self.sensordata_list = []  # Each entry is a flat array
        self.ncon_list = []

    def log_step(self, time, qpos, qvel, ctrl, sensordata_flat, ncon):
        """
        Log a single simulation step.

        Args:
            time: Simulation time (scalar).
            qpos: Joint positions (array-like).
            qvel: Joint velocities (array-like).
            ctrl: Control input (array-like).
            sensordata_flat: Flattened sensor data (array-like).
            ncon: Number of contacts (int).
        """
        # Defensively copy all arrays
        self.times.append(np.float64(time))
        self.qpos_list.append(np.array(qpos, dtype=np.float64).copy())
        self.qvel_list.append(np.array(qvel, dtype=np.float64).copy())
        self.ctrl_list.append(np.array(ctrl, dtype=np.float64).copy())
        self.sensordata_list.append(np.array(sensordata_flat, dtype=np.float64).copy())
        self.ncon_list.append(np.int32(ncon))

    def save(self, out_path):
        """
        Save the logged data to disk.

        Writes two files:
        - out_path + ".npz": Binary array data (time, qpos, qvel, ctrl, sensordata, ncon)
        - out_path + ".json": Metadata (model_path, seed, config, sha256, mujoco_version)

        Args:
            out_path: Path without extension (e.g., "runs/my_experiment").
        """
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(out_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # Stack arrays into 2D (or 1D where appropriate)
        time_array = np.array(self.times, dtype=np.float64)  # 1D
        qpos_array = np.array(self.qpos_list, dtype=np.float64)  # 2D: n_steps x nq
        qvel_array = np.array(self.qvel_list, dtype=np.float64)  # 2D: n_steps x nvel
        ctrl_array = np.array(self.ctrl_list, dtype=np.float64)  # 2D: n_steps x nctrl
        sensordata_array = np.array(self.sensordata_list, dtype=np.float64)  # 2D: n_steps x nsensor
        ncon_array = np.array(self.ncon_list, dtype=np.int32)  # 1D

        # Save as .npz
        npz_path = out_path + ".npz"
        np.savez(
            npz_path,
            time=time_array,
            qpos=qpos_array,
            qvel=qvel_array,
            ctrl=ctrl_array,
            sensordata=sensordata_array,
            ncon=ncon_array,
        )

        # Compute SHA-256 of model file
        with open(self.model_path, "rb") as f:
            model_content = f.read()
        sha256_digest = hashlib.sha256(model_content).hexdigest()

        # Prepare metadata
        metadata = {
            "model_path": self.model_path,
            "seed": self.seed,
            "config": self.config,
            "sha256": sha256_digest,
            "mujoco_version": mujoco.__version__,
        }

        # Save as .json
        json_path = out_path + ".json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_log(path):
    """
    Load a previously saved log.

    Args:
        path: Path without extension (same convention as Logger.save).

    Returns:
        A tuple (npz_data_dict, metadata_dict) where:
        - npz_data_dict is a plain dict of numpy arrays (file not left open)
        - metadata_dict is the parsed JSON metadata
    """
    # Load .npz
    npz_path = path + ".npz"
    npz_file = np.load(npz_path)
    npz_data_dict = dict(npz_file)  # Convert to plain dict to close file handle
    npz_file.close()

    # Load .json
    json_path = path + ".json"
    with open(json_path, "r") as f:
        metadata_dict = json.load(f)

    return npz_data_dict, metadata_dict


def replay(log_path, model_path, control_dt=0.02):
    """
    Replay a logged run: reset a fresh BipedSim to the log's initial state,
    re-apply the logged ctrl sequence step-by-step, and return the replayed
    qpos/qvel trajectories for comparison against the original log.

    Args:
        log_path: Path to the saved log (without extension).
        model_path: Path to the MJCF model file.
        control_dt: Control timestep in seconds (default 0.02).

    Returns:
        A dict: {"qpos": np.ndarray (n_steps x nq), "qvel": np.ndarray (n_steps x nv)}
        containing the REPLAYED trajectory (one row per step, matching the row count
        and meaning of the logged ctrl array - i.e. row i is the state AFTER applying
        ctrl row i).
    """
    from .biped_sim import BipedSim

    # Load the log
    npz_data, metadata = load_log(log_path)
    ctrl_array = npz_data["ctrl"]  # shape: (n_steps, nctrl)

    # Create and reset the simulator
    sim = BipedSim(model_path, control_dt=control_dt)
    sim.reset()

    # Initialize arrays to store replayed trajectory
    n_steps = ctrl_array.shape[0]
    nq = sim.model.nq
    nv = sim.model.nv

    replayed_qpos = np.zeros((n_steps, nq), dtype=np.float64)
    replayed_qvel = np.zeros((n_steps, nv), dtype=np.float64)

    # Replay the control sequence
    for i in range(n_steps):
        sim.step(ctrl_array[i])
        replayed_qpos[i] = sim.qpos()
        replayed_qvel[i] = sim.qvel()

    return {
        "qpos": replayed_qpos,
        "qvel": replayed_qvel,
    }
