import numpy as np
import mujoco
import xml.etree.ElementTree as ET
import os
import uuid


class BipedSim:
    """MuJoCo simulation wrapper for biped robot."""

    def __init__(self, model_path, control_dt=0.02, suspended=False):
        """
        Initialize the simulator.

        Args:
            model_path: Path to the MJCF model file.
            control_dt: Control timestep in seconds. Must be an exact integer multiple
                       of model.opt.timestep.
            suspended: If True, remove the floating base joint and weld the root body
                      to the world, creating a suspended variant.

        Raises:
            ValueError: If control_dt is not an exact multiple of model.opt.timestep.
        """
        tmp_model_path = None
        try:
            if suspended:
                # Create the suspended model and get the temp path
                tmp_model_path = self._create_suspended_model(model_path)
                actual_model_path = tmp_model_path
            else:
                actual_model_path = model_path

            self.model = mujoco.MjModel.from_xml_path(actual_model_path)
            self.data = mujoco.MjData(self.model)

            # Verify that control_dt is an exact integer multiple of physics timestep
            physics_dt = self.model.opt.timestep
            n_substeps_float = control_dt / physics_dt
            n_substeps = round(n_substeps_float)

            # Check that multiplying back reproduces control_dt within tolerance
            reproduced_dt = n_substeps * physics_dt
            tolerance = 1e-9
            if abs(reproduced_dt - control_dt) > tolerance:
                raise ValueError(
                    f"control_dt={control_dt} is not an exact integer multiple of "
                    f"physics timestep={physics_dt}. "
                    f"n_substeps={n_substeps} reproduces dt={reproduced_dt}."
                )

            self.n_substeps = n_substeps
        finally:
            # Clean up temp file immediately after loading
            if tmp_model_path is not None and os.path.exists(tmp_model_path):
                try:
                    os.remove(tmp_model_path)
                except OSError:
                    pass  # Ignore cleanup errors

    def _create_suspended_model(self, model_path):
        """
        Create a suspended variant of the model by removing the floating base joint
        and truncating keyframe qpos to retain only hinge joint angles.

        Args:
            model_path: Path to the original MJCF model file.

        Returns:
            Path to the temp XML file. The caller must clean it up after loading.
        """
        # Parse the original model
        tree = ET.parse(model_path)
        root_elem = tree.getroot()
        worldbody = root_elem.find('worldbody')
        root_body = worldbody.find('body')  # the "root" body

        # Remove the freejoint
        freejoint = root_body.find('freejoint')
        if freejoint is not None:
            root_body.remove(freejoint)

        # Truncate all keyframe qpos: drop first 7 values (freejoint pos+quat),
        # keep only the hinge joint angles
        keyframe_elem = root_elem.find('keyframe')
        if keyframe_elem is not None:
            for key in keyframe_elem.findall('key'):
                qpos_vals = key.get('qpos').split()
                key.set('qpos', ' '.join(qpos_vals[7:]))

        # Create a unique temp file in the same directory as model_path
        model_dir = os.path.dirname(os.path.abspath(model_path))
        unique_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        tmp_path = os.path.join(model_dir, f'_tmp_suspended_{unique_id}.xml')

        # Write temp file
        tree.write(tmp_path)

        return tmp_path

    def reset(self, keyframe="stand"):
        """
        Reset the simulation to a named keyframe.

        Args:
            keyframe: Name of the keyframe (default "stand").

        Raises:
            ValueError: If the keyframe name is not found.
        """
        # Look up keyframe id
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe
        )
        if key_id < 0:
            raise ValueError(f"Keyframe '{keyframe}' not found in model.")

        # Reset to keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # Forward pass to populate derived quantities (sensordata, etc.)
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl):
        """
        Step the simulation with the given control input.

        Args:
            ctrl: Control array of length model.nu, values within actuator ranges.
        """
        # Clip control to actuator ranges
        clipped_ctrl = np.array(ctrl, dtype=np.float64)

        for i in range(self.model.nu):
            if self.model.actuator_ctrllimited[i]:
                lo, hi = self.model.actuator_ctrlrange[i]
                clipped_ctrl[i] = np.clip(clipped_ctrl[i], lo, hi)

        # Set control and step
        self.data.ctrl[:] = clipped_ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

    def sensors(self):
        """
        Get current sensor readings.

        Returns:
            A dictionary mapping sensor names to numpy arrays (copies).
        """
        sensor_dict = {}
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, i
            )
            start = self.model.sensor_adr[i]
            dim = self.model.sensor_dim[i]
            sensor_data = self.data.sensordata[start : start + dim].copy()
            sensor_dict[sensor_name] = sensor_data
        return sensor_dict

    def qpos(self):
        """Return a copy of joint positions."""
        return self.data.qpos.copy()

    def qvel(self):
        """Return a copy of joint velocities."""
        return self.data.qvel.copy()

    def com(self):
        """
        Return the whole-body center of mass position.

        Returns:
            A 3-element numpy array with the CoM position in world coordinates.
        """
        total_mass = np.sum(self.model.body_mass)
        com_pos = np.zeros(3)
        for i in range(self.model.nbody):
            com_pos += self.model.body_mass[i] * self.data.xipos[i]
        com_pos /= total_mass
        return com_pos.copy()

    def contacts(self):
        """
        Return the number of active contacts.

        Returns:
            An integer count of contacts.
        """
        return self.data.ncon
