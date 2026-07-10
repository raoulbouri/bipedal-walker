"""
Local single-environment wrapper for visual/manual testing (Phase 6.E
companion, not part of the frozen vectorized BipedEnvCfg/registry shape).

`BipedEnvCfg` (config.py) is a vectorized-shaped config that mirrors
mjlab's real API surface but is never actually steppable locally - no
vectorized CPU env exists or is planned (mjlab/MuJoCo Warp on Colab owns
that in Phase 7). This module is a SEPARATE, single-environment wrapper
that actually runs the assembled observation/reward/termination/command/
init-noise pieces against the real `biped_warp.xml` physics via
`sim.BipedSim` (the Phase 1 CPU harness), purely so a human can sanity-
check and visually inspect the Phase 6.A-6.D pieces on macOS ahead of the
real Colab training run. It is intentionally NOT registered in
`TASK_REGISTRY` and is not part of the mjlab-shaped config.

Important MuJoCo pitfall (see MEMORY.md's 2026-07-10 Phase 3 entry):
`mujoco.mj_step` computes position/velocity-stage sensors from the state
at the START of the step, before the integrator advances qpos/qvel - so
`BipedSim.sensors()` read immediately after `BipedSim.step()` would be one
full step STALE relative to the just-advanced qpos/qvel. This wrapper
calls an explicit `mujoco.mj_forward` after every `sim.step()` (and after
manually writing a noised qpos on reset) before ever reading sensors, to
guarantee observations always reflect the current, just-advanced state.
"""

from typing import Optional, Tuple

import numpy as np
import mujoco

from sim.biped_sim import BipedSim
from .config import BipedEnvCfg
from .observations import build_actor_obs, build_critic_obs, ACTOR_OBS_DIM
from .commands import sample_command
from .init_noise import sample_init_state, apply_init_state
from .domain_randomization import DomainRandomizer
from .rewards import compute_reward
from .terminations import fall, fall_tilt, fall_height, time_out

_NUM_ACTUATED = 6
_STAND_KEYFRAME = "stand"


class BipedLocalEnv:
    """
    Single-environment, CPU-only wrapper around BipedSim driven by a
    BipedEnvCfg. Gym-like `reset()`/`step(action)` interface, but N=1
    (no vectorization) - for local sanity-checking and visualization only.
    """

    def __init__(
        self,
        model_path: str = "models/mjcf/biped_warp.xml",
        env_cfg: Optional[BipedEnvCfg] = None,
        seed: Optional[int] = None,
    ):
        self.env_cfg = env_cfg if env_cfg is not None else BipedEnvCfg()
        control_dt = self.env_cfg.sim.decimation * self.env_cfg.sim.timestep

        self.sim = BipedSim(model_path, control_dt=control_dt, suspended=False)
        self.rng = np.random.default_rng(seed)

        self._stand_key_id = mujoco.mj_name2id(
            self.sim.model, mujoco.mjtObj.mjOBJ_KEY, _STAND_KEYFRAME
        )
        if self._stand_key_id < 0:
            raise ValueError(f"Keyframe '{_STAND_KEYFRAME}' not found in model.")
        self._nominal_qpos = self.sim.model.key_qpos[self._stand_key_id].copy()

        # Captures the model's pristine baseline once, before any DR is
        # ever applied - see DomainRandomizer's no-leakage guarantee.
        self._randomizer = DomainRandomizer(self.sim.model)

        self.step_count = 0
        self.previous_action = np.zeros(_NUM_ACTUATED, dtype=np.float64)
        self.command = np.zeros(3, dtype=np.float64)

    def reset(self) -> np.ndarray:
        """Reset to a (possibly domain-randomized, noise-perturbed) stand pose."""
        self.step_count = 0
        self.previous_action = np.zeros(_NUM_ACTUATED, dtype=np.float64)
        self.command = sample_command(self.rng, self.env_cfg.command_cfg)

        # Domain randomization: always resampled from the pristine baseline
        # captured in __init__, never compounding across resets.
        self._randomizer.apply(self.sim.model, self.rng, self.env_cfg.domain_randomization_cfg)

        # Init-state noise on top of the nominal stand-keyframe qpos.
        noise_sample = sample_init_state(self.rng, self.env_cfg.init_noise_cfg)
        noised_qpos = apply_init_state(self._nominal_qpos, noise_sample)

        # mj_resetDataKeyframe first (clears stale warmstart/derived state),
        # then overwrite qpos with the noised pose, then a fresh mj_forward
        # so sensors reflect the perturbed state, not the nominal keyframe.
        mujoco.mj_resetDataKeyframe(self.sim.model, self.sim.data, self._stand_key_id)
        self.sim.data.qpos[:] = noised_qpos
        self.sim.data.qvel[:] = 0.0
        mujoco.mj_forward(self.sim.model, self.sim.data)

        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment one control step.

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float64)
        self.sim.step(action)

        # Resynchronize sensors to the just-advanced state (see module
        # docstring - mj_step's sensors are one step stale otherwise).
        mujoco.mj_forward(self.sim.model, self.sim.data)

        sensors = self.sim.sensors()
        torso_quat_b = sensors["torso_quat"][None, :]
        base_height_b = np.array([sensors["torso_pos"][2]])
        base_linvel_b = sensors["torso_linvel"][None, :]
        action_b = action[None, :]
        prev_action_b = self.previous_action[None, :]
        command_b = self.command[None, :]

        reward = float(
            compute_reward(
                self.env_cfg.reward_cfg,
                torso_quat_b,
                base_linvel_b,
                command_b,
                action_b,
                prev_action_b,
            )[0]
        )

        fall_tilt_flag = bool(fall_tilt(torso_quat_b, self.env_cfg.termination_cfg)[0])
        fall_height_flag = bool(fall_height(base_height_b, self.env_cfg.termination_cfg)[0])
        terminated = fall_tilt_flag or fall_height_flag

        self.step_count += 1
        truncated = bool(
            time_out(np.array([self.step_count]), self.env_cfg.termination_cfg)[0]
        )

        self.previous_action = action.copy()
        obs = self._get_obs()

        info = {
            "reward": reward,
            "fall_tilt": fall_tilt_flag,
            "fall_height": fall_height_flag,
            "terminated": terminated,
            "truncated": truncated,
            "step_count": self.step_count,
            "command": self.command.copy(),
            "base_height": float(base_height_b[0]),
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        sensors = self.sim.sensors()
        return build_actor_obs(sensors, self.previous_action, self.command)

    def get_critic_obs(self) -> np.ndarray:
        """Privileged critic observation (training-only, not used by the visualizer)."""
        sensors = self.sim.sensors()
        com = self.sim.com()
        return build_critic_obs(sensors, self.previous_action, self.command, com)
