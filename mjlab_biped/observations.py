"""
Observation manager for the biped RL environment (Phase 7.pre, obs spec v1).

Builds the actor (deployment-available) and critic (privileged, training-
only) observation vectors per the frozen spec in `docs/observation_spec.md`.
Do not modify term order/dims here without updating that spec doc and
bumping a spec version — the ordering is a golden-tested invariant.

v1 asymmetry (see docs/observation_spec.md and MEMORY.md's 2026-07-11
"Phase 7 plan v4" entry for full rationale): the real robot has no
orientation estimate (no EKF/sensor fusion) and no foot-contact hardware,
so the actor no longer computes `projected_gravity` or `foot_touch` at
all — those terms moved to the critic (privileged, training-only) group.
Because a single frame's raw gyro (angular *velocity*, not angle) can't
recover orientation, the actor's per-frame 24-dim observation is stacked
into a 5-frame history (`ObsHistory`) so the policy can integrate gyro
itself over time.

This module is pure Python + numpy (no mjlab/torch imports — those are
Colab-only per CLAUDE.md, not in Mac core deps).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Term ordering (golden, per docs/observation_spec.md)
# ---------------------------------------------------------------------------

ACTOR_TERM_NAMES = (
    "joint_pos_rel",
    "joint_vel_rel",
    "gyro",
    "previous_action",
    "velocity_command",
)

CRITIC_TERM_NAMES = ACTOR_TERM_NAMES + (
    "projected_gravity",
    "foot_touch",
    "accelerometer",
    "base_linvel",
    "base_height",
    "com",
)

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------

ACTOR_OBS_DIM = 24
CRITIC_OBS_DIM = 39

# History stacking (v1): the actor sees ACTOR_HISTORY_LEN raw per-frame
# observations concatenated oldest -> newest, so it can integrate gyro
# into an implicit orientation estimate itself.
ACTOR_HISTORY_LEN = 5
STACKED_ACTOR_OBS_DIM = ACTOR_OBS_DIM * ACTOR_HISTORY_LEN

# ---------------------------------------------------------------------------
# Sensor name ordering for the actuated joints (spec table term 1/2)
# ---------------------------------------------------------------------------

_ACTUATED_JOINT_ORDER = ("hip_roll_l", "knee_l", "ankle_l", "hip_roll_r", "knee_r", "ankle_r")

_POS_SENSOR_NAMES = tuple(f"pos_{name}" for name in _ACTUATED_JOINT_ORDER)
_VEL_SENSOR_NAMES = tuple(f"vel_{name}" for name in _ACTUATED_JOINT_ORDER)

# ---------------------------------------------------------------------------
# Deployment whitelist (sensor names the actor path may read)
# ---------------------------------------------------------------------------

ACTOR_SENSOR_WHITELIST = frozenset(
    _POS_SENSOR_NAMES
    + _VEL_SENSOR_NAMES
    + (
        "torso_gyro",
    )
)


def project_gravity(torso_quat: np.ndarray) -> np.ndarray:
    """
    Rotate the world gravity direction [0, 0, -1] into the torso body frame.

    Args:
        torso_quat: MuJoCo-convention quaternion [w, x, y, z] describing the
            torso's orientation (world -> body via its conjugate/inverse).

    Returns:
        A 3-element numpy array: gravity direction expressed in the torso's
        local frame.
    """
    q = np.asarray(torso_quat, dtype=np.float64)
    w, x, y, z = q
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("torso_quat has zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Inverse (= conjugate, for a unit quaternion) rotates world -> body.
    w_inv, x_inv, y_inv, z_inv = w, -x, -y, -z

    gravity_world = np.array([0.0, 0.0, -1.0])

    # Rotate vector v by quaternion q_inv: v' = q_inv * v * q_inv_conjugate
    # Using the standard quaternion-vector rotation formula:
    # v' = v + 2*w*(qv x v) + 2*(qv x (qv x v)), where qv = [x, y, z]
    qv = np.array([x_inv, y_inv, z_inv])
    t = 2.0 * np.cross(qv, gravity_world)
    rotated = gravity_world + w_inv * t + np.cross(qv, t)

    return rotated


def build_actor_obs(
    sensors: dict,
    previous_action: np.ndarray,
    velocity_command: np.ndarray,
) -> np.ndarray:
    """
    Build the 24-dim single-frame actor observation vector.

    Reads only the whitelisted sensor keys from `sensors` (per
    ACTOR_SENSOR_WHITELIST) — never `torso_quat`, `touch_l`, `touch_r`,
    `torso_acc`, `torso_pos`, `torso_linvel`, `torso_angvel`, and never
    computes CoM. Callers driving a real episode should stack this
    per-frame output via `ObsHistory` before feeding it to the policy.

    Args:
        sensors: dict of sensor_name -> np.ndarray, as returned by
            BipedSim.sensors().
        previous_action: 6-element array, the action applied last step.
        velocity_command: 3-element array, [vx, vy, yaw_rate].

    Returns:
        A (24,) numpy array in the frozen term order.
    """
    joint_pos_rel = np.concatenate([np.atleast_1d(sensors[name]) for name in _POS_SENSOR_NAMES])
    joint_vel_rel = np.concatenate([np.atleast_1d(sensors[name]) for name in _VEL_SENSOR_NAMES])
    gyro = np.atleast_1d(sensors["torso_gyro"])
    prev_action = np.atleast_1d(previous_action).astype(np.float64)
    vel_command = np.atleast_1d(velocity_command).astype(np.float64)

    obs = np.concatenate(
        [
            joint_pos_rel,
            joint_vel_rel,
            gyro,
            prev_action,
            vel_command,
        ]
    ).astype(np.float64)

    if obs.shape != (ACTOR_OBS_DIM,):
        raise ValueError(f"actor obs shape mismatch: got {obs.shape}, expected ({ACTOR_OBS_DIM},)")

    return obs


def build_critic_obs(
    sensors: dict,
    previous_action: np.ndarray,
    velocity_command: np.ndarray,
    com: np.ndarray,
) -> np.ndarray:
    """
    Build the 39-dim critic observation vector.

    Equals build_actor_obs(...) (24 dims, single frame) concatenated with
    the privileged terms: projected_gravity, foot_touch, accelerometer,
    base_linvel, base_height, com (15 dims).

    Args:
        sensors: dict of sensor_name -> np.ndarray, as returned by
            BipedSim.sensors(). Must contain torso_quat, touch_l, touch_r,
            torso_acc, torso_linvel, torso_pos in addition to the actor
            whitelist.
        previous_action: 6-element array.
        velocity_command: 3-element array.
        com: 3-element array, whole-body center of mass (world frame),
            e.g. from BipedSim.com().

    Returns:
        A (39,) numpy array in the frozen term order.
    """
    actor_obs = build_actor_obs(sensors, previous_action, velocity_command)

    projected_gravity = project_gravity(sensors["torso_quat"])
    foot_touch = np.concatenate([np.atleast_1d(sensors["touch_l"]), np.atleast_1d(sensors["touch_r"])])
    accelerometer = np.atleast_1d(sensors["torso_acc"]).astype(np.float64)
    base_linvel = np.atleast_1d(sensors["torso_linvel"]).astype(np.float64)
    base_height = np.atleast_1d(sensors["torso_pos"])[2:3].astype(np.float64)
    com_arr = np.atleast_1d(com).astype(np.float64)

    obs = np.concatenate(
        [actor_obs, projected_gravity, foot_touch, accelerometer, base_linvel, base_height, com_arr]
    )

    if obs.shape != (CRITIC_OBS_DIM,):
        raise ValueError(f"critic obs shape mismatch: got {obs.shape}, expected ({CRITIC_OBS_DIM},)")

    return obs


class ObsHistory:
    """Fixed-length history stack of raw per-frame actor observations.
    Pure numpy ring buffer, oldest->newest concatenation. NOT a filter -
    the policy network does its own temporal integration over this stack.
    See docs/observation_spec.md and CLAUDE.md Phase 7 v4 for why."""

    def __init__(self, obs_dim: int = ACTOR_OBS_DIM, history_len: int = ACTOR_HISTORY_LEN):
        self.obs_dim = obs_dim
        self.history_len = history_len
        self._buffer = np.zeros((history_len, obs_dim), dtype=np.float64)

    def reset(self, first_obs: np.ndarray) -> np.ndarray:
        """Fill the entire buffer with copies of first_obs (called on episode reset).
        Returns the (history_len * obs_dim,) stacked observation."""
        first_obs = np.asarray(first_obs, dtype=np.float64)
        self._buffer[:] = first_obs[None, :]
        return self._stacked()

    def push(self, obs: np.ndarray) -> np.ndarray:
        """Shift the buffer left by one frame and append obs as the newest.
        Returns the (history_len * obs_dim,) stacked observation."""
        obs = np.asarray(obs, dtype=np.float64)
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = obs
        return self._stacked()

    def _stacked(self) -> np.ndarray:
        return self._buffer.reshape(-1).copy()
