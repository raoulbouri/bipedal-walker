"""
Observation manager for the biped RL environment (Phase 6.B).

Builds the actor (deployment-available) and critic (privileged, training-
only) observation vectors per the frozen spec in `docs/observation_spec.md`.
Do not modify term order/dims here without updating that spec doc and
bumping a spec version — the ordering is a golden-tested invariant.

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
    "projected_gravity",
    "gyro",
    "foot_touch",
    "previous_action",
    "velocity_command",
)

CRITIC_TERM_NAMES = ACTOR_TERM_NAMES + (
    "accelerometer",
    "base_linvel",
    "base_height",
    "com",
)

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------

ACTOR_OBS_DIM = 29
CRITIC_OBS_DIM = 39

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
        "torso_quat",  # consumed only to derive projected_gravity, never exposed raw
        "torso_gyro",
        "touch_l",
        "touch_r",
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
    Build the 29-dim actor observation vector.

    Reads only the whitelisted sensor keys from `sensors` (per
    ACTOR_SENSOR_WHITELIST) — never `torso_acc`, `torso_pos`,
    `torso_linvel`, `torso_angvel`, and never computes CoM.

    Args:
        sensors: dict of sensor_name -> np.ndarray, as returned by
            BipedSim.sensors().
        previous_action: 6-element array, the action applied last step.
        velocity_command: 3-element array, [vx, vy, yaw_rate].

    Returns:
        A (29,) numpy array in the frozen term order.
    """
    joint_pos_rel = np.concatenate([np.atleast_1d(sensors[name]) for name in _POS_SENSOR_NAMES])
    joint_vel_rel = np.concatenate([np.atleast_1d(sensors[name]) for name in _VEL_SENSOR_NAMES])
    projected_gravity = project_gravity(sensors["torso_quat"])
    gyro = np.atleast_1d(sensors["torso_gyro"])
    foot_touch = np.concatenate([np.atleast_1d(sensors["touch_l"]), np.atleast_1d(sensors["touch_r"])])
    prev_action = np.atleast_1d(previous_action).astype(np.float64)
    vel_command = np.atleast_1d(velocity_command).astype(np.float64)

    obs = np.concatenate(
        [
            joint_pos_rel,
            joint_vel_rel,
            projected_gravity,
            gyro,
            foot_touch,
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

    Equals build_actor_obs(...) (29 dims) concatenated with the privileged
    terms: accelerometer, base_linvel, base_height, com (10 dims).

    Args:
        sensors: dict of sensor_name -> np.ndarray, as returned by
            BipedSim.sensors(). Must contain torso_acc, torso_linvel,
            torso_pos in addition to the actor whitelist.
        previous_action: 6-element array.
        velocity_command: 3-element array.
        com: 3-element array, whole-body center of mass (world frame),
            e.g. from BipedSim.com().

    Returns:
        A (39,) numpy array in the frozen term order.
    """
    actor_obs = build_actor_obs(sensors, previous_action, velocity_command)

    accelerometer = np.atleast_1d(sensors["torso_acc"]).astype(np.float64)
    base_linvel = np.atleast_1d(sensors["torso_linvel"]).astype(np.float64)
    base_height = np.atleast_1d(sensors["torso_pos"])[2:3].astype(np.float64)
    com_arr = np.atleast_1d(com).astype(np.float64)

    obs = np.concatenate([actor_obs, accelerometer, base_linvel, base_height, com_arr])

    if obs.shape != (CRITIC_OBS_DIM,):
        raise ValueError(f"critic obs shape mismatch: got {obs.shape}, expected ({CRITIC_OBS_DIM},)")

    return obs
