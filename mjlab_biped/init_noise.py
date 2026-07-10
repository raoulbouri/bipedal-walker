"""
Init-state randomization for the biped RL environment (Phase 6.C).

Samples small per-episode perturbations on top of the `stand` keyframe's
nominal pose (base at [0, 0, 0.2030], identity orientation, all 8 hinge
joints at 0 rad), per the frozen spec in `docs/events_spec.md`. The two
passive foot joints (`foot_l`/`foot_r`) are never randomized — their
resting angle is a physical consequence of ground contact, not a
controllable initial condition.

Pure Python + numpy (no mjlab/torch imports — Colab-only per CLAUDE.md).
"""

from dataclasses import dataclass

import numpy as np

# qpos layout for the 15-dim floating-base + 8-hinge model:
#   [0:3]   base xyz
#   [3:7]   base quaternion [w, x, y, z]
#   [7:13]  6 actuated joints (hip_roll_l, knee_l, ankle_l, hip_roll_r, knee_r, ankle_r)
#   [13:15] 2 passive foot joints (foot_l, foot_r) — never randomized
NUM_ACTUATED_JOINTS = 6


@dataclass
class InitNoiseCfg:
    """Init-state randomization magnitudes."""

    joint_angle_noise: float = 0.02  # rad, symmetric range half-width
    base_xy_noise: float = 0.01  # m
    base_z_noise: float = 0.003  # m
    base_tilt_noise: float = 0.05  # rad, max tilt magnitude


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2, both in MuJoCo [w, x, y, z] convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def sample_init_state(rng: np.random.Generator, cfg: InitNoiseCfg) -> dict:
    """
    Sample an init-state noise perturbation.

    Args:
        rng: explicit numpy Generator, owned by the caller.
        cfg: noise magnitudes.

    Returns:
        dict with keys:
            "joint_angles": (6,) uniform in [-joint_angle_noise, +joint_angle_noise]
                per actuated joint.
            "base_xy": (2,) uniform in [-base_xy_noise, +base_xy_noise].
            "base_z": scalar, uniform in [-base_z_noise, +base_z_noise].
            "base_quat_offset": (4,) unit quaternion for a random small-angle
                rotation: tilt angle ~ U(0, base_tilt_noise) about a uniformly
                random 3D unit axis.
    """
    joint_angles = rng.uniform(
        -cfg.joint_angle_noise, cfg.joint_angle_noise, size=NUM_ACTUATED_JOINTS
    )
    base_xy = rng.uniform(-cfg.base_xy_noise, cfg.base_xy_noise, size=2)
    base_z = rng.uniform(-cfg.base_z_noise, cfg.base_z_noise)

    tilt_angle = rng.uniform(0.0, cfg.base_tilt_noise)
    axis = rng.normal(size=3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        axis = np.array([1.0, 0.0, 0.0])
        axis_norm = 1.0
    axis = axis / axis_norm

    half = tilt_angle / 2.0
    base_quat_offset = np.array(
        [np.cos(half), *(axis * np.sin(half))], dtype=np.float64
    )

    return {
        "joint_angles": joint_angles,
        "base_xy": base_xy,
        "base_z": base_z,
        "base_quat_offset": base_quat_offset,
    }


def apply_init_state(base_qpos: np.ndarray, sample: dict) -> np.ndarray:
    """
    Apply a sampled init-state perturbation to the nominal `stand` qpos.

    Args:
        base_qpos: nominal 15-dim keyframe qpos
            ([0, 0, 0.2030, 1, 0, 0, 0, <8 zeros>]).
        sample: dict as returned by `sample_init_state`.

    Returns:
        A new (15,) numpy array: perturbed qpos. Base xy/z noise is added
        to qpos[0:3]; base orientation is composed via quaternion
        multiplication (base_quat_offset * nominal_quat) into qpos[3:7];
        joint angle noise is added to qpos[7:13] (the 6 actuated joints);
        qpos[13:15] (passive foot joints) is left unchanged.
    """
    qpos = np.array(base_qpos, dtype=np.float64, copy=True)

    qpos[0:2] += sample["base_xy"]
    qpos[2] += sample["base_z"]

    nominal_quat = qpos[3:7]
    qpos[3:7] = _quat_multiply(sample["base_quat_offset"], nominal_quat)

    qpos[7:13] += sample["joint_angles"]

    # qpos[13:15] (foot_l, foot_r) intentionally left unchanged.

    return qpos
