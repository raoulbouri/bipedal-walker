"""
Recovery-progress reward + fallen-pose reset parameters (Phase 7.R.3).

Pure Python + numpy (no mjlab/torch imports -- Colab-only per CLAUDE.md,
not in Mac core deps). See mjlab_biped/mjlab_task.py's
`recovery_progress_fn` and `reset_biped_recovery_mix` for the real mjlab
translation, which imports the frozen constants below directly so the
distribution parameters used in real training are the exact same numbers
tested here (single source of truth, matching the rest of this package's
convention).

**Design note on why this is a gated, opt-in addition, not always-on:**
the existing training reset is always `reset_scene_to_default` (always
stand pose); recovery needs a genuinely different reset -- some fraction
of episodes start from a fully randomized fallen/toppled pose. Mixing
these lets one training run see both regimes without destabilizing the
balance behavior from Phase 7.R.1/7.R.2, but ONLY if the reward and
termination logic account for the new "started on the ground" state (see
`recovery_progress_term` and CLAUDE.md's Phase 7.R.3 termination note --
a fallen reset would otherwise instantly re-trigger fall_tilt/fall_height
on the very next check, exactly the events={} bug from 2026-07-11).
"""

from dataclasses import dataclass, field

import numpy as np

# Frozen fallen-pose reset parameters -- imported directly by
# mjlab_task.py's real event function (mjlab's reset_root_state_uniform
# `pose_range=` / reset_joints_by_offset `position_range=` args) so the
# real training distribution matches exactly what's tested here.
FALLEN_POSE_RANGE = {
    "roll": (-3.14159265, 3.14159265),
    "pitch": (-3.14159265, 3.14159265),
    "yaw": (-3.14159265, 3.14159265),
    # Small positive z offset above the stand-keyframe height so a
    # toppled orientation doesn't clip through the floor at reset --
    # physics settles the exact contact within the first few steps
    # regardless.
    "z": (0.0, 0.05),
}
FALLEN_JOINT_POSITION_RANGE = (-0.3, 0.3)  # rad, offset from default (0) joint angles
RECOVERY_RESET_PROB = 0.3  # fraction of resets that start fallen, not standing


@dataclass
class RecoveryCfg:
    """Recovery-progress reward weight. Off (weight=0 equivalent handled
    by the caller not including this term) unless the recovery curriculum
    stage is explicitly enabled -- see `make_biped_env_cfg`'s
    `recovery_enabled` flag in mjlab_task.py."""

    weight: float = 0.5  # TODO(user): finalize once recovery curriculum is enabled


def recovery_progress_term(
    height: np.ndarray,
    upright_value: np.ndarray,
    base_linvel_z: np.ndarray,
    height_threshold: float,
    tilt_threshold: float,
) -> np.ndarray:
    """
    Reward upward progress while the robot is still "fallen" -- gated to
    exactly the same fallen definition TerminationCfg already uses
    (height < height_threshold OR upright < tilt_threshold), so it adds
    signal for getting up WITHOUT double-counting the existing
    upright/alive terms once the robot is reasonably upright/tall again.

    Args:
        height: (N,) array, base z position.
        upright_value: (N,) array, see rewards.upright() -- same
            1 - 2*(qx^2+qy^2) closed form, range [-1, 1].
        base_linvel_z: (N,) array, world-frame vertical base velocity
            (this IS d(height)/dt exactly, no finite-difference/previous-
            state tracking needed).
        height_threshold: same value as TerminationCfg.height_threshold.
        tilt_threshold: same value as TerminationCfg.tilt_threshold.

    Returns:
        (N,) array: relu(base_linvel_z) + relu(upright_value), zeroed
        outside the "fallen" regime. Both terms are relu'd (clipped at 0)
        so downward motion / being upside-down never turns this into a
        penalty -- that's already handled by the existing upright/alive
        terms and termination; this term ONLY ever adds a bonus for
        genuinely rising/righting while down.
    """
    height = np.asarray(height, dtype=np.float64)
    upright_value = np.asarray(upright_value, dtype=np.float64)
    base_linvel_z = np.asarray(base_linvel_z, dtype=np.float64)

    in_recovery = (height < height_threshold) | (upright_value < tilt_threshold)
    progress = np.clip(base_linvel_z, 0.0, None) + np.clip(upright_value, 0.0, None)
    return np.where(in_recovery, progress, 0.0)


@dataclass
class FallenPoseSampleCfg:
    """Mirrors FALLEN_POSE_RANGE/FALLEN_JOINT_POSITION_RANGE as a
    dataclass for the pure-Python sampler below, so its distribution can
    be tested without mjlab installed."""

    roll_range: tuple = FALLEN_POSE_RANGE["roll"]
    pitch_range: tuple = FALLEN_POSE_RANGE["pitch"]
    yaw_range: tuple = FALLEN_POSE_RANGE["yaw"]
    z_offset_range: tuple = FALLEN_POSE_RANGE["z"]
    joint_angle_range: tuple = FALLEN_JOINT_POSITION_RANGE


def sample_fallen_pose(rng: np.random.Generator, cfg: FallenPoseSampleCfg) -> dict:
    """
    Sample one fallen-pose perturbation, using the exact same range
    parameters mjlab_task.py's real event passes to mjlab's
    reset_root_state_uniform/reset_joints_by_offset.

    Args:
        rng: explicit numpy Generator, owned by the caller (reproducible
            under a fixed seed).
        cfg: range magnitudes.

    Returns:
        dict with keys "roll", "pitch", "yaw" (radians, orientation
        offset composed onto the default upright pose), "z_offset"
        (meters, added to the default stand height), "joint_angles"
        ((6,) array, radians, offset from the default 0 pose).
    """
    return {
        "roll": rng.uniform(*cfg.roll_range),
        "pitch": rng.uniform(*cfg.pitch_range),
        "yaw": rng.uniform(*cfg.yaw_range),
        "z_offset": rng.uniform(*cfg.z_offset_range),
        "joint_angles": rng.uniform(
            cfg.joint_angle_range[0], cfg.joint_angle_range[1], size=6
        ),
    }
