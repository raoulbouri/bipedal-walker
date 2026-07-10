"""
Reward stubs for the biped RL environment (Phase 6.D).

Per CLAUDE.md's Phase 6.D scope, these are deliberately minimal
placeholders — the user finalizes actual reward shaping/curriculum in
Phase 7. Every weight is marked with a TODO(user) comment. Structure is
frozen in `docs/rewards_termination_spec.md`; do not deviate from it here.

This module is pure Python + numpy (no mjlab/torch imports — those are
Colab-only per CLAUDE.md, not in Mac core deps). Every function accepts
and returns arrays with a leading `(num_envs,)` batch dimension.
"""

from dataclasses import dataclass

import numpy as np


def upright(torso_quat: np.ndarray) -> np.ndarray:
    """
    Upright measure derived from a batch of unit quaternions.

    Args:
        torso_quat: (N, 4) array, MuJoCo convention [qw, qx, qy, qz].

    Returns:
        (N,) array: 1 - 2*(qx**2 + qy**2). Range [-1, 1]; +1 = perfectly
        upright, 0 = torso horizontal (90 deg tilt), -1 = upside down.
    """
    q = np.asarray(torso_quat, dtype=np.float64)
    qx = q[..., 1]
    qy = q[..., 2]
    return 1.0 - 2.0 * (qx**2 + qy**2)


@dataclass
class RewardCfg:
    """
    Reward term weights. These are Phase 6.D placeholder defaults — the
    user finalizes actual values in Phase 7 once training begins.
    """

    alive_bonus_weight: float = 1.0  # TODO(user): finalize in Phase 7
    upright_weight: float = 1.0  # TODO(user): finalize in Phase 7
    command_tracking_weight: float = 0.0  # TODO(user): finalize in Phase 7
    control_effort_weight: float = -0.001  # TODO(user): finalize in Phase 7
    action_rate_weight: float = -0.01  # TODO(user): finalize in Phase 7


def alive_bonus(num_envs: int) -> np.ndarray:
    """Flat per-step survival bonus. Returns (num_envs,) array of ones."""
    return np.ones(num_envs)


def upright_term(torso_quat: np.ndarray) -> np.ndarray:
    """Upright reward term. Returns (N,) array, see `upright()`."""
    return upright(torso_quat)


def command_tracking_term(base_linvel: np.ndarray, velocity_command: np.ndarray) -> np.ndarray:
    """
    Negative xy linear-velocity tracking error (yaw-rate tracking deferred).

    Args:
        base_linvel: (N, 3) array, world-frame base linear velocity.
        velocity_command: (N, 3) array, [vx, vy, yaw_rate].

    Returns:
        (N,) array: -norm(base_linvel[:, :2] - velocity_command[:, :2]).
    """
    base_linvel = np.asarray(base_linvel, dtype=np.float64)
    velocity_command = np.asarray(velocity_command, dtype=np.float64)
    return -np.linalg.norm(base_linvel[:, :2] - velocity_command[:, :2], axis=-1)


def control_effort_term(action: np.ndarray) -> np.ndarray:
    """Negative quadratic control-effort penalty. action: (N, 6)."""
    action = np.asarray(action, dtype=np.float64)
    return -np.sum(action**2, axis=-1)


def action_rate_term(action: np.ndarray, previous_action: np.ndarray) -> np.ndarray:
    """Negative quadratic action-rate (anti-vibration) penalty."""
    action = np.asarray(action, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    return -np.sum((action - previous_action) ** 2, axis=-1)


def compute_reward(
    cfg: RewardCfg,
    torso_quat: np.ndarray,
    base_linvel: np.ndarray,
    velocity_command: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    """
    Combine all reward terms into the weighted total reward.

    Args:
        cfg: RewardCfg holding term weights.
        torso_quat: (N, 4) array.
        base_linvel: (N, 3) array.
        velocity_command: (N, 3) array.
        action: (N, 6) array.
        previous_action: (N, 6) array.

    Returns:
        (N,) array, the weighted sum of all reward terms.
    """
    num_envs = np.asarray(torso_quat).shape[0]

    total = (
        cfg.alive_bonus_weight * alive_bonus(num_envs)
        + cfg.upright_weight * upright_term(torso_quat)
        + cfg.command_tracking_weight * command_tracking_term(base_linvel, velocity_command)
        + cfg.control_effort_weight * control_effort_term(action)
        + cfg.action_rate_weight * action_rate_term(action, previous_action)
    )
    return total
