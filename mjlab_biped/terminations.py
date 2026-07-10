"""
Termination stubs for the biped RL environment (Phase 6.D).

Per CLAUDE.md's Phase 6.D scope and `docs/rewards_termination_spec.md`:
fall detection (tilt OR height) plus episode time-out truncation. The
tilt/height thresholds are physically-motivated defaults from Phase 0-2
baselines, not arbitrary placeholders; `max_episode_steps` is episode-
length dependent and left for Phase 7 tuning.

This module is pure Python + numpy (no mjlab/torch imports — those are
Colab-only per CLAUDE.md, not in Mac core deps). Every function accepts
and returns arrays with a leading `(num_envs,)` batch dimension.
"""

from dataclasses import dataclass

import numpy as np

from mjlab_biped.rewards import upright


@dataclass
class TerminationCfg:
    """Termination thresholds."""

    tilt_threshold: float = 0.5
    height_threshold: float = 0.15
    max_episode_steps: int = 1000  # TODO(user): finalize in Phase 7


def fall_tilt(torso_quat: np.ndarray, cfg: TerminationCfg) -> np.ndarray:
    """
    True where the torso has tilted beyond `cfg.tilt_threshold`.

    Args:
        torso_quat: (N, 4) array, MuJoCo convention [qw, qx, qy, qz].
        cfg: TerminationCfg.

    Returns:
        (N,) boolean array: upright(torso_quat) < cfg.tilt_threshold.
    """
    return upright(torso_quat) < cfg.tilt_threshold


def fall_height(base_height: np.ndarray, cfg: TerminationCfg) -> np.ndarray:
    """
    True where the base height has dropped below `cfg.height_threshold`.

    Args:
        base_height: (N,) array.
        cfg: TerminationCfg.

    Returns:
        (N,) boolean array: base_height < cfg.height_threshold.
    """
    base_height = np.asarray(base_height, dtype=np.float64)
    return base_height < cfg.height_threshold


def fall(torso_quat: np.ndarray, base_height: np.ndarray, cfg: TerminationCfg) -> np.ndarray:
    """Combined fall termination: fall_tilt OR fall_height."""
    return fall_tilt(torso_quat, cfg) | fall_height(base_height, cfg)


def time_out(step_count: np.ndarray, cfg: TerminationCfg) -> np.ndarray:
    """
    Episode truncation flag (not a fall).

    Args:
        step_count: (N,) int array, current step count per env.
        cfg: TerminationCfg.

    Returns:
        (N,) boolean array: step_count >= cfg.max_episode_steps.
    """
    step_count = np.asarray(step_count)
    return step_count >= cfg.max_episode_steps
