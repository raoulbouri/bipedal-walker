"""
Velocity command sampler for the biped RL environment (Phase 6.C).

Samples a per-episode velocity command (vx, vy, yaw_rate) from independent
uniform ranges, per the frozen spec in `docs/events_spec.md`. Defaults to
the zero-command balance-first gate ({0} for all three axes); widened
later for the Phase 7 walking curriculum via config only, not code.

Pure Python + numpy (no mjlab/torch imports — those are Colab-only per
CLAUDE.md, not in Mac core deps). RNG is always an explicit
`numpy.random.Generator` argument, per the Phase 1 harness convention —
no module-level or hidden global RNG state.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CommandRangeCfg:
    """Per-axis uniform sampling ranges for the velocity command."""

    vx_range: Tuple[float, float] = (0.0, 0.0)
    vy_range: Tuple[float, float] = (0.0, 0.0)
    yaw_rate_range: Tuple[float, float] = (0.0, 0.0)


# Phase 7.W.1 (2026-07-14): first walking-curriculum stage, narrow
# forward-only range. Single source of truth for the walk task's command
# range -- mjlab_task.py imports this into mjlab's real
# UniformVelocityCommandCfg.Ranges rather than hardcoding numbers there.
# CommandRangeCfg()'s own bare default stays all-zero (relied on by
# local_env.py, config.py's EnvCfg default, and test_events.py as the
# balance-first gate) -- this is a separate named preset, not a change to
# that default.
WALK_STAGE_1_RANGE = CommandRangeCfg(
    vx_range=(0.0, 0.3), vy_range=(0.0, 0.0), yaw_rate_range=(0.0, 0.0)
)


def sample_command(rng: np.random.Generator, cfg: CommandRangeCfg) -> np.ndarray:
    """
    Sample a velocity command [vx, vy, yaw_rate].

    Each axis is drawn independently via `rng.uniform(low, high)`. A
    zero-width range always returns exactly that value (no drift).

    Args:
        rng: explicit numpy Generator, owned by the caller.
        cfg: per-axis (low, high) ranges.

    Returns:
        A (3,) numpy array: [vx, vy, yaw_rate].
    """
    vx = rng.uniform(cfg.vx_range[0], cfg.vx_range[1])
    vy = rng.uniform(cfg.vy_range[0], cfg.vy_range[1])
    yaw_rate = rng.uniform(cfg.yaw_rate_range[0], cfg.yaw_rate_range[1])
    return np.array([vx, vy, yaw_rate], dtype=np.float64)
