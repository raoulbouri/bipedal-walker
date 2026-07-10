"""mjlab_biped: MuJoCo Warp + mjlab biped task package for RL training.

This package provides RL environment configuration for the biped robot
on mjlab (MuJoCo Warp GPU backend) with RSL-RL training via Google Colab.

Phase 6 structure:
- Phase 6.A (done): entity.py — Robot EntityCfg with biped_warp.xml
- Phase 6.B (done): observations.py — Actor/critic observation groups
- Phase 6.C (next): commands.py — Velocity command + domain randomization
- Phase 6.D: rewards.py — Reward + termination functions (stubs)
- Phase 6.E: config.py — Full ManagerBasedRlEnvCfg assembly

For local CPU testing (Phases 0-5 gates), use sim/ package with biped.xml.
For GPU training on Colab (Phase 7+), use mjlab_biped/ with biped_warp.xml.
"""

from .entity import BipedEntityCfg, ActuatorConfig, InitialStateConfig
from .observations import (
    build_actor_obs,
    build_critic_obs,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    ACTOR_TERM_NAMES,
    CRITIC_TERM_NAMES,
    ACTOR_SENSOR_WHITELIST,
)

__all__ = [
    "BipedEntityCfg",
    "ActuatorConfig",
    "InitialStateConfig",
    "build_actor_obs",
    "build_critic_obs",
    "ACTOR_OBS_DIM",
    "CRITIC_OBS_DIM",
    "ACTOR_TERM_NAMES",
    "CRITIC_TERM_NAMES",
    "ACTOR_SENSOR_WHITELIST",
]
