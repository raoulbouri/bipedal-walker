"""mjlab_biped: MuJoCo Warp + mjlab biped task package for RL training.

This package provides RL environment configuration for the biped robot
on mjlab (MuJoCo Warp GPU backend) with RSL-RL training via Google Colab.

Phase 6 structure:
- Phase 6.A (current): entity.py — Robot EntityCfg with biped_warp.xml
- Phase 6.B (next): observations.py — Actor/critic observation groups
- Phase 6.C: commands.py — Velocity command + domain randomization
- Phase 6.D: rewards.py — Reward + termination functions (stubs)
- Phase 6.E: config.py — Full ManagerBasedRlEnvCfg assembly

For local CPU testing (Phases 0-5 gates), use sim/ package with biped.xml.
For GPU training on Colab (Phase 7+), use mjlab_biped/ with biped_warp.xml.
"""

from .entity import BipedEntityCfg, ActuatorConfig, InitialStateConfig

__all__ = [
    "BipedEntityCfg",
    "ActuatorConfig",
    "InitialStateConfig",
]
