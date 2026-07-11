"""mjlab_biped: MuJoCo Warp + mjlab biped task package for RL training.

This package provides RL environment configuration for the biped robot
on mjlab (MuJoCo Warp GPU backend) with RSL-RL training via Google Colab.

Phase 6 structure:
- Phase 6.A (done): entity.py — Robot EntityCfg with biped_warp.xml
- Phase 6.B (done): observations.py — Actor/critic observation groups
- Phase 6.C (done): commands.py, init_noise.py, domain_randomization.py —
  velocity command sampler, init-state noise, domain randomization
- Phase 6.D (done): rewards.py, terminations.py — Reward + termination
  functions (stubs)
- Phase 6.E (done): config.py — Full env config assembly + local task
  registry

Also included: local_env.py — BipedLocalEnv, a single-environment, CPU-only
wrapper (not part of the mjlab-shaped registry) for local visual
sanity-checking of the assembled obs/reward/termination/command/noise
pieces on macOS via `scripts/visualize_env.py`, ahead of the real Colab
training run.

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
    ObsHistory,
    ACTOR_HISTORY_LEN,
    STACKED_ACTOR_OBS_DIM,
)
from .commands import CommandRangeCfg, sample_command
from .init_noise import InitNoiseCfg, sample_init_state, apply_init_state
from .domain_randomization import DomainRandomizationCfg, DomainRandomizer
from .rl_cfg import RunnerCfg, ModelCfg, AlgorithmCfg, ACTION_DIM
from .rewards import (
    upright,
    RewardCfg,
    alive_bonus,
    upright_term,
    command_tracking_term,
    control_effort_term,
    action_rate_term,
    compute_reward,
)
from .terminations import (
    TerminationCfg,
    fall_tilt,
    fall_height,
    fall,
    time_out,
)
from .config import (
    SceneCfg,
    SimCfg,
    BipedEnvCfg,
    make_play_env_cfg,
    TASK_REGISTRY,
    register_mjlab_task,
    get_task_cfg,
)
# BipedLocalEnv (local_env.py) is the macOS-only, single-env visualization
# companion (see its own module docstring) - it imports `sim.biped_sim`,
# the top-level `sim/` package, which is deliberately NOT part of the
# minimal Colab training bundle (docs/colab_upload_manifest.md). Import it
# lazily/optionally so `import mjlab_biped` still succeeds on Colab, where
# only `mjlab_biped/` itself is present. On the Mac, where `sim/` sits
# alongside `mjlab_biped/`, this import succeeds normally.
try:
    from .local_env import BipedLocalEnv
except ModuleNotFoundError:
    BipedLocalEnv = None

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
    "ObsHistory",
    "ACTOR_HISTORY_LEN",
    "STACKED_ACTOR_OBS_DIM",
    "CommandRangeCfg",
    "sample_command",
    "InitNoiseCfg",
    "sample_init_state",
    "apply_init_state",
    "DomainRandomizationCfg",
    "DomainRandomizer",
    "RunnerCfg",
    "ModelCfg",
    "AlgorithmCfg",
    "ACTION_DIM",
    "upright",
    "RewardCfg",
    "alive_bonus",
    "upright_term",
    "command_tracking_term",
    "control_effort_term",
    "action_rate_term",
    "compute_reward",
    "TerminationCfg",
    "fall_tilt",
    "fall_height",
    "fall",
    "time_out",
    "SceneCfg",
    "SimCfg",
    "BipedEnvCfg",
    "make_play_env_cfg",
    "TASK_REGISTRY",
    "register_mjlab_task",
    "get_task_cfg",
    "BipedLocalEnv",
]
