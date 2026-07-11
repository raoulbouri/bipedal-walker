"""
Full environment configuration for the biped RL task (Phase 6.E).

This module assembles the pieces built in Phases 6.A-6.D into a single
`BipedEnvCfg` and provides a local task registry that mirrors the shape
of mjlab's real `register_mjlab_task` API closely enough that Phase 7's
Colab wiring is a thin substitution, not a rewrite.

Per docs/env_assembly_spec.md:
- SceneCfg: num_envs, env_spacing, terrain_type, entity_cfg
- SimCfg: timestep, integrator, decimation
- BipedEnvCfg: bundles scene, sim, reward_cfg, termination_cfg,
  command_cfg, init_noise_cfg, domain_randomization_cfg, episode_length_s
- make_play_env_cfg(): smaller num_envs, domain randomization off
- TASK_REGISTRY / register_mjlab_task / get_task_cfg: local registry

Note: this is a vectorized-shaped config, never actually steppable
locally (no vectorized CPU env exists). It does not import mjlab, torch,
or Gymnasium. Real wiring into mjlab's ManagerBasedRlEnvCfg happens in
Phase 7.
"""

from dataclasses import dataclass, field

from .entity import BipedEntityCfg
from .rewards import RewardCfg
from .terminations import TerminationCfg
from .commands import CommandRangeCfg
from .init_noise import InitNoiseCfg
from .domain_randomization import DomainRandomizationCfg
from .rl_cfg import RunnerCfg


@dataclass
class SceneCfg:
    """Scene configuration: parallel-env layout and terrain."""
    num_envs: int = 4096
    env_spacing: float = 2.0
    terrain_type: str = "plane"
    entity_cfg: BipedEntityCfg = field(default_factory=BipedEntityCfg)


@dataclass
class SimCfg:
    """Physics simulation configuration."""
    timestep: float = 0.002
    integrator: str = "implicit"
    decimation: int = 10


@dataclass
class BipedEnvCfg:
    """
    Full biped RL environment configuration.

    Bundles the scene, sim, and all Phase 6.A-6.D pieces (reward,
    termination, command, init-noise, domain-randomization configs) into
    a single config object, plus the episode length.
    """
    scene: SceneCfg = field(default_factory=SceneCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    reward_cfg: RewardCfg = field(default_factory=RewardCfg)
    termination_cfg: TerminationCfg = field(default_factory=TerminationCfg)
    command_cfg: CommandRangeCfg = field(default_factory=CommandRangeCfg)
    init_noise_cfg: InitNoiseCfg = field(default_factory=InitNoiseCfg)
    domain_randomization_cfg: DomainRandomizationCfg = field(
        default_factory=DomainRandomizationCfg
    )
    episode_length_s: float = 20.0

    def __post_init__(self):
        """Validate the decimation/timestep 50 Hz control-rate invariant."""
        product = self.sim.decimation * self.sim.timestep
        if abs(product - 0.02) > 1e-9:
            raise ValueError(
                f"sim.decimation ({self.sim.decimation}) * sim.timestep "
                f"({self.sim.timestep}) = {product} must equal 0.02 "
                f"(the frozen 50 Hz control rate)"
            )


def make_play_env_cfg() -> BipedEnvCfg:
    """
    Build the play/eval variant of the env config: fewer parallel envs,
    domain randomization explicitly disabled (self-documenting, even
    though False is already DomainRandomizationCfg's own default).
    """
    return BipedEnvCfg(
        scene=SceneCfg(num_envs=4),
        domain_randomization_cfg=DomainRandomizationCfg(enabled=False),
    )


# --- Local task registry -----------------------------------------------
#
# Since real mjlab.register_mjlab_task doesn't exist locally (mjlab is a
# Colab-only dependency), this is a minimal substitute with the same
# essential shape, so Phase 7's Colab wiring is a thin substitution.

TASK_REGISTRY: dict = {}


def register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg=None) -> None:
    """
    Register a task by id.

    Raises ValueError if task_id is already registered (no silent
    overwrite - this would be a genuine bug if it happened at Colab-
    registration time too).
    """
    if task_id in TASK_REGISTRY:
        raise ValueError(f"Task '{task_id}' is already registered.")
    TASK_REGISTRY[task_id] = {
        "env_cfg": env_cfg,
        "play_env_cfg": play_env_cfg,
        "rl_cfg": rl_cfg,
    }


def get_task_cfg(task_id):
    """
    Look up a registered task's cfg dict.

    Raises KeyError with a clear message (including the list of
    registered ids) if not found.
    """
    if task_id not in TASK_REGISTRY:
        raise KeyError(
            f"Task '{task_id}' not found. "
            f"Registered tasks: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]


# Register the single frozen task at module import time.
register_mjlab_task(
    "Mjlab-Biped-Balance-v0",
    env_cfg=BipedEnvCfg(),
    play_env_cfg=make_play_env_cfg(),
    rl_cfg=RunnerCfg(),
)
