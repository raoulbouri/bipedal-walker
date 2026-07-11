"""
RSL-RL PPO runner configuration for the biped RL task (Phase 7.B).

This module mirrors the REAL `rsl_rl`/IsaacLab 3-tier config shape
(`RslRlOnPolicyRunnerCfg` nesting `RslRlPpoActorCriticCfg` and
`RslRlPpoAlgorithmCfg` — see IsaacLab's `isaaclab_rl.rsl_rl.rl_cfg` docs,
verified 2026-07-11) as three plain Python dataclasses. It does NOT import
the real `rsl_rl` or `torch` packages — those are Colab-only dependencies
per CLAUDE.md, never part of the lean Mac core deps. Mirroring the real
nesting now means that wiring this into actual `rsl_rl` on Colab in a
later phase (7.C+) is a straight rename/re-parent of these dataclasses
into the real `RslRlOnPolicyRunnerCfg`/`RslRlPpoActorCriticCfg`/
`RslRlPpoAlgorithmCfg` types, not a redesign.

Per CLAUDE.md's Sub-task 7.B: this is config only. No real rsl_rl wiring,
no Colab work, no actual training happens here (that is Phase 7.C). All
frozen starting values below are sourced from IsaacLab's flat-terrain
velocity-task PPO config, cross-checked against the real rsl_rl/IsaacLab
docs, and are explicitly marked as tunable via `# TODO(user): tune`
comments on every field.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# Not a real rsl_rl field — a local sanity anchor for the dimension
# consistency check (see tests/test_rl_cfg.py), matching the 6 position
# actuators frozen in mjlab_biped/entity.py's ActuatorConfig.
ACTION_DIM = 6


@dataclass
class PolicyCfg:
    """Mirrors `RslRlPpoActorCriticCfg`: actor/critic network + noise config."""

    actor_hidden_dims: List[int] = field(
        default_factory=lambda: [512, 256, 128]
    )  # TODO(user): tune
    critic_hidden_dims: List[int] = field(
        default_factory=lambda: [512, 256, 128]
    )  # TODO(user): tune
    activation: str = "elu"  # TODO(user): tune
    actor_obs_normalization: bool = True  # TODO(user): tune
    critic_obs_normalization: bool = True  # TODO(user): tune
    init_noise_std: float = 1.0  # TODO(user): tune


@dataclass
class AlgorithmCfg:
    """Mirrors `RslRlPpoAlgorithmCfg`: PPO hyperparameters."""

    num_learning_epochs: int = 5  # TODO(user): tune
    num_mini_batches: int = 4  # TODO(user): tune
    learning_rate: float = 1.0e-3  # TODO(user): tune
    schedule: str = "adaptive"  # TODO(user): tune  (KL-based)
    gamma: float = 0.99  # TODO(user): tune
    lam: float = 0.95  # TODO(user): tune  (GAE lambda)
    entropy_coef: float = 0.005  # TODO(user): tune
    desired_kl: float = 0.01  # TODO(user): tune
    max_grad_norm: float = 1.0  # TODO(user): tune
    value_loss_coef: float = 1.0  # TODO(user): tune
    use_clipped_value_loss: bool = True  # TODO(user): tune
    clip_param: float = 0.2  # TODO(user): tune


@dataclass
class RunnerCfg:
    """
    Mirrors `RslRlOnPolicyRunnerCfg`: top-level runner config, nesting
    `policy: PolicyCfg` and `algorithm: AlgorithmCfg`.

    `obs_groups` wires to the Phase 6.B observation manager's real group
    names ("actor" and "critic", see mjlab_biped/observations.py's
    ACTOR_TERM_NAMES/CRITIC_TERM_NAMES) rather than being redefined here.
    """

    num_steps_per_env: int = 24  # TODO(user): tune
    max_iterations: int = 1500  # TODO(user): tune (budget; gate on eval, not on exhausting it)
    save_interval: int = 50  # TODO(user): tune
    obs_groups: Dict[str, List[str]] = field(
        default_factory=lambda: {"actor": ["actor"], "critic": ["critic"]}
    )
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    algorithm: AlgorithmCfg = field(default_factory=AlgorithmCfg)
