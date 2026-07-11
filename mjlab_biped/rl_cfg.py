"""
RSL-RL PPO runner configuration for the biped RL task (Phase 7.B).

This module mirrors the REAL mjlab `RslRlOnPolicyRunnerCfg`/
`RslRlModelCfg`/`RslRlPpoAlgorithmCfg` shape (see
`src/mjlab/rl/config.py`, verified directly 2026-07-11 after a live
Colab `TypeError` exposed that the original IsaacLab-docs-derived guess
was structurally wrong, not just differently named) as plain Python
dataclasses. It does NOT import the real `mjlab`/`rsl_rl`/`torch`
packages — those are Colab-only dependencies per CLAUDE.md, never part
of the lean Mac core deps. Mirroring the real structure now means wiring
this into actual mjlab on Colab is a straight pass-through of field
values, not a redesign.

**2026-07-11 correction (was wrong before this):** mjlab's real
`RslRlOnPolicyRunnerCfg` has SEPARATE `actor: RslRlModelCfg` and
`critic: RslRlModelCfg` fields — there is no single combined "policy"
config with `actor_hidden_dims`/`critic_hidden_dims`. `RslRlModelCfg`'s
real fields are `hidden_dims` (singular, one network's own dims),
`activation`, `obs_normalization` (singular bool). `init_noise_std` is
NOT a direct field on `RslRlModelCfg` at all — in real mjlab it lives
inside `distribution_cfg["init_std"]`, and only the actor needs a
`distribution_cfg` (the critic just outputs a scalar value estimate, no
action distribution) — `mjlab_task.py` handles that translation at the
one place it's needed; this Mac-side mirror keeps a flat
`init_noise_std` field on `ModelCfg` for simplicity/testability and
`mjlab_task.py` reads it only off `.actor`. `obs_groups` values are
tuples in real mjlab (`RslRlBaseRunnerCfg`'s own default:
`{"actor": ("actor",), "critic": ("critic",)}`), not lists.

Per CLAUDE.md's Sub-task 7.B: this is config only. No real rsl_rl
wiring, no Colab work, no actual training happens here (that is Phase
7.C). All frozen starting values below are sourced from IsaacLab's
flat-terrain velocity-task PPO config, and are explicitly marked as
tunable via `# TODO(user): tune` comments on every field.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# Not a real rsl_rl field — a local sanity anchor for the dimension
# consistency check (see tests/test_rl_cfg.py), matching the 6 position
# actuators frozen in mjlab_biped/entity.py's ActuatorConfig.
ACTION_DIM = 6


@dataclass
class ModelCfg:
    """
    Mirrors real mjlab's `RslRlModelCfg` — one instance each for the
    actor and the critic network (NOT a single combined policy config).
    `init_noise_std` is only meaningful (and only read) for the actor's
    instance — real mjlab nests it inside `distribution_cfg`, not as a
    direct field; kept flat here for simplicity, translated by
    `mjlab_task.py`.
    """

    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])  # TODO(user): tune
    activation: str = "elu"  # TODO(user): tune
    obs_normalization: bool = True  # TODO(user): tune
    init_noise_std: float = 1.0  # TODO(user): tune (actor only)


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
    Mirrors `RslRlOnPolicyRunnerCfg`: top-level runner config, with
    SEPARATE `actor: ModelCfg` and `critic: ModelCfg` fields (not one
    combined policy config — see module docstring) and
    `algorithm: AlgorithmCfg`.

    `obs_groups` wires to the Phase 6.B observation manager's real group
    names ("actor" and "critic", see mjlab_biped/observations.py's
    ACTOR_TERM_NAMES/CRITIC_TERM_NAMES) rather than being redefined here.
    """

    num_steps_per_env: int = 24  # TODO(user): tune
    max_iterations: int = 1500  # TODO(user): tune (budget; gate on eval, not on exhausting it)
    save_interval: int = 50  # TODO(user): tune
    obs_groups: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)}
    )
    actor: ModelCfg = field(default_factory=ModelCfg)
    critic: ModelCfg = field(default_factory=ModelCfg)
    algorithm: AlgorithmCfg = field(default_factory=AlgorithmCfg)
