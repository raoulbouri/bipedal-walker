"""
Tests for the mjlab_biped RSL-RL PPO runner config (Phase 7.B).

Verifies structure and frozen values per CLAUDE.md's Sub-task 7.B table,
corrected 2026-07-11 to match mjlab's REAL RslRlOnPolicyRunnerCfg shape
(separate actor/critic RslRlModelCfg-shaped fields, not one combined
"policy" config — see mjlab_biped/rl_cfg.py's module docstring for the
full correction, caught live via a real Colab TypeError):
- Construction with no missing fields
- obs_groups matches the real group names (and they aren't typos)
- Dimension consistency against mjlab_biped.observations' real constants
- Every frozen field value, across actor/critic/algorithm
- Registry wiring (rl_cfg=RunnerCfg() on the registered task)
- TODO(user) marker presence (process check)
"""

import inspect

import mjlab_biped
from mjlab_biped.config import get_task_cfg
from mjlab_biped.observations import STACKED_ACTOR_OBS_DIM, CRITIC_OBS_DIM
from mjlab_biped.rl_cfg import RunnerCfg, ModelCfg, AlgorithmCfg, ACTION_DIM
from mjlab_biped import rl_cfg as rl_cfg_module


# ---------------------------------------------------------------------------
# 1. Construction tests
# ---------------------------------------------------------------------------

def test_runner_cfg_constructs():
    cfg = RunnerCfg()
    assert cfg is not None


def test_model_cfg_constructs():
    cfg = ModelCfg()
    assert cfg is not None


def test_algorithm_cfg_constructs():
    cfg = AlgorithmCfg()
    assert cfg is not None


def test_runner_cfg_nests_actor_critic_and_algorithm():
    cfg = RunnerCfg()
    assert isinstance(cfg.actor, ModelCfg)
    assert isinstance(cfg.critic, ModelCfg)
    assert isinstance(cfg.algorithm, AlgorithmCfg)
    # actor and critic are independent instances, not the same object
    # or accidentally sharing mutable state.
    assert cfg.actor is not cfg.critic


# ---------------------------------------------------------------------------
# 2. obs_groups matches real group names (not typos)
# ---------------------------------------------------------------------------

def test_obs_groups_matches_real_group_names():
    cfg = RunnerCfg()
    assert cfg.obs_groups == {"actor": ("actor",), "critic": ("critic",)}

    # Sanity-check the group names aren't typos by confirming the real
    # Phase 6.B term-name tuples exist and are non-empty.
    assert isinstance(mjlab_biped.ACTOR_TERM_NAMES, tuple)
    assert len(mjlab_biped.ACTOR_TERM_NAMES) > 0
    assert isinstance(mjlab_biped.CRITIC_TERM_NAMES, tuple)
    assert len(mjlab_biped.CRITIC_TERM_NAMES) > 0


# ---------------------------------------------------------------------------
# 3. Dimension consistency (import, don't hardcode)
# ---------------------------------------------------------------------------

def test_dimension_consistency_against_observations_module():
    assert STACKED_ACTOR_OBS_DIM == 120
    assert CRITIC_OBS_DIM == 39
    assert ACTION_DIM == 6


# ---------------------------------------------------------------------------
# 4. Frozen-value tests (every field)
# ---------------------------------------------------------------------------

def test_runner_cfg_frozen_values():
    cfg = RunnerCfg()
    assert cfg.num_steps_per_env == 24
    assert cfg.max_iterations == 1500
    assert cfg.save_interval == 50
    assert cfg.obs_groups == {"actor": ("actor",), "critic": ("critic",)}


def test_actor_model_cfg_frozen_values():
    cfg = RunnerCfg().actor
    assert cfg.hidden_dims == [512, 256, 128]
    assert cfg.activation == "elu"
    assert cfg.obs_normalization is True
    assert cfg.init_noise_std == 1.0


def test_critic_model_cfg_frozen_values():
    cfg = RunnerCfg().critic
    assert cfg.hidden_dims == [512, 256, 128]
    assert cfg.activation == "elu"
    assert cfg.obs_normalization is True
    # init_noise_std exists on ModelCfg (shared dataclass shape) but is
    # never read for the critic in mjlab_task.py's real translation
    # (mjlab's critic RslRlModelCfg has no distribution_cfg) -- no
    # frozen-value assertion needed for it here.


def test_algorithm_cfg_frozen_values():
    cfg = RunnerCfg().algorithm
    assert cfg.num_learning_epochs == 5
    assert cfg.num_mini_batches == 4
    assert cfg.learning_rate == 1.0e-3
    assert cfg.schedule == "adaptive"
    assert cfg.gamma == 0.99
    assert cfg.lam == 0.95
    # Phase 7.W.2 (2026-07-14): raised 0.005 -> 0.02, see rl_cfg.py's
    # AlgorithmCfg.entropy_coef comment for the wandb-evidenced rationale.
    assert cfg.entropy_coef == 0.02
    assert cfg.desired_kl == 0.01
    assert cfg.max_grad_norm == 1.0
    assert cfg.value_loss_coef == 1.0
    assert cfg.use_clipped_value_loss is True
    assert cfg.clip_param == 0.2


# ---------------------------------------------------------------------------
# 5. Registry wiring
# ---------------------------------------------------------------------------

def test_registry_wiring():
    cfg = get_task_cfg("Mjlab-Biped-Balance-v0")
    assert cfg["rl_cfg"] is not None
    assert isinstance(cfg["rl_cfg"], RunnerCfg)


# ---------------------------------------------------------------------------
# 6. TODO(user) marker presence (process check)
# ---------------------------------------------------------------------------

def test_todo_markers_present_in_source():
    src = inspect.getsource(rl_cfg_module)
    todo_count = src.count("TODO(user)")
    # One TODO per tunable field: ModelCfg (4 fields, defined once, used
    # for both actor and critic instances) + AlgorithmCfg (12 fields) +
    # RunnerCfg's 3 top-level tunables (num_steps_per_env, max_iterations,
    # save_interval) = 19 minimum.
    assert todo_count >= 19
