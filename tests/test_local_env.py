"""
Tests for mjlab_biped.local_env.BipedLocalEnv - the single-environment,
CPU-only wrapper used for local visualization/sanity-checking of the
Phase 6.A-6.E pieces ahead of the real Colab training run.
"""
import numpy as np
import pytest

from mjlab_biped.config import BipedEnvCfg, make_play_env_cfg
from mjlab_biped.local_env import BipedLocalEnv
from mjlab_biped.observations import STACKED_ACTOR_OBS_DIM


def test_reset_returns_finite_actor_obs():
    env = BipedLocalEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (STACKED_ACTOR_OBS_DIM,)
    assert np.all(np.isfinite(obs))


def test_reset_is_reproducible_with_same_seed():
    env_a = BipedLocalEnv(seed=123)
    obs_a = env_a.reset()
    env_b = BipedLocalEnv(seed=123)
    obs_b = env_b.reset()
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(env_a.command, env_b.command)


def test_step_zero_action_finite_until_termination():
    """
    NOTE: zero action means "hold the stand-pose target passively" - with no
    balance controller (that's what Phase 7's RL training is for), this
    robot is EXPECTED to topple within roughly 30-70 steps, per CLAUDE.md's
    own documented design reality ("the robot topples - no balance
    controller yet - expected"; "quiet standing is mechanically hard
    without ankle-strategy torque"). This test only checks that every
    step's outputs remain finite and correctly typed up to (and including)
    whatever step termination first fires on - it does NOT assert survival.
    """
    env = BipedLocalEnv(env_cfg=make_play_env_cfg(), seed=1)
    env.reset()
    for _ in range(80):
        obs, reward, terminated, truncated, info = env.step(np.zeros(6))
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        if terminated or truncated:
            break
    else:
        pytest.fail("expected termination (fall) within 80 zero-action steps")


def test_step_returns_correct_info_keys():
    env = BipedLocalEnv(seed=2)
    env.reset()
    _, _, _, _, info = env.step(np.zeros(6))
    for key in ("reward", "fall_tilt", "fall_height", "terminated", "truncated", "step_count", "command", "base_height"):
        assert key in info
    assert info["step_count"] == 1


def test_aggressive_random_actions_eventually_terminate():
    """Sanity check that the termination wiring is actually live end-to-end,
    not just always False. Aggressive random actions should destabilize the
    robot and trigger a fall within a bounded number of steps."""
    env = BipedLocalEnv(seed=7)
    env.reset()
    rng = np.random.default_rng(0)
    terminated = False
    for _ in range(300):
        action = rng.uniform(-1.4, 1.4, size=6)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break
    assert terminated, "expected aggressive random actions to trigger a fall within 300 steps"
    assert info["fall_tilt"] or info["fall_height"]


def test_time_out_truncates_at_max_episode_steps():
    """
    Verifies the step_count -> time_out wiring directly, decoupled from
    whether/when the robot falls: uses a deliberately tiny
    max_episode_steps (3) so time_out fires well before any realistic
    fall could occur (per the trajectory investigated in
    test_step_zero_action_finite_until_termination, the robot doesn't
    begin tipping meaningfully within the first few steps).
    """
    from mjlab_biped.terminations import TerminationCfg

    cfg = BipedEnvCfg(termination_cfg=TerminationCfg(max_episode_steps=3))
    env = BipedLocalEnv(env_cfg=cfg, seed=3)
    env.reset()
    terminated = truncated = False
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(np.zeros(6))
    assert not terminated, "did not expect a fall within 3 steps from the stand pose"
    assert truncated
    assert info["step_count"] == 3


def test_domain_randomization_changes_model_when_enabled():
    """
    Confirms DR actually perturbs the model across resets when explicitly
    enabled, vs the play cfg (DR off by default) which should not perturb
    it at all. NOTE: DomainRandomizationCfg.enabled defaults to False
    everywhere per Phase 6.C (including BipedEnvCfg()'s own default) - DR
    must be explicitly turned on to observe an effect.
    """
    from mjlab_biped.domain_randomization import DomainRandomizationCfg

    dr_cfg = BipedEnvCfg(domain_randomization_cfg=DomainRandomizationCfg(enabled=True))
    env_dr = BipedLocalEnv(env_cfg=dr_cfg, seed=10)
    baseline_mass = env_dr._randomizer._baseline_body_mass.copy()
    env_dr.reset()
    mass_after_dr = env_dr.sim.model.body_mass.copy()
    assert not np.allclose(baseline_mass, mass_after_dr), "expected DR to perturb body mass away from baseline"

    env_play = BipedLocalEnv(env_cfg=make_play_env_cfg(), seed=10)
    baseline_mass_play = env_play._randomizer._baseline_body_mass.copy()
    env_play.reset()
    mass_after_play = env_play.sim.model.body_mass.copy()
    np.testing.assert_array_equal(baseline_mass_play, mass_after_play)


def test_get_critic_obs_shape():
    env = BipedLocalEnv(seed=5)
    env.reset()
    critic_obs = env.get_critic_obs()
    from mjlab_biped.observations import CRITIC_OBS_DIM
    assert critic_obs.shape == (CRITIC_OBS_DIM,)
    assert np.all(np.isfinite(critic_obs))
