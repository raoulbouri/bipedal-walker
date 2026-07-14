"""
Tests for the mjlab_biped reward/termination stubs (Phase 6.D).

Verifies structure and behavior per `docs/rewards_termination_spec.md`:
- Finite-shape guarantees for all reward terms and compute_reward
- Stand-keyframe non-termination boundary case
- Independent fall_tilt / fall_height behavior
- Strict threshold boundary (< not <=) for both termination conditions
- time_out behavior
- Config defaults + TODO(user) markers
- command_tracking_weight=0.0 truly excludes the term from the default reward
"""

import inspect

import numpy as np
import pytest

from mjlab_biped import (
    upright,
    RewardCfg,
    alive_bonus,
    upright_term,
    command_tracking_term,
    control_effort_term,
    action_rate_term,
    compute_reward,
    TerminationCfg,
    fall_tilt,
    fall_height,
    fall,
    time_out,
)
from mjlab_biped import rewards as rewards_module
from mjlab_biped import terminations as terminations_module


IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
SIDEWAYS_QUAT = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # upright ~= 0
STAND_HEIGHT = 0.2030


def _batch(vec, n):
    return np.tile(np.asarray(vec, dtype=np.float64), (n, 1))


def _random_finite_state(rng, n):
    """Randomized-but-finite batch of inputs, quaternions normalized."""
    quat = rng.normal(size=(n, 4))
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    base_height = rng.uniform(0.0, 0.3, size=(n,))
    base_linvel = rng.normal(size=(n, 3))
    velocity_command = rng.normal(size=(n, 3))
    action = rng.normal(size=(n, 6))
    previous_action = rng.normal(size=(n, 6))
    actuator_torque = rng.uniform(-2.5, 2.5, size=(n, 6))  # within the real +-2.5 N*m actuator limit
    return quat, base_height, base_linvel, velocity_command, action, previous_action, actuator_torque


# ---------------------------------------------------------------------------
# 1. Finite shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 4])
def test_reward_terms_finite_and_shaped(n):
    rng = np.random.default_rng(42)
    quat, base_height, base_linvel, velocity_command, action, previous_action, actuator_torque = _random_finite_state(rng, n)

    ab = alive_bonus(n)
    assert ab.shape == (n,)
    assert np.all(np.isfinite(ab))

    up = upright_term(quat)
    assert up.shape == (n,)
    assert np.all(np.isfinite(up))

    ct = command_tracking_term(base_linvel, velocity_command)
    assert ct.shape == (n,)
    assert np.all(np.isfinite(ct))

    ce = control_effort_term(actuator_torque)
    assert ce.shape == (n,)
    assert np.all(np.isfinite(ce))
    # 2026-07-14: raw positive magnitude now (sign-inversion fix) --
    # penalty sign applied entirely via the negative RewardCfg weight.
    assert np.all(ce >= 0.0)

    ar = action_rate_term(action, previous_action)
    assert ar.shape == (n,)
    assert np.all(np.isfinite(ar))

    cfg = RewardCfg()
    total = compute_reward(cfg, quat, base_linvel, velocity_command, action, previous_action, actuator_torque)
    assert total.shape == (n,)
    assert np.all(np.isfinite(total))


# ---------------------------------------------------------------------------
# 2. Stand-keyframe non-termination
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 3])
def test_stand_keyframe_does_not_terminate(n):
    cfg = TerminationCfg()
    quat = _batch(IDENTITY_QUAT, n)
    height = np.full(n, STAND_HEIGHT)

    result = fall(quat, height, cfg)
    assert result.shape == (n,)
    assert not np.any(result)


# ---------------------------------------------------------------------------
# 3. Independent fall_tilt test
# ---------------------------------------------------------------------------

def test_fall_tilt_independent_of_height():
    cfg = TerminationCfg()
    quat = _batch(SIDEWAYS_QUAT, 1)
    height = np.array([STAND_HEIGHT])

    assert upright(quat)[0] < cfg.tilt_threshold
    assert bool(fall_tilt(quat, cfg)[0]) is True
    assert bool(fall_height(height, cfg)[0]) is False


# ---------------------------------------------------------------------------
# 4. Independent fall_height test
# ---------------------------------------------------------------------------

def test_fall_height_independent_of_tilt():
    cfg = TerminationCfg()
    quat = _batch(IDENTITY_QUAT, 1)
    height = np.array([0.05])

    assert bool(fall_height(height, cfg)[0]) is True
    assert bool(fall_tilt(quat, cfg)[0]) is False


# ---------------------------------------------------------------------------
# 5. Threshold boundary tests (strict <, not <=)
# ---------------------------------------------------------------------------

def test_tilt_threshold_boundary():
    cfg = TerminationCfg()  # tilt_threshold = 0.5
    # Solve 1 - 2*qx^2 = 0.5 -> qx = 0.5, qy = 0, qw = sqrt(1 - qx^2)
    qx = 0.5
    qw = np.sqrt(1.0 - qx**2)
    quat_at = np.array([[qw, qx, 0.0, 0.0]])
    assert np.isclose(upright(quat_at)[0], 0.5)
    # Exactly at threshold: must NOT fire (strict <)
    assert bool(fall_tilt(quat_at, cfg)[0]) is False

    # Slightly more tilted (upright just below 0.5): should fire
    qx_below = 0.51
    qw_below = np.sqrt(1.0 - qx_below**2)
    quat_below = np.array([[qw_below, qx_below, 0.0, 0.0]])
    assert upright(quat_below)[0] < 0.5
    assert bool(fall_tilt(quat_below, cfg)[0]) is True

    # Slightly less tilted (upright just above 0.5): should not fire
    qx_above = 0.49
    qw_above = np.sqrt(1.0 - qx_above**2)
    quat_above = np.array([[qw_above, qx_above, 0.0, 0.0]])
    assert upright(quat_above)[0] > 0.5
    assert bool(fall_tilt(quat_above, cfg)[0]) is False


def test_height_threshold_boundary():
    cfg = TerminationCfg()  # height_threshold = 0.15
    height_at = np.array([0.15])
    assert bool(fall_height(height_at, cfg)[0]) is False  # exactly at: must NOT fire

    height_above = np.array([0.151])
    assert bool(fall_height(height_above, cfg)[0]) is False

    height_below = np.array([0.149])
    assert bool(fall_height(height_below, cfg)[0]) is True


# ---------------------------------------------------------------------------
# 6. time_out test
# ---------------------------------------------------------------------------

def test_time_out():
    cfg = TerminationCfg()  # max_episode_steps = 1000
    step_count = np.array([500, 999, 1000, 1001, 2000])
    expected = step_count >= cfg.max_episode_steps
    result = time_out(step_count, cfg)
    assert result.shape == step_count.shape
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# 7. Weight defaults + TODO markers
# ---------------------------------------------------------------------------

def test_reward_cfg_defaults():
    cfg = RewardCfg()
    assert cfg.alive_bonus_weight == 1.0
    assert cfg.upright_weight == 1.0
    assert cfg.command_tracking_weight == 0.0
    # Phase 7.R.2 (2026-07-13): the first retune (-0.1/-1, briefly
    # committed) turned out 10-100x too aggressive vs. real references
    # (mjlab's own G1/Go1 action_rate_l2=-0.1 for the identical formula;
    # legged_gym/ANYmal torques=-1e-5) and stopped the policy from
    # learning to stay upright/alive at all -- see MEMORY.md for the
    # full research writeup. control_effort is now torque-based (see
    # control_effort_term), not action-magnitude-based.
    assert cfg.control_effort_weight == -1e-4
    assert cfg.action_rate_weight == -0.1


def test_termination_cfg_defaults():
    cfg = TerminationCfg()
    assert cfg.tilt_threshold == 0.5
    assert cfg.height_threshold == 0.15
    assert cfg.max_episode_steps == 1000


def test_todo_markers_present_in_source():
    rewards_src = inspect.getsource(rewards_module)
    todo_count = rewards_src.count("TODO(user)")
    # One TODO per reward weight field (5 fields)
    assert todo_count >= 5

    terminations_src = inspect.getsource(terminations_module)
    assert "TODO(user)" in terminations_src
    # Only max_episode_steps should carry a TODO in TerminationCfg;
    # tilt_threshold/height_threshold are physically-motivated, not TODOs.
    assert terminations_src.count("TODO(user)") == 1


# ---------------------------------------------------------------------------
# 8. command_tracking_weight=0.0 truly zero-effect check
# ---------------------------------------------------------------------------

def test_command_tracking_zero_weight_excluded_by_default():
    n = 1
    quat = _batch(IDENTITY_QUAT, n)
    base_linvel = np.array([[1.0, 2.0, 0.0]])  # nonzero
    velocity_command = np.zeros((n, 3))  # zero command -> nonzero tracking error
    action = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
    previous_action = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    actuator_torque = np.array([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2]])

    ct = command_tracking_term(base_linvel, velocity_command)
    assert not np.isclose(ct[0], 0.0)  # confirm the term itself is nonzero

    default_cfg = RewardCfg()
    default_reward = compute_reward(default_cfg, quat, base_linvel, velocity_command, action, previous_action, actuator_torque)

    # Manually compute the reward with command_tracking_weight forced nonzero
    boosted_cfg = RewardCfg(command_tracking_weight=10.0)
    boosted_reward = compute_reward(boosted_cfg, quat, base_linvel, velocity_command, action, previous_action, actuator_torque)

    assert not np.isclose(default_reward[0], boosted_reward[0])

    # Default-weight result must equal the sum of the other 4 weighted terms alone
    other_terms_sum = (
        default_cfg.alive_bonus_weight * alive_bonus(n)
        + default_cfg.upright_weight * upright_term(quat)
        + default_cfg.control_effort_weight * control_effort_term(actuator_torque)
        + default_cfg.action_rate_weight * action_rate_term(action, previous_action)
    )
    np.testing.assert_allclose(default_reward, other_terms_sum)
