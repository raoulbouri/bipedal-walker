"""
Tests for Phase 7.R.3: recovery-progress reward + fallen-pose reset
parameters (mjlab_biped/recovery.py).

Covers:
1. recovery_progress_term: finite, correct sign, correctly gated (zero
   outside the "fallen" regime, never negative even for downward
   motion/upside-down orientation while fallen).
2. sample_fallen_pose: distribution coverage (roll/pitch/yaw span close
   to the full configured range across many samples) and exact
   reproducibility under a seeded RNG.
"""

import numpy as np
import pytest

from mjlab_biped import (
    RecoveryCfg,
    recovery_progress_term,
    FallenPoseSampleCfg,
    sample_fallen_pose,
    FALLEN_POSE_RANGE,
    FALLEN_JOINT_POSITION_RANGE,
    RECOVERY_RESET_PROB,
)

HEIGHT_THRESHOLD = 0.15
TILT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# 1. recovery_progress_term
# ---------------------------------------------------------------------------


class TestRecoveryProgressTerm:
    def test_zero_outside_recovery_regime(self):
        """Standing tall and upright (height/upright both well above
        threshold) -> zero reward, regardless of velocity/orientation
        sign, so this term never double-counts once already standing."""
        n = 4
        height = np.full(n, 0.2030)  # stand height
        upright_value = np.full(n, 1.0)  # perfectly upright
        base_linvel_z = np.array([1.0, -1.0, 0.0, 5.0])  # varied, shouldn't matter

        out = recovery_progress_term(
            height, upright_value, base_linvel_z, HEIGHT_THRESHOLD, TILT_THRESHOLD
        )
        assert out.shape == (n,)
        np.testing.assert_array_equal(out, np.zeros(n))

    def test_positive_when_rising_while_fallen(self):
        """Below height threshold, moving upward -> positive reward."""
        n = 1
        height = np.array([0.05])  # well below threshold
        upright_value = np.array([-0.5])  # tilted
        base_linvel_z = np.array([2.0])  # rising

        out = recovery_progress_term(
            height, upright_value, base_linvel_z, HEIGHT_THRESHOLD, TILT_THRESHOLD
        )
        assert out[0] > 0.0
        assert np.isfinite(out[0])

    def test_never_negative_even_when_falling_or_upside_down(self):
        """Downward velocity and negative upright while fallen must NOT
        turn this term into an extra penalty -- both relu'd to 0, so the
        worst case is exactly 0, never negative."""
        n = 1
        height = np.array([0.05])
        upright_value = np.array([-1.0])  # fully upside down
        base_linvel_z = np.array([-3.0])  # falling fast

        out = recovery_progress_term(
            height, upright_value, base_linvel_z, HEIGHT_THRESHOLD, TILT_THRESHOLD
        )
        assert out[0] == 0.0

    def test_gated_by_either_height_or_tilt(self):
        """Fallen by height alone (upright still nominally 'ok' but body
        low) or by tilt alone (upright bad but height still nominal) both
        activate the term -- it's an OR gate matching fall_tilt_fn OR
        fall_height_fn's own union used at termination time."""
        # Low height, upright still "fine" numerically.
        out_low_height = recovery_progress_term(
            np.array([0.05]), np.array([0.9]), np.array([1.0]), HEIGHT_THRESHOLD, TILT_THRESHOLD
        )
        assert out_low_height[0] > 0.0

        # Height nominal, upright bad (tilted over while still "tall").
        out_bad_tilt = recovery_progress_term(
            np.array([0.203]), np.array([-0.9]), np.array([1.0]), HEIGHT_THRESHOLD, TILT_THRESHOLD
        )
        assert out_bad_tilt[0] > 0.0

    def test_finite_over_random_batch(self):
        rng = np.random.default_rng(7)
        n = 32
        height = rng.uniform(0.0, 0.3, size=n)
        upright_value = rng.uniform(-1.0, 1.0, size=n)
        base_linvel_z = rng.normal(size=n)

        out = recovery_progress_term(
            height, upright_value, base_linvel_z, HEIGHT_THRESHOLD, TILT_THRESHOLD
        )
        assert out.shape == (n,)
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)


# ---------------------------------------------------------------------------
# 2. sample_fallen_pose
# ---------------------------------------------------------------------------


class TestSampleFallenPose:
    def test_reproducible_under_seeded_rng(self):
        cfg = FallenPoseSampleCfg()
        sample_a = sample_fallen_pose(np.random.default_rng(123), cfg)
        sample_b = sample_fallen_pose(np.random.default_rng(123), cfg)
        for key in ("roll", "pitch", "yaw", "z_offset"):
            assert sample_a[key] == sample_b[key]
        np.testing.assert_array_equal(sample_a["joint_angles"], sample_b["joint_angles"])

    def test_different_seeds_differ(self):
        cfg = FallenPoseSampleCfg()
        sample_a = sample_fallen_pose(np.random.default_rng(1), cfg)
        sample_b = sample_fallen_pose(np.random.default_rng(2), cfg)
        assert sample_a["roll"] != sample_b["roll"]

    def test_distribution_covers_full_configured_range(self):
        """Across many samples, roll/pitch/yaw must span close to the
        full configured (-pi, pi) range -- not, say, accidentally
        clamped to a tiny sliver by a units/range bug."""
        cfg = FallenPoseSampleCfg()
        rng = np.random.default_rng(0)
        rolls, pitches, yaws, z_offsets = [], [], [], []
        joint_angles_all = []
        for _ in range(2000):
            s = sample_fallen_pose(rng, cfg)
            rolls.append(s["roll"])
            pitches.append(s["pitch"])
            yaws.append(s["yaw"])
            z_offsets.append(s["z_offset"])
            joint_angles_all.append(s["joint_angles"])

        for name, values, expected_range in (
            ("roll", rolls, FALLEN_POSE_RANGE["roll"]),
            ("pitch", pitches, FALLEN_POSE_RANGE["pitch"]),
            ("yaw", yaws, FALLEN_POSE_RANGE["yaw"]),
        ):
            values = np.array(values)
            lo, hi = expected_range
            assert values.min() >= lo and values.max() <= hi, f"{name} out of configured range"
            # With 2000 samples uniform over the full range, expect close
            # coverage of both extremes -- catches an accidental
            # off-by-orders-of-magnitude or degenerate-range bug.
            span = hi - lo
            assert values.min() < lo + 0.05 * span, f"{name} samples never got near the lower bound"
            assert values.max() > hi - 0.05 * span, f"{name} samples never got near the upper bound"

        z_offsets = np.array(z_offsets)
        lo, hi = FALLEN_POSE_RANGE["z"]
        assert z_offsets.min() >= lo and z_offsets.max() <= hi

        joint_angles_all = np.array(joint_angles_all)
        assert joint_angles_all.shape == (2000, 6)
        lo, hi = FALLEN_JOINT_POSITION_RANGE
        assert joint_angles_all.min() >= lo and joint_angles_all.max() <= hi


# ---------------------------------------------------------------------------
# 3. Sanity on the frozen constants themselves
# ---------------------------------------------------------------------------


def test_recovery_reset_prob_is_a_valid_fraction():
    assert 0.0 < RECOVERY_RESET_PROB < 1.0


def test_recovery_cfg_default_weight_positive():
    cfg = RecoveryCfg()
    assert cfg.weight > 0.0
