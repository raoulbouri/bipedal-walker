"""
Tests for Phase 6.C: velocity command sampler, init-state noise, and
domain randomization (mjlab_biped/commands.py, init_noise.py,
domain_randomization.py).

Covers 8 categories per docs/events_spec.md:
1. Command sampler bounds/exactness
2. Init-state noise bounds + passive-joint invariance
3. DR bounds
4. DR no-leakage (non-compounding across repeated apply() calls)
5. DR disabled restores baseline
6. kp/kv invariant preserved (biasprm[1] == -gainprm[0])
7. Jetson toggle exactness
8. Live integration (mutated model + noised qpos steps without NaN/Inf)
"""

import numpy as np
import mujoco
import pytest

from mjlab_biped import (
    CommandRangeCfg,
    sample_command,
    InitNoiseCfg,
    sample_init_state,
    apply_init_state,
    DomainRandomizationCfg,
    DomainRandomizer,
)

WARP_MODEL = "models/mjcf/biped_warp.xml"

NOMINAL_QPOS = np.array(
    [0.0, 0.0, 0.2030, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)


# ---------------------------------------------------------------------------
# 1. Command sampler
# ---------------------------------------------------------------------------


class TestCommandSampler:
    def test_zero_width_range_exact(self):
        rng = np.random.default_rng(0)
        cfg = CommandRangeCfg()  # all default (0.0, 0.0)
        for _ in range(50):
            cmd = sample_command(rng, cfg)
            assert cmd.shape == (3,)
            np.testing.assert_array_equal(cmd, np.array([0.0, 0.0, 0.0]))

    def test_wide_range_bounds_and_mean(self):
        rng = np.random.default_rng(42)
        cfg = CommandRangeCfg(
            vx_range=(-1.0, 1.0), vy_range=(-1.0, 1.0), yaw_rate_range=(-1.0, 1.0)
        )
        samples = np.array([sample_command(rng, cfg) for _ in range(1000)])
        assert np.all(samples >= -1.0)
        assert np.all(samples <= 1.0)
        means = samples.mean(axis=0)
        assert np.all(np.abs(means) < 0.15)  # loose sanity, not formal GOF


# ---------------------------------------------------------------------------
# 2. Init-state noise
# ---------------------------------------------------------------------------


class TestInitNoise:
    def test_bounds_and_unit_quaternion(self):
        rng = np.random.default_rng(1)
        cfg = InitNoiseCfg()
        for _ in range(1000):
            sample = sample_init_state(rng, cfg)
            assert np.all(np.abs(sample["joint_angles"]) <= cfg.joint_angle_noise)
            assert sample["joint_angles"].shape == (6,)
            assert np.all(np.abs(sample["base_xy"]) <= cfg.base_xy_noise)
            assert sample["base_xy"].shape == (2,)
            assert abs(sample["base_z"]) <= cfg.base_z_noise
            quat = sample["base_quat_offset"]
            assert quat.shape == (4,)
            assert abs(np.linalg.norm(quat) - 1.0) < 1e-9

    def test_apply_init_state_passive_joints_unchanged(self):
        rng = np.random.default_rng(2)
        cfg = InitNoiseCfg()
        for _ in range(200):
            sample = sample_init_state(rng, cfg)
            qpos = apply_init_state(NOMINAL_QPOS, sample)
            assert qpos.shape == (15,)
            np.testing.assert_array_equal(qpos[13:15], NOMINAL_QPOS[13:15])

    def test_apply_init_state_perturbs_expected_slices(self):
        rng = np.random.default_rng(3)
        cfg = InitNoiseCfg()
        sample = sample_init_state(rng, cfg)
        qpos = apply_init_state(NOMINAL_QPOS, sample)
        # base xy/z should reflect the sampled noise added to nominal
        np.testing.assert_allclose(qpos[0:2], NOMINAL_QPOS[0:2] + sample["base_xy"])
        np.testing.assert_allclose(qpos[2], NOMINAL_QPOS[2] + sample["base_z"])
        # quaternion should remain unit norm after composition
        assert abs(np.linalg.norm(qpos[3:7]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Shared fixture-ish helpers for DR tests
# ---------------------------------------------------------------------------


def _load_model():
    return mujoco.MjModel.from_xml_path(WARP_MODEL)


# ---------------------------------------------------------------------------
# 3. Domain randomization — bounds
# ---------------------------------------------------------------------------


class TestDomainRandomizationBounds:
    def test_mass_friction_gain_bounds(self):
        model = _load_model()
        randomizer = DomainRandomizer(model)
        baseline_mass = model.body_mass.copy()
        baseline_friction = model.geom_friction.copy()
        baseline_gainprm = model.actuator_gainprm.copy()
        baseline_biasprm = model.actuator_biasprm.copy()

        cfg = DomainRandomizationCfg(enabled=True, jetson_toggle_enabled=False)

        for seed in range(200):
            rng = np.random.default_rng(seed)
            randomizer.apply(model, rng, cfg)

            mass_lo, mass_hi = cfg.mass_range
            for i in range(1, model.nbody):
                lo = baseline_mass[i] * mass_lo
                hi = baseline_mass[i] * mass_hi
                lo, hi = min(lo, hi), max(lo, hi)
                assert lo - 1e-9 <= model.body_mass[i] <= hi + 1e-9

            fric_lo, fric_hi = cfg.friction_range
            nonzero = baseline_friction[:, 0] != 0.0
            ratios = model.geom_friction[nonzero, 0] / baseline_friction[nonzero, 0]
            assert np.all(ratios >= fric_lo - 1e-9)
            assert np.all(ratios <= fric_hi + 1e-9)

            gain_lo, gain_hi = cfg.gain_range
            for i in range(model.nu):
                kp_ratio = model.actuator_gainprm[i, 0] / baseline_gainprm[i, 0]
                assert gain_lo - 1e-9 <= kp_ratio <= gain_hi + 1e-9
                kv_ratio = model.actuator_biasprm[i, 2] / baseline_biasprm[i, 2]
                assert gain_lo - 1e-9 <= kv_ratio <= gain_hi + 1e-9


# ---------------------------------------------------------------------------
# 4. Domain randomization — no leakage (non-compounding)
# ---------------------------------------------------------------------------


class TestDomainRandomizationNoLeakage:
    def test_repeated_apply_does_not_compound(self):
        cfg = DomainRandomizationCfg(enabled=True)

        model_a = _load_model()
        randomizer_a = DomainRandomizer(model_a)
        randomizer_a.apply(model_a, np.random.default_rng(111), cfg)  # seed A
        randomizer_a.apply(model_a, np.random.default_rng(222), cfg)  # seed B, on top

        model_fresh = _load_model()
        randomizer_fresh = DomainRandomizer(model_fresh)
        randomizer_fresh.apply(model_fresh, np.random.default_rng(222), cfg)  # seed B only

        np.testing.assert_array_equal(model_a.body_mass, model_fresh.body_mass)
        np.testing.assert_array_equal(model_a.body_inertia, model_fresh.body_inertia)
        np.testing.assert_array_equal(model_a.geom_friction, model_fresh.geom_friction)
        np.testing.assert_array_equal(
            model_a.actuator_gainprm, model_fresh.actuator_gainprm
        )
        np.testing.assert_array_equal(
            model_a.actuator_biasprm, model_fresh.actuator_biasprm
        )


# ---------------------------------------------------------------------------
# 5. Domain randomization — disabled restores baseline
# ---------------------------------------------------------------------------


class TestDomainRandomizationDisabledRestores:
    def test_disabled_restores_pristine_baseline(self):
        model = _load_model()
        randomizer = DomainRandomizer(model)

        baseline_mass = model.body_mass.copy()
        baseline_inertia = model.body_inertia.copy()
        baseline_friction = model.geom_friction.copy()
        baseline_gainprm = model.actuator_gainprm.copy()
        baseline_biasprm = model.actuator_biasprm.copy()

        enabled_cfg = DomainRandomizationCfg(enabled=True)
        randomizer.apply(model, np.random.default_rng(7), enabled_cfg)
        # sanity: model actually perturbed away from baseline
        assert not np.array_equal(model.body_mass, baseline_mass)

        disabled_cfg = DomainRandomizationCfg(enabled=False)
        randomizer.apply(model, np.random.default_rng(8), disabled_cfg)

        np.testing.assert_array_equal(model.body_mass, baseline_mass)
        np.testing.assert_array_equal(model.body_inertia, baseline_inertia)
        np.testing.assert_array_equal(model.geom_friction, baseline_friction)
        np.testing.assert_array_equal(model.actuator_gainprm, baseline_gainprm)
        np.testing.assert_array_equal(model.actuator_biasprm, baseline_biasprm)


# ---------------------------------------------------------------------------
# 6. kp/kv invariant preserved
# ---------------------------------------------------------------------------


class TestKpKvInvariant:
    def test_biasprm1_equals_negative_gainprm0(self):
        model = _load_model()
        randomizer = DomainRandomizer(model)
        cfg = DomainRandomizationCfg(enabled=True)
        for seed in range(50):
            randomizer.apply(model, np.random.default_rng(seed), cfg)
            for i in range(model.nu):
                assert model.actuator_biasprm[i, 1] == pytest.approx(
                    -model.actuator_gainprm[i, 0]
                )


# ---------------------------------------------------------------------------
# 7. Jetson toggle
# ---------------------------------------------------------------------------


class TestJetsonToggle:
    def test_torso_mass_exactly_one_of_two_values(self):
        model = _load_model()
        randomizer = DomainRandomizer(model)
        torso_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "composite_part_1__1_"
        )
        cfg = DomainRandomizationCfg(enabled=True, jetson_toggle_enabled=True)

        seen_values = set()
        for seed in range(100):
            randomizer.apply(model, np.random.default_rng(seed), cfg)
            mass = model.body_mass[torso_id]
            assert mass == pytest.approx(0.099) or mass == pytest.approx(0.279)
            seen_values.add(round(float(mass), 3))

        assert seen_values == {0.099, 0.279}


# ---------------------------------------------------------------------------
# 8. Live integration
# ---------------------------------------------------------------------------


class TestLiveIntegration:
    def test_randomized_model_and_noised_state_step_without_nan(self):
        model = _load_model()
        randomizer = DomainRandomizer(model)
        dr_cfg = DomainRandomizationCfg(enabled=True)
        randomizer.apply(model, np.random.default_rng(99), dr_cfg)

        init_cfg = InitNoiseCfg()
        rng = np.random.default_rng(100)
        sample = sample_init_state(rng, init_cfg)
        qpos = apply_init_state(NOMINAL_QPOS, sample)

        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qacc))
