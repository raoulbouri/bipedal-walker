"""
Tests for Phase 6.E: env config assembly + local task registry.

Covers docs/env_assembly_spec.md's "Verification requirements":
1. Task registered and discoverable.
2. BipedEnvCfg() constructs with no missing fields.
3. decimation * timestep == 0.02 invariant, valid case.
4. decimation * timestep == 0.02 invariant, broken case raises.
5. play_env_cfg is smaller with DR off.
6. Double-registration raises.
7. Unregistered lookup raises KeyError with a helpful message.
"""

import pytest

import mjlab_biped
from mjlab_biped.config import (
    SceneCfg,
    SimCfg,
    BipedEnvCfg,
    make_play_env_cfg,
    TASK_REGISTRY,
    register_mjlab_task,
    get_task_cfg,
)
from mjlab_biped.domain_randomization import DomainRandomizationCfg


class TestTaskRegistration:
    def test_task_registered_and_discoverable(self):
        cfg = get_task_cfg("Mjlab-Biped-Balance-v0")
        assert isinstance(cfg, dict)
        assert "env_cfg" in cfg
        assert "play_env_cfg" in cfg
        assert "rl_cfg" in cfg
        # Phase 7.B wires a real RunnerCfg() in (was None before 7.B existed).
        from mjlab_biped.rl_cfg import RunnerCfg
        assert isinstance(cfg["rl_cfg"], RunnerCfg)
        assert isinstance(cfg["env_cfg"], BipedEnvCfg)
        assert isinstance(cfg["play_env_cfg"], BipedEnvCfg)


class TestBipedEnvCfgDefaults:
    def test_constructs_with_no_args(self):
        cfg = BipedEnvCfg()
        assert cfg.episode_length_s == 20.0
        assert cfg.scene.num_envs == 4096
        assert cfg.sim.decimation == 10
        assert cfg.sim.timestep == 0.002
        assert cfg.sim.integrator == "implicitfast"
        assert cfg.scene.env_spacing == 2.0
        assert cfg.scene.terrain_type == "plane"


class TestDecimationTimestepInvariant:
    def test_valid_default_case(self):
        # Should not raise.
        cfg = BipedEnvCfg(sim=SimCfg())
        assert cfg.sim.decimation == 10
        assert cfg.sim.timestep == 0.002

    def test_broken_case_raises(self):
        with pytest.raises(ValueError):
            BipedEnvCfg(sim=SimCfg(decimation=7))


class TestPlayVariant:
    def test_smaller_num_envs_and_dr_off(self):
        play_cfg = make_play_env_cfg()
        default_cfg = BipedEnvCfg()
        assert play_cfg.scene.num_envs < default_cfg.scene.num_envs
        assert play_cfg.domain_randomization_cfg.enabled is False


class TestRegistryErrors:
    def test_double_registration_raises(self):
        with pytest.raises(ValueError):
            register_mjlab_task(
                "Mjlab-Biped-Balance-v0",
                env_cfg=BipedEnvCfg(),
                play_env_cfg=make_play_env_cfg(),
            )

    def test_unregistered_lookup_raises_keyerror_with_helpful_message(self):
        with pytest.raises(KeyError) as excinfo:
            get_task_cfg("Nonexistent-Task-v0")
        assert "Mjlab-Biped-Balance-v0" in str(excinfo.value)
