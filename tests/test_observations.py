"""
Tests for the Phase 6.B observation manager (mjlab_biped/observations.py).

Covers: shape correctness, golden term ordering, the privileged-leak guard,
project_gravity correctness, a live BipedSim integration check, and
dimension-constant consistency.
"""

import numpy as np
import pytest

from mjlab_biped.observations import (
    build_actor_obs,
    build_critic_obs,
    project_gravity,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    ACTOR_TERM_NAMES,
    CRITIC_TERM_NAMES,
    ACTOR_SENSOR_WHITELIST,
    ObsHistory,
    ACTOR_HISTORY_LEN,
    STACKED_ACTOR_OBS_DIM,
)
from sim.biped_sim import BipedSim

MODEL_PATH = "models/mjcf/biped_warp.xml"


def _real_sensors():
    sim = BipedSim(MODEL_PATH)
    sim.reset(keyframe="stand")
    return sim, sim.sensors()


# ---------------------------------------------------------------------------
# 1. Shape tests
# ---------------------------------------------------------------------------

class TestShapes:
    def test_actor_obs_shape(self):
        _, sensors = _real_sensors()
        prev_action = np.zeros(6)
        vel_command = np.zeros(3)
        obs = build_actor_obs(sensors, prev_action, vel_command)
        assert obs.shape == (24,)

    def test_critic_obs_shape(self):
        sim, sensors = _real_sensors()
        prev_action = np.zeros(6)
        vel_command = np.zeros(3)
        com = sim.com()
        obs = build_critic_obs(sensors, prev_action, vel_command, com)
        assert obs.shape == (39,)


# ---------------------------------------------------------------------------
# 2. Term ordering golden test
# ---------------------------------------------------------------------------

class TestTermOrdering:
    def test_actor_term_names(self):
        assert ACTOR_TERM_NAMES == (
            "joint_pos_rel",
            "joint_vel_rel",
            "gyro",
            "previous_action",
            "velocity_command",
        )

    def test_critic_term_names(self):
        assert CRITIC_TERM_NAMES == (
            "joint_pos_rel",
            "joint_vel_rel",
            "gyro",
            "previous_action",
            "velocity_command",
            "projected_gravity",
            "foot_touch",
            "accelerometer",
            "base_linvel",
            "base_height",
            "com",
        )


# ---------------------------------------------------------------------------
# 3. Privileged-leak guard
# ---------------------------------------------------------------------------

class TestPrivilegedLeakGuard:
    def test_whitelist_excludes_privileged_sensors(self):
        for forbidden in (
            "torso_acc",
            "torso_pos",
            "torso_linvel",
            "torso_angvel",
            "torso_quat",
            "touch_l",
            "touch_r",
        ):
            assert forbidden not in ACTOR_SENSOR_WHITELIST

    def test_actor_obs_does_not_need_privileged_keys(self):
        """Deleting the privileged sensor keys must not break build_actor_obs
        — proves by construction that the actor path never reads them."""
        _, sensors = _real_sensors()
        stripped = dict(sensors)
        for forbidden in (
            "torso_acc",
            "torso_pos",
            "torso_linvel",
            "torso_angvel",
            "torso_quat",
            "touch_l",
            "touch_r",
        ):
            stripped.pop(forbidden, None)

        prev_action = np.zeros(6)
        vel_command = np.zeros(3)
        # Should not raise KeyError.
        obs = build_actor_obs(stripped, prev_action, vel_command)
        assert obs.shape == (24,)


# ---------------------------------------------------------------------------
# 4. project_gravity correctness
# ---------------------------------------------------------------------------

class TestProjectGravity:
    def test_identity_quaternion(self):
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        g = project_gravity(quat)
        np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-10)

    def test_90deg_pitch_about_y(self):
        # Quaternion for a 90-degree rotation about the body Y axis:
        # [cos(45deg), 0, sin(45deg), 0].
        half = np.deg2rad(45.0)
        quat = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
        g = project_gravity(quat)
        # A +90-degree rotation about Y maps world Z -> body X (world -x maps
        # to body... reasoning via rotation matrix): for quat [w,0,y,0] with
        # w=y=cos/sin(45deg), the rotation matrix R (body->world, since we
        # apply the inverse to go world->body in project_gravity) rotates
        # the world -z vector into the body frame. For a pure +90 deg
        # rotation about Y, world -z (0,0,-1) rotates to body frame as
        # (-1, 0, 0) or (1, 0, 0) depending on convention; verify it lands
        # purely along the X axis with unit magnitude and zero Y/Z.
        assert abs(abs(g[0]) - 1.0) < 1e-8
        assert abs(g[1]) < 1e-8
        assert abs(g[2]) < 1e-8

    def test_unit_norm_output(self):
        quat = np.array([0.7, 0.1, 0.2, 0.3])
        g = project_gravity(quat)
        assert abs(np.linalg.norm(g) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# 5. Live integration test
# ---------------------------------------------------------------------------

class TestLiveIntegration:
    def test_no_nan_and_actor_critic_consistency(self):
        sim, sensors = _real_sensors()
        prev_action = np.zeros(6)
        vel_command = np.zeros(3)
        com = sim.com()

        actor_obs = build_actor_obs(sensors, prev_action, vel_command)
        critic_obs = build_critic_obs(sensors, prev_action, vel_command, com)

        assert not np.any(np.isnan(actor_obs))
        assert not np.any(np.isinf(actor_obs))
        assert not np.any(np.isnan(critic_obs))
        assert not np.any(np.isinf(critic_obs))

        np.testing.assert_array_equal(critic_obs[:24], actor_obs)


# ---------------------------------------------------------------------------
# 6. Dimension constants match spec / actual builder output
# ---------------------------------------------------------------------------

class TestDimensionConstants:
    def test_constants_match_spec(self):
        assert ACTOR_OBS_DIM == 24
        assert CRITIC_OBS_DIM == 39
        assert ACTOR_HISTORY_LEN == 5
        assert STACKED_ACTOR_OBS_DIM == 120

    def test_constants_match_real_builder_output(self):
        sim, sensors = _real_sensors()
        prev_action = np.zeros(6)
        vel_command = np.zeros(3)
        com = sim.com()

        actor_obs = build_actor_obs(sensors, prev_action, vel_command)
        critic_obs = build_critic_obs(sensors, prev_action, vel_command, com)

        assert len(actor_obs) == ACTOR_OBS_DIM
        assert len(critic_obs) == CRITIC_OBS_DIM


# ---------------------------------------------------------------------------
# 7. ObsHistory (v1 actor history stacking)
# ---------------------------------------------------------------------------

class TestObsHistory:
    def test_reset_fills_all_slots_with_first_frame(self):
        rng = np.random.default_rng(0)
        first_obs = rng.uniform(-1, 1, size=ACTOR_OBS_DIM)
        hist = ObsHistory()
        stacked = hist.reset(first_obs)
        assert stacked.shape == (STACKED_ACTOR_OBS_DIM,)
        np.testing.assert_array_equal(stacked, np.tile(first_obs, ACTOR_HISTORY_LEN))

    def test_push_appends_newest_at_the_end(self):
        rng = np.random.default_rng(1)
        frame_a = rng.uniform(-1, 1, size=ACTOR_OBS_DIM)
        frame_b = rng.uniform(-1, 1, size=ACTOR_OBS_DIM)
        hist = ObsHistory()
        hist.reset(frame_a)
        stacked = hist.push(frame_b)
        np.testing.assert_array_equal(stacked[-ACTOR_OBS_DIM:], frame_b)

    def test_push_drops_oldest_frame(self):
        rng = np.random.default_rng(2)
        frame_a = rng.uniform(-1, 1, size=ACTOR_OBS_DIM)
        frames = [rng.uniform(-1, 1, size=ACTOR_OBS_DIM) for _ in range(6)]
        hist = ObsHistory()
        hist.reset(frame_a)
        stacked = None
        for f in frames:
            stacked = hist.push(f)
        expected = np.concatenate(frames[1:])  # frame_2 .. frame_6, frame_1 evicted
        np.testing.assert_array_equal(stacked, expected)

    def test_stacked_shape_is_120(self):
        rng = np.random.default_rng(3)
        hist = ObsHistory()
        stacked = hist.reset(rng.uniform(-1, 1, size=ACTOR_OBS_DIM))
        assert stacked.shape == (STACKED_ACTOR_OBS_DIM,)
        stacked = hist.push(rng.uniform(-1, 1, size=ACTOR_OBS_DIM))
        assert stacked.shape == (STACKED_ACTOR_OBS_DIM,)
