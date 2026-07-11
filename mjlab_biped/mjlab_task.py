"""
Real mjlab task definition for the biped balance policy (Phase 7.C).

COLAB-ONLY MODULE. Unlike every other file in `mjlab_biped/`, this one
imports the real `mjlab` package (and `torch` transitively) -- it is NOT
part of the Mac-side pure-Python mirror, and `mjlab_biped/__init__.py`
deliberately does not import it (same pattern already used for
`local_env.py`'s `sim` dependency, Phase 7.A). This file is only ever
imported from inside the Colab notebook, after `mjlab` has been installed
from `requirements-colab.txt`.

It translates the ALREADY-FROZEN, already-tested specs from the rest of
`mjlab_biped/` (entity.py, observations.py, rewards.py, terminations.py,
rl_cfg.py) into real mjlab manager-API objects and registers one task,
"Mjlab-Biped-Balance-v0", via mjlab's real `register_mjlab_task`.

Every import path and class/field name below was verified 2026-07-11
directly against mjlab's own source on GitHub (not guessed):
- src/mjlab/tasks/cartpole/cartpole_env_cfg.py's exact import block (the
  single source for every module path used here).
- src/mjlab/tasks/registry.py -- register_mjlab_task(task_id, env_cfg,
  play_env_cfg, rl_cfg, runner_cls=None) signature.
- src/mjlab/tasks/velocity/velocity_env_cfg.py -- field names for
  SceneCfg, ObservationGroupCfg ("terms"/"concatenate_terms"/
  "enable_corruption"), RewardTermCfg ("func"/"weight"/"params"),
  TerminationTermCfg ("func"/"params"/"time_out"), EventTermCfg
  ("func"/"mode"/"params"/"interval_range_s"), ActionTermCfg via
  JointPositionActionCfg ("entity_name"/"actuator_names"/"scale"/
  "use_default_offset"), SimulationCfg(mujoco=MujocoCfg(timestep=...)).

See docs/mjlab_adapter_notes.md for the full list of what remains
UNVERIFIED (things that cannot be checked without an actual mjlab
install + GPU) and the fallback plan if the Phase 7.C smoke-test cell
fails on one of them. The single biggest unknown: the exact attribute
path for reading raw MuJoCo <sensor> data (gyro, accelerometer, touch)
off an mjlab Entity at manager-function time -- every custom mdp function
below that touches sensor data has an inline `# UNVERIFIED:` comment.
"""

from __future__ import annotations

import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import joint_pos_rel, joint_vel_rel, time_out as mdp_time_out
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.tasks.registry import register_mjlab_task

# Re-use the already-frozen, already-tested pure-Python specs as the
# single source of truth for names/values -- this file only translates
# their SHAPE into real mjlab classes, it never redefines a number.
from mjlab_biped.entity import BipedEntityCfg
from mjlab_biped.rewards import RewardCfg
from mjlab_biped.terminations import TerminationCfg
from mjlab_biped.rl_cfg import RunnerCfg as OurRunnerCfg
from mjlab_biped.config import BipedEnvCfg as OurBipedEnvCfg

_biped = BipedEntityCfg()
_reward_cfg = RewardCfg()
_term_cfg = TerminationCfg()
_rl_cfg = OurRunnerCfg()
_env_cfg = OurBipedEnvCfg()

ROBOT_ENTITY_NAME = "robot"


def _get_biped_spec():
    """Load biped_warp.xml as an mjlab MjSpec. Path is relative to the
    Colab bundle root (see docs/colab_upload_manifest.md)."""
    import mujoco

    return mujoco.MjSpec.from_file(_biped.model_path)


BIPED_ENTITY_CFG = EntityCfg(
    spec_fn=_get_biped_spec,
    articulation=EntityArticulationInfoCfg(
        actuators=(
            XmlActuatorCfg(target_names_expr=tuple(_biped.actuators.target_names)),
        ),
    ),
    init_state=EntityCfg.InitialStateCfg(
        pos=_biped.init_state.base_pos,
        rot=_biped.init_state.base_quat,
        joint_pos=_biped.get_qpos_dict(),
        joint_vel=_biped.get_qvel_dict(),
    ),
)


# ---------------------------------------------------------------------------
# Custom mdp functions: gyro, projected_gravity, foot_touch, previous_action,
# velocity_command, accelerometer, base_linvel, base_height, com.
#
# joint_pos_rel / joint_vel_rel / time_out are mjlab BUILT-INS (imported
# above) -- not reimplemented here.
#
# VERIFIED 2026-07-11 against real mjlab 1.5.0 / mujoco-warp 3.10.0.1,
# installed and run locally on macOS CPU (Warp's CPU fallback -- no GPU
# needed for this, see docs/mjlab_adapter_notes.md's "local CPU testing"
# section). Raw MuJoCo <sensor> elements already baked into
# biped_warp.xml (torso_gyro, torso_acc, touch_l, touch_r, ...) are
# auto-discovered by mjlab.scene.Scene._add_sensors() and wrapped as
# BuiltinSensor.from_existing(name) -- read via `env.scene.sensors[name]
# .data`, NOT `entity.data.sensor_data[...]` (that attribute doesn't
# exist; confirmed via a live AttributeError). Sensor keys are prefixed
# with the entity name ("robot/torso_gyro", not "torso_gyro") -- also
# confirmed live via a direct `Scene(cfg.scene, device="cpu").sensors
# .keys()` inspection. Body-pose attributes on
# EntityData use a `_link_` infix mjlab's own Isaac-Lab-style naming
# convention doesn't drop: `root_link_pos_w`, `root_link_quat_w`,
# `root_link_lin_vel_w` (not `root_pos_w`/`root_quat_w`/`root_lin_vel_w`
# as originally guessed). Whole-robot CoM is `root_com_pos_w` (backed by
# MuJoCo's `subtree_com` at the root body -- the root's kinematic subtree
# is the entire robot, since it's the floating base).
# ---------------------------------------------------------------------------


def gyro(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    del asset_cfg
    return env.scene.sensors[f"{ROBOT_ENTITY_NAME}/torso_gyro"].data


def projected_gravity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # No direct EntityData attribute for this (unlike IsaacLab) -- mjlab
    # exposes body orientation but not a precomputed gravity projection,
    # so it's derived here the same way IsaacLab itself does: rotate the
    # world-frame gravity direction into the body frame via the inverse
    # of the root orientation quaternion. mjlab.utils.lab_api.math's
    # quat_apply_inverse is the verified real helper for this (used
    # internally by EntityData.root_com_lin_vel_b/.root_com_ang_vel_b).
    from mjlab.utils.lab_api.math import quat_apply_inverse

    entity = env.scene[asset_cfg.name]
    quat = entity.data.root_link_quat_w
    gravity_dir_w = torch.tensor([0.0, 0.0, -1.0], device=quat.device).expand(quat.shape[0], 3)
    return quat_apply_inverse(quat, gravity_dir_w)


def foot_touch(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    del asset_cfg
    touch_l = env.scene.sensors[f"{ROBOT_ENTITY_NAME}/touch_l"].data
    touch_r = env.scene.sensors[f"{ROBOT_ENTITY_NAME}/touch_r"].data
    return torch.cat([touch_l, touch_r], dim=-1)


def accelerometer(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    del asset_cfg
    return env.scene.sensors[f"{ROBOT_ENTITY_NAME}/torso_acc"].data


def base_linvel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    return entity.data.root_link_lin_vel_w


def base_height(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    return entity.data.root_link_pos_w[:, 2:3]


def com(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    return entity.data.root_com_pos_w


def previous_action(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    # Verified: env.action_manager.action / .prev_action are real
    # properties (mjlab/managers/action_manager.py's ActionManager).
    return env.action_manager.prev_action


def velocity_command(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    # 2026-07-11: make_biped_env_cfg() below does not (yet) pass a
    # `commands=` dict to ManagerBasedRlEnvCfg, so env.command_manager is
    # mjlab's NullCommandManager, whose get_command() always returns
    # None -- confirmed live (a real env construction prints
    # "<NullCommandManager> (inactive)"). Wiring a real mjlab
    # CommandTermCfg (resampling mjlab_biped.commands.sample_command's
    # logic each episode) is deferred to when Phase 7's curriculum
    # widens CommandRangeCfg past all-zero; until then this returns a
    # literal zero vector, which is exactly correct for the current
    # balance-first gate (CommandRangeCfg()'s frozen default is
    # vx/vy/yaw_rate all (0.0, 0.0)).
    del asset_cfg
    return torch.zeros(env.num_envs, 3, device=env.device)


# ---------------------------------------------------------------------------
# Reward terms -- direct translation of mjlab_biped/rewards.py's formulas.
# mjlab reward functions return an UNWEIGHTED per-env tensor; the weight
# lives on RewardTermCfg.weight, not inside the function (unlike our
# numpy compute_reward, which applies weights internally) -- so `weight=`
# below carries the exact values from RewardCfg()'s frozen defaults.
# ---------------------------------------------------------------------------


def alive_bonus_fn(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def upright_fn(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # 1 - 2*(qx^2 + qy^2), same closed-form as mjlab_biped.rewards.upright().
    entity = env.scene[asset_cfg.name]
    quat = entity.data.root_link_quat_w  # [w,x,y,z], MuJoCo convention
    qx, qy = quat[:, 1], quat[:, 2]
    return 1.0 - 2.0 * (qx**2 + qy**2)


def command_tracking_fn(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # See velocity_command()'s comment: command is a literal zero until
    # a real CommandTermCfg is wired in for Phase 7's curriculum.
    entity = env.scene[asset_cfg.name]
    lin_vel = entity.data.root_link_lin_vel_w[:, :2]
    cmd = torch.zeros_like(lin_vel)
    return -torch.norm(lin_vel - cmd, dim=-1)


def control_effort_fn(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    del asset_cfg
    action = env.action_manager.action
    return -torch.sum(action**2, dim=-1)


def action_rate_fn(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    del asset_cfg
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    return -torch.sum((action - prev) ** 2, dim=-1)


# ---------------------------------------------------------------------------
# Termination terms -- direct translation of mjlab_biped/terminations.py.
# ---------------------------------------------------------------------------


def fall_tilt_fn(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return upright_fn(env, asset_cfg) < _term_cfg.tilt_threshold


def fall_height_fn(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    height = entity.data.root_link_pos_w[:, 2]
    return height < _term_cfg.height_threshold


# ---------------------------------------------------------------------------
# Observation groups. Actor = 5 terms per frame; single-frame dim is
# 28 (not the originally assumed 24 -- joint_pos_rel/joint_vel_rel each
# report all 8 hinge joints, 6 actuated + 2 passive ankles, not just the
# 6 actuated ones; confirmed 2026-07-11 via a real local mjlab env).
#
# History stacking VERIFIED 2026-07-11 (local CPU mjlab install, not
# guessed): mjlab's real ObservationGroupCfg has a native
# `history_length: int | None` field (mjlab/managers/observation_manager
# .py) -- a group-level override applied to every term in the group, with
# `flatten_history_dim=True` (the default) giving shape
# (num_envs, obs_dim * history_length). No custom wrapper or LSTM
# fallback needed (items 1-2 of the fallback ladder in
# docs/mjlab_adapter_notes.md are moot). Stacked actor dim is therefore
# 28 * 5 = 140, not the originally frozen 120 -- see
# mjlab_biped/observations.py's updated STACKED_ACTOR_OBS_DIM. Critic =
# single-frame 43-dim, all 11 terms, no stacking (history_length=None,
# the field's default, leaves per-term history_length=0 in effect).
# ---------------------------------------------------------------------------

_robot_scene_cfg = SceneEntityCfg(ROBOT_ENTITY_NAME)

ACTOR_OBS_GROUP = ObservationGroupCfg(
    terms={
        "joint_pos_rel": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": _robot_scene_cfg}),
        "joint_vel_rel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": _robot_scene_cfg}),
        "gyro": ObservationTermCfg(func=gyro, params={"asset_cfg": _robot_scene_cfg}),
        "previous_action": ObservationTermCfg(func=previous_action, params={}),
        "velocity_command": ObservationTermCfg(func=velocity_command, params={}),
    },
    concatenate_terms=True,
    enable_corruption=False,  # noise/DR robustness is a later Phase 7 stage
    history_length=5,
)

CRITIC_OBS_GROUP = ObservationGroupCfg(
    terms={
        "joint_pos_rel": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": _robot_scene_cfg}),
        "joint_vel_rel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": _robot_scene_cfg}),
        "gyro": ObservationTermCfg(func=gyro, params={"asset_cfg": _robot_scene_cfg}),
        "previous_action": ObservationTermCfg(func=previous_action, params={}),
        "velocity_command": ObservationTermCfg(func=velocity_command, params={}),
        "projected_gravity": ObservationTermCfg(func=projected_gravity, params={"asset_cfg": _robot_scene_cfg}),
        "foot_touch": ObservationTermCfg(func=foot_touch, params={"asset_cfg": _robot_scene_cfg}),
        "accelerometer": ObservationTermCfg(func=accelerometer, params={"asset_cfg": _robot_scene_cfg}),
        "base_linvel": ObservationTermCfg(func=base_linvel, params={"asset_cfg": _robot_scene_cfg}),
        "base_height": ObservationTermCfg(func=base_height, params={"asset_cfg": _robot_scene_cfg}),
        "com": ObservationTermCfg(func=com, params={"asset_cfg": _robot_scene_cfg}),
    },
    concatenate_terms=True,
    enable_corruption=False,
)


# ---------------------------------------------------------------------------
# Reward + termination managers.
# ---------------------------------------------------------------------------

REWARD_TERMS = {
    "alive_bonus": RewardTermCfg(func=alive_bonus_fn, weight=_reward_cfg.alive_bonus_weight, params={}),
    "upright": RewardTermCfg(func=upright_fn, weight=_reward_cfg.upright_weight, params={"asset_cfg": _robot_scene_cfg}),
    "command_tracking": RewardTermCfg(func=command_tracking_fn, weight=_reward_cfg.command_tracking_weight, params={"asset_cfg": _robot_scene_cfg}),
    "control_effort": RewardTermCfg(func=control_effort_fn, weight=_reward_cfg.control_effort_weight, params={}),
    "action_rate": RewardTermCfg(func=action_rate_fn, weight=_reward_cfg.action_rate_weight, params={}),
}

TERMINATION_TERMS = {
    "fall_tilt": TerminationTermCfg(func=fall_tilt_fn, params={"asset_cfg": _robot_scene_cfg}),
    "fall_height": TerminationTermCfg(func=fall_height_fn, params={"asset_cfg": _robot_scene_cfg}),
    "time_out": TerminationTermCfg(func=mdp_time_out, time_out=True, params={}),
}

# ---------------------------------------------------------------------------
# Actions: 6 position-target actuators, matching entity.py's ActuatorConfig.
# use_default_offset=False since our targets are absolute joint angles
# (radians), not deltas from the stand-keyframe default pose.
# ---------------------------------------------------------------------------

ACTION_CFG = JointPositionActionCfg(
    entity_name=ROBOT_ENTITY_NAME,
    actuator_names=tuple(_biped.actuators.target_names),
    scale=1.0,
    use_default_offset=False,
)


# ---------------------------------------------------------------------------
# Scene + sim + full env cfg assembly.
# ---------------------------------------------------------------------------

SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=_biped.timestep,
        integrator=_biped.integrator,
    ),
)


def make_biped_env_cfg(num_envs: int = _env_cfg.scene.num_envs) -> ManagerBasedRlEnvCfg:
    """Build the training env cfg. `num_envs` overridable at train time
    via `--env.scene.num-envs N` (tyro CLI, see docs/colab_upload_manifest.md)."""
    scene_cfg = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={ROBOT_ENTITY_NAME: BIPED_ENTITY_CFG},
        num_envs=num_envs,
        env_spacing=2.0,
    )
    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        observations={"actor": ACTOR_OBS_GROUP, "critic": CRITIC_OBS_GROUP},
        actions={"joint_pos": ACTION_CFG},
        # CRITICAL FIX 2026-07-11: `events={}` (the original value here)
        # silently disabled mjlab's own default `reset_scene_to_default`
        # event -- the ONLY mechanism that applies `BIPED_ENTITY_CFG
        # .init_state` (the stand-pose position/orientation/joint angles,
        # entity.py's STAND_HEIGHT=0.2030) to the entity at reset. With
        # events={}, every episode spawned at raw qpos=0 (z=0, not
        # 0.203), instantly failed `fall_height_fn`'s height<0.15 check,
        # and mjlab's `auto_reset=True` (the default) reset the env again
        # on the very next call -- confirmed live via a local CPU run:
        # `terminated=True` on literally every single step, forever, so
        # no episode could ever run longer than one transition and no
        # policy could ever learn anything. DR/init-noise (Phase 6.C)
        # is still deferred -- this only restores the required default.
        events={"reset_scene_to_default": EventTermCfg(func=reset_scene_to_default, mode="reset")},
        rewards=REWARD_TERMS,
        terminations=TERMINATION_TERMS,
        sim=SIM_CFG,
        decimation=_biped.control_decimation,
        episode_length_s=_env_cfg.episode_length_s,
    )


def make_play_env_cfg() -> ManagerBasedRlEnvCfg:
    return make_biped_env_cfg(num_envs=4)


# ---------------------------------------------------------------------------
# RSL-RL runner cfg: translate our frozen mjlab_biped/rl_cfg.py values into
# mjlab's real RslRlOnPolicyRunnerCfg/RslRlModelCfg/RslRlPpoAlgorithmCfg.
#
# VERIFIED 2026-07-11 (was UNVERIFIED; caught live via a real Colab
# TypeError, then confirmed against src/mjlab/rl/config.py directly).
# The structure is genuinely different from the IsaacLab-docs-derived
# guess this file originally shipped with, not just renamed fields:
#   - RslRlOnPolicyRunnerCfg has SEPARATE `actor: RslRlModelCfg` and
#     `critic: RslRlModelCfg` fields -- there is no single combined
#     "policy" config with actor_hidden_dims/critic_hidden_dims.
#   - RslRlModelCfg's real fields are `hidden_dims` (singular, one
#     network's own dims), `activation`, `obs_normalization` (singular
#     bool), plus `cnn_cfg`, `distribution_cfg`, `rnn_type`,
#     `rnn_hidden_dim`, `rnn_num_layers`, `class_name`.
#   - `init_noise_std` is NOT a direct field at all -- it lives inside
#     `distribution_cfg["init_std"]`, and only the ACTOR needs a
#     distribution_cfg (the critic just outputs a scalar value estimate,
#     no action distribution).
#   - `obs_groups` values are tuples, e.g. {"actor": ("actor",),
#     "critic": ("critic",)} (RslRlBaseRunnerCfg's own default), not
#     lists.
# See mjlab_biped/rl_cfg.py's module docstring for the same correction
# applied to our Mac-side pure-Python mirror.
# ---------------------------------------------------------------------------

MJLAB_RL_CFG = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=_rl_cfg.num_steps_per_env,
    max_iterations=_rl_cfg.max_iterations,
    save_interval=_rl_cfg.save_interval,
    obs_groups=_rl_cfg.obs_groups,
    actor=RslRlModelCfg(
        hidden_dims=tuple(_rl_cfg.actor.hidden_dims),
        activation=_rl_cfg.actor.activation,
        obs_normalization=_rl_cfg.actor.obs_normalization,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": _rl_cfg.actor.init_noise_std,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=tuple(_rl_cfg.critic.hidden_dims),
        activation=_rl_cfg.critic.activation,
        obs_normalization=_rl_cfg.critic.obs_normalization,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        num_learning_epochs=_rl_cfg.algorithm.num_learning_epochs,
        num_mini_batches=_rl_cfg.algorithm.num_mini_batches,
        learning_rate=_rl_cfg.algorithm.learning_rate,
        schedule=_rl_cfg.algorithm.schedule,
        gamma=_rl_cfg.algorithm.gamma,
        lam=_rl_cfg.algorithm.lam,
        entropy_coef=_rl_cfg.algorithm.entropy_coef,
        desired_kl=_rl_cfg.algorithm.desired_kl,
        max_grad_norm=_rl_cfg.algorithm.max_grad_norm,
        value_loss_coef=_rl_cfg.algorithm.value_loss_coef,
        use_clipped_value_loss=_rl_cfg.algorithm.use_clipped_value_loss,
        clip_param=_rl_cfg.algorithm.clip_param,
    ),
)

TASK_ID = "Mjlab-Biped-Balance-v0"

register_mjlab_task(
    TASK_ID,
    env_cfg=make_biped_env_cfg(),
    play_env_cfg=make_play_env_cfg(),
    rl_cfg=MJLAB_RL_CFG,
)
