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
# UNVERIFIED: the exact attribute for raw <sensor> readout on an mjlab
# Entity. Best-guess convention (matches the joint_pos/joint_vel pattern
# of env.scene[name].data.<field>): `env.scene[asset_cfg.name].data
# .sensor_data["<sensor_name>"]`. If the Phase 7.C smoke-test cell fails
# here, check `dir(env.scene["robot"].data)` interactively in Colab and
# fix these three functions before touching anything else in this file.
# ---------------------------------------------------------------------------


def gyro(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    return entity.data.sensor_data["torso_gyro"]  # UNVERIFIED: attribute path


def projected_gravity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    return entity.data.projected_gravity_b  # UNVERIFIED: attribute name


def foot_touch(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    touch_l = entity.data.sensor_data["touch_l"]  # UNVERIFIED
    touch_r = entity.data.sensor_data["touch_r"]  # UNVERIFIED
    return torch.cat([touch_l, touch_r], dim=-1)


def accelerometer(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    return entity.data.sensor_data["torso_acc"]  # UNVERIFIED


def base_linvel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    return entity.data.root_lin_vel_w  # UNVERIFIED: attribute name


def base_height(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    return entity.data.root_pos_w[:, 2:3]  # UNVERIFIED: attribute name


def com(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    return entity.data.com_pos_w  # UNVERIFIED: attribute name


def previous_action(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    # mjlab's ManagerBasedRlEnv keeps the last applied action on
    # env.action_manager.action -- matches the Isaac-Lab-style manager
    # convention this API mirrors (cartpole/velocity examples confirm
    # env.action_manager exists; the exact attribute name for the
    # PREVIOUS (not current) action buffer is UNVERIFIED -- check
    # `env.action_manager.prev_action` vs `.action` if this errors).
    return env.action_manager.prev_action  # UNVERIFIED: attribute name


def velocity_command(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    # UNVERIFIED: mjlab's command manager attribute path. Best guess,
    # matching the "actor obs reads the current command" pattern from
    # velocity_env_cfg.py's own observation term for command tracking.
    return env.command_manager.get_command("base_velocity")


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
    quat = entity.data.root_quat_w  # UNVERIFIED: attribute name, [w,x,y,z] assumed
    qx, qy = quat[:, 1], quat[:, 2]
    return 1.0 - 2.0 * (qx**2 + qy**2)


def command_tracking_fn(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    lin_vel = entity.data.root_lin_vel_w[:, :2]  # UNVERIFIED
    cmd = env.command_manager.get_command("base_velocity")[:, :2]
    return -torch.norm(lin_vel - cmd, dim=-1)


def control_effort_fn(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    action = env.action_manager.action  # UNVERIFIED: attribute name
    return -torch.sum(action**2, dim=-1)


def action_rate_fn(env, asset_cfg: SceneEntityCfg = None) -> torch.Tensor:
    action = env.action_manager.action  # UNVERIFIED
    prev = env.action_manager.prev_action  # UNVERIFIED
    return -torch.sum((action - prev) ** 2, dim=-1)


# ---------------------------------------------------------------------------
# Termination terms -- direct translation of mjlab_biped/terminations.py.
# ---------------------------------------------------------------------------


def fall_tilt_fn(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return upright_fn(env, asset_cfg) < _term_cfg.tilt_threshold


def fall_height_fn(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.scene[asset_cfg.name]
    height = entity.data.root_pos_w[:, 2]  # UNVERIFIED
    return height < _term_cfg.height_threshold


# ---------------------------------------------------------------------------
# Observation groups. Actor = 5 terms per frame (24-dim, per
# docs/observation_spec.md v1); history stacking to 120-dim is mjlab's
# job via ObservationTermCfg's history-length mechanism -- UNVERIFIED
# exactly how (native `history_length` kwarg vs a wrapper); this is the
# explicit Sonnet pre-verification item CLAUDE.md's Sub-task 7.B flagged
# for "before dispatching 7.C" -- check on Colab, then set the kwarg
# below (currently a placeholder comment, not a real field, until
# confirmed). Critic = single-frame 39-dim, all 11 terms, no stacking.
# ---------------------------------------------------------------------------

_robot_scene_cfg = SceneEntityCfg(ROBOT_ENTITY_NAME)

ACTOR_OBS_GROUP = ObservationGroupCfg(
    terms={
        "joint_pos_rel": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": _robot_scene_cfg}),
        "joint_vel_rel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": _robot_scene_cfg}),
        "gyro": ObservationTermCfg(func=gyro, params={"asset_cfg": _robot_scene_cfg}),
        "previous_action": ObservationTermCfg(func=previous_action, params={}),
        "velocity_command": ObservationTermCfg(func=velocity_command, params={}),
        # TODO(7.C, Colab-verify): add history_length=5 (or equivalent)
        # once mjlab's actual stacking mechanism is confirmed empirically.
    },
    concatenate_terms=True,
    enable_corruption=False,  # noise/DR robustness is a later Phase 7 stage
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
        events={},  # DR/init-noise wiring deferred -- see docs/mjlab_adapter_notes.md
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
