#!/usr/bin/env python3
"""Run a trained mjlab/RSL-RL checkpoint through the local mjpython viewer.

Loads a real RSL-RL actor checkpoint (.pt) trained via
notebooks/train_biped.ipynb on Colab, reconstructs its exact observation
format entirely outside mjlab/MuJoCo Warp, and drives the CPU BipedSim
(models/mjcf/biped.xml) with it in an interactive mujoco.viewer -- the same
viewer pattern as scripts/visualize.py -- so the trained balance policy can
be watched directly, independent of the training harness.

Needs `torch` (see pyproject.toml's `eval` optional dependency group:
`uv sync --extra eval`). Does NOT need mjlab/rsl-rl-lib/mujoco-warp: the
actor is a tiny 4-layer MLP, reimplemented directly below from the
checkpoint's own state dict rather than depending on the real rsl_rl
package, so this stays a lightweight, Mac-friendly, GPU-free script.

Every constant below (joint order, action order, obs composition, scale/
offset, checkpoint architecture, per-term history stacking, the
one-step-delayed previous_action, the normalizer's +eps) was verified
empirically against a live local mjlab env and the checkpoint's own state
dict on 2026-07-13, not guessed -- see docs/mjlab_adapter_notes.md and
CLAUDE.md's Phase 7.D notes for the full provenance. Getting any of these
wrong would silently feed the trained policy a corrupted observation or
misroute its action outputs to the wrong joints; nothing here is a
placeholder. Cross-validated bit-exact against mjlab's own real inference
machinery (a live `MjlabOnPolicyRunner.get_inference_policy()` call
against the same checkpoint): the first action this script produces
matches mjlab's own first action to displayed precision.

Known, expected divergence after the first few steps: this script drives
`models/mjcf/biped.xml` (full-mesh collision, the CPU/deployment-
representative model), while the checkpoint was trained against
`biped_warp.xml` (primitive-proxy collision, Phase 6.0's Warp-compatible
variant). The two have different contact dynamics by design (documented
in docs/warp_model.md) -- a policy that survives indefinitely on the
Warp/training model may still fall sooner here. That is itself a
meaningful sim-to-sim transfer finding, not a bug in this script; if it
happens, watch the actuator action magnitudes -- physically saturating,
sign-oscillating outputs are a sign the policy hasn't yet learned robust
recovery, most likely because it is still early in training (max
training budget is 1500 iterations; a much lower iteration count means
substantially less trained).

Usage:
    uv run --extra eval mjpython scripts/eval_checkpoint.py \
        "/Users/rahulbouri/Downloads/Model 499.pt"

(`mjpython`, not plain `python`, is required on macOS for the interactive
mujoco.viewer GUI -- same requirement as scripts/visualize.py.)
"""

import argparse
import os
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim import BipedSim

MODEL_PATH = "models/mjcf/biped.xml"

# Joint order used by mjlab's built-in joint_pos_rel/joint_vel_rel
# observation terms (SceneEntityCfg("robot") with no joint filter -> the
# full entity.joint_names order, which mjlab resolves from the compiled
# model's own joint declaration order -- NOT alphabetical, NOT the order
# mjlab_biped/entity.py's ActuatorConfig.target_names lists).
#
# Verified live 2026-07-13 two independent ways: (1) printed
# `entity.joint_names` from a real local mjlab env built against
# biped_warp.xml; (2) printed MuJoCo's own joint list from biped.xml
# (models/mjcf/biped.xml) directly via mj_id2name -- both report this
# exact 8-joint sequence, confirming biped.xml and biped_warp.xml share
# the same joint order (expected, same pipeline/URDF) and that this
# script can safely read from biped.xml with no remapping.
OBS_JOINT_ORDER = (
    "ankle_l", "foot_l", "knee_l", "hip_roll_l",
    "hip_roll_r", "knee_r", "ankle_r", "foot_r",
)

# The 6 ACTUATED joints, in the same relative order as OBS_JOINT_ORDER with
# the 2 passive ankle joints (foot_l/foot_r) removed. mjlab's ActionManager
# is built with preserve_order=False (mjlab_task.py's ACTION_CFG), which
# re-sorts the configured actuator_names to match the model's internal
# joint order rather than preserving the tuple order passed in -- verified
# live by inspecting env.action_manager._terms["joint_pos"]._target_names
# against a real env, which returned exactly this sequence, NOT
# mjlab_biped/entity.py's declared (hip_roll_l, knee_l, ankle_l, ...) order.
ACTION_JOINT_ORDER = (
    "ankle_l", "knee_l", "hip_roll_l",
    "hip_roll_r", "knee_r", "ankle_r",
)

ACTOR_HISTORY_LEN = 5
ACTOR_FRAME_DIM = 2 * len(OBS_JOINT_ORDER) + 3 + len(ACTION_JOINT_ORDER) + 3  # 8+8+3+6+3=28
ACTOR_STACKED_DIM = ACTOR_FRAME_DIM * ACTOR_HISTORY_LEN  # 140
ACTOR_HIDDEN_DIMS = (512, 256, 128)  # frozen in mjlab_biped/rl_cfg.py's ModelCfg

# Fall thresholds mirrored from mjlab_biped/terminations.py's TerminationCfg
# -- used only for this script's own diagnostic printout, so an eval run
# reports "fallen" using the exact same definition training optimized
# against.
TILT_THRESHOLD = 0.5
HEIGHT_THRESHOLD = 0.15


def upright(quat_wxyz):
    """1 - 2*(qx^2+qy^2); identical closed form to mjlab_biped.rewards.upright()
    and mjlab_task.py's upright_fn (verified against that source directly)."""
    qx, qy = quat_wxyz[1], quat_wxyz[2]
    return 1.0 - 2.0 * (qx**2 + qy**2)


def build_actor_mlp():
    dims = [ACTOR_STACKED_DIM, *ACTOR_HIDDEN_DIMS, len(ACTION_JOINT_ORDER)]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ELU())
    return torch.nn.Sequential(*layers)


# RSL-RL's EmpiricalNormalization default (rsl_rl/modules/normalization.py:
# `def __init__(self, shape, eps: float = 1e-2, ...)`). Not a registered
# buffer, so it does NOT appear in the checkpoint's state dict at all --
# and it's not exposed anywhere in mjlab's own config surface either
# (grepped mjlab/rl/config.py and mjlab_biped/rl_cfg.py/mjlab_task.py: no
# `eps` field exists to override it), so the checkpoint was necessarily
# normalized with this exact default during training. Confirmed necessary,
# not optional: velocity_command is literally always [0, 0, 0] throughout
# training (no CommandTermCfg wired), so its 15 history-stacked dims have
# EXACTLY zero variance in the checkpoint's own obs_normalizer._std --
# dividing by std directly (without +eps) produces 0/0 = NaN immediately,
# confirmed by reproducing the exact NaN on the very first inference call
# before this fix was found.
NORMALIZER_EPS = 1e-2


class CheckpointPolicy:
    """Deterministic-inference wrapper around an RSL-RL actor checkpoint.

    Reconstructs the actor's MLP (verified 2026-07-13 against a real
    checkpoint's state dict: mlp.{0,2,4,6}.{weight,bias}, ELU activations
    between, no activation on the final layer -- the raw output is the
    Gaussian policy's mean action, used directly here rather than sampling,
    since this script is for evaluating/verifying learned behavior, not
    exploring) and its observation normalizer (obs_normalizer._mean/_std,
    shape (1, 140) in the checkpoint -- confirms normalization is applied
    to the full stacked 140-dim vector as a whole, not per-frame).
    """

    def __init__(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor_sd = ckpt["actor_state_dict"]

        mlp_sd = {
            k[len("mlp."):]: v for k, v in actor_sd.items() if k.startswith("mlp.")
        }
        self.mlp = build_actor_mlp()
        self.mlp.load_state_dict(mlp_sd)
        self.mlp.eval()

        self.obs_mean = actor_sd["obs_normalizer._mean"].squeeze(0).float()
        self.obs_std = actor_sd["obs_normalizer._std"].squeeze(0).float()
        assert self.obs_mean.shape == (ACTOR_STACKED_DIM,), self.obs_mean.shape
        assert self.obs_std.shape == (ACTOR_STACKED_DIM,), self.obs_std.shape

        self.iteration = ckpt.get("iter")

    @torch.no_grad()
    def act(self, stacked_obs):
        obs = torch.as_tensor(stacked_obs, dtype=torch.float32)
        normed = (obs - self.obs_mean) / (self.obs_std + NORMALIZER_EPS)
        action = self.mlp(normed.unsqueeze(0)).squeeze(0)
        return action.numpy().astype(np.float64)


class TermHistory:
    """Oldest -> newest ring buffer over ACTOR_HISTORY_LEN copies of ONE
    term's raw value (NOT a whole 28-dim frame -- see ActorObsBuilder's
    docstring for why history stacking must be done per-term)."""

    def __init__(self):
        self._buf = []

    def reset(self, value):
        self._buf = [np.asarray(value, dtype=np.float64).copy() for _ in range(ACTOR_HISTORY_LEN)]

    def push(self, value):
        self._buf.pop(0)
        self._buf.append(np.asarray(value, dtype=np.float64).copy())

    def stack(self):
        return np.concatenate(self._buf)


class ActorObsBuilder:
    """Builds the real 140-dim actor observation vector.

    Two non-obvious things had to be verified empirically (2026-07-13,
    against a live local mjlab env) before this could be written correctly
    -- both would silently corrupt the policy's input if gotten wrong,
    with no crash or error to signal it:

    1. **History is stacked per-TERM, not per-FRAME.** mjlab's real
       history mechanism (mjlab/managers/observation_manager.py) keeps a
       separate CircularBuffer per observation term and concatenates each
       term's own N-frame history, THEN concatenates terms in group
       order. The real layout is therefore
       [joint_pos_rel_hist(40), joint_vel_rel_hist(40), gyro_hist(15),
       previous_action_hist(30), velocity_command_hist(15)] = 140 --
       NOT 5 concatenated 28-dim frames as the printed observation-manager
       table's per-term "(140) <- 5x(28,)"-style rows might suggest at a
       glance (that table reports each TERM's own history size, not a
       frame-level grouping). Confirmed by tracing a live env: after
       feeding 5 distinct actions, the resulting values landed at offsets
       consistent with this term-major layout, not frame-major.
    2. **previous_action is delayed by an extra step.**
       mjlab_task.py's previous_action() returns
       `env.action_manager.prev_action`, not `.action`. Traced 10 steps
       against a live env: the newest previous_action history slot after
       applying action_N always equals the action applied at step N-1,
       never action_N itself. Replicated here via `_pending_prev_action`,
       which is pushed into history BEFORE being overwritten with the
       action just taken -- so it always lags by exactly one step, same
       as the real training-time observation the checkpoint learned from.

    joint_pos_rel/joint_vel_rel equal the raw pos_*/vel_* sensors directly
    for this model (mjlab's real versions subtract each joint's default
    pos/vel, which are 0 for every joint here -- verified against
    entity.data.default_joint_pos/default_joint_vel). velocity_command is
    always zero (no CommandTermCfg wired -- the Phase 7 zero-command
    balance gate).
    """

    def __init__(self):
        self.pos_hist = TermHistory()
        self.vel_hist = TermHistory()
        self.gyro_hist = TermHistory()
        self.prevact_hist = TermHistory()
        self.cmd_hist = TermHistory()
        self._pending_prev_action = np.zeros(len(ACTION_JOINT_ORDER), dtype=np.float64)

    @staticmethod
    def _read_pos_vel_gyro(sensors):
        pos = np.array([sensors[f"pos_{j}"][0] for j in OBS_JOINT_ORDER], dtype=np.float64)
        vel = np.array([sensors[f"vel_{j}"][0] for j in OBS_JOINT_ORDER], dtype=np.float64)
        gyro = np.asarray(sensors["torso_gyro"], dtype=np.float64)
        return pos, vel, gyro

    def reset(self, sensors):
        self._pending_prev_action = np.zeros(len(ACTION_JOINT_ORDER), dtype=np.float64)
        pos, vel, gyro = self._read_pos_vel_gyro(sensors)
        self.pos_hist.reset(pos)
        self.vel_hist.reset(vel)
        self.gyro_hist.reset(gyro)
        self.prevact_hist.reset(self._pending_prev_action)
        self.cmd_hist.reset(np.zeros(3))
        return self.stack()

    def step(self, sensors, action_just_taken):
        pos, vel, gyro = self._read_pos_vel_gyro(sensors)
        self.pos_hist.push(pos)
        self.vel_hist.push(vel)
        self.gyro_hist.push(gyro)
        self.prevact_hist.push(self._pending_prev_action)  # one-step-delayed, see class docstring
        self.cmd_hist.push(np.zeros(3))
        self._pending_prev_action = np.asarray(action_just_taken, dtype=np.float64).copy()
        return self.stack()

    def stack(self):
        return np.concatenate([
            self.pos_hist.stack(), self.vel_hist.stack(), self.gyro_hist.stack(),
            self.prevact_hist.stack(), self.cmd_hist.stack(),
        ])


def build_ctrl(model, action_by_joint):
    """Maps the policy's 6-dim output (ACTION_JOINT_ORDER) to model.nu-length
    ctrl, by actuator NAME (act_<joint>) -- not by positional index, since
    biped.xml's own actuator declaration order differs yet again from both
    OBS_JOINT_ORDER and ACTION_JOINT_ORDER. scale=1.0, offset=0.0,
    encoder_bias=0 in the real training config, so the raw MLP output is
    used directly as the radian position target with no rescaling."""
    ctrl = np.zeros(model.nu, dtype=np.float64)
    for joint_name, value in zip(ACTION_JOINT_ORDER, action_by_joint):
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{joint_name}")
        assert act_id >= 0, f"actuator 'act_{joint_name}' not found in {model}"
        ctrl[act_id] = value
    return ctrl


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Path to the RSL-RL actor checkpoint (.pt)")
    parser.add_argument("--model", default=MODEL_PATH, help=f"MJCF model path (default: {MODEL_PATH})")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    print("=" * 60)
    print("Biped Checkpoint Evaluation Viewer")
    print("=" * 60)
    policy = CheckpointPolicy(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Training iteration: {policy.iteration}")
    print(f"  Actor: {ACTOR_STACKED_DIM} -> {ACTOR_HIDDEN_DIMS} -> {len(ACTION_JOINT_ORDER)}")

    sim = BipedSim(args.model, control_dt=0.02, suspended=False)

    obs_builder = ActorObsBuilder()
    state = {}

    def do_reset():
        sim.reset(keyframe="stand")
        state["stacked"] = obs_builder.reset(sim.sensors())
        state["step"] = 0

    do_reset()

    paused = {"value": True}
    policy_enabled = {"value": True}

    def key_callback(keycode):
        if 0 <= keycode < 256:
            ch = chr(keycode)
            if ch == " ":
                paused["value"] = not paused["value"]
                print("Physics:", "RUNNING" if not paused["value"] else "PAUSED")
            elif ch.lower() == "p":
                policy_enabled["value"] = not policy_enabled["value"]
                print("Policy:", "ON (trained checkpoint)" if policy_enabled["value"]
                      else "OFF (holding stand-pose target, zero action)")
            elif ch.lower() == "r":
                do_reset()
                print("Reset to stand pose.")

    print("\nControls:")
    print("  Mouse: Rotate (left), Pan (right); Scroll: Zoom")
    print("  Space: Pause/Resume physics")
    print("  P: Toggle policy ON (trained checkpoint) / OFF (dumb zero-target hold)")
    print("  R: Reset to stand pose")
    print("  Close window to exit")
    print("\nStarted PAUSED. Press SPACE to begin.\n")

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if not paused["value"]:
                if policy_enabled["value"]:
                    action = policy.act(state["stacked"])
                else:
                    action = np.zeros(len(ACTION_JOINT_ORDER), dtype=np.float64)

                ctrl = build_ctrl(sim.model, action)
                sim.step(ctrl)
                # BipedSim.step() does not call mj_forward() after its last
                # substep, so sensordata read immediately afterward is one
                # substep stale relative to the new qpos/qvel (the same
                # MuJoCo pipeline-timing characteristic Phase 3 documented:
                # mj_step computes sensors from the state at the START of
                # the step). mjlab's real env.step() explicitly resyncs
                # with an extra forward() call before computing
                # observations -- replicate that here so this script reads
                # the same freshness the checkpoint was trained against.
                mujoco.mj_forward(sim.model, sim.data)

                state["stacked"] = obs_builder.step(sim.sensors(), action)
                state["step"] += 1

                if state["step"] % 50 == 0:  # ~1s at 50 Hz control rate
                    sensors = sim.sensors()
                    quat = sensors["torso_quat"]
                    height = sensors["torso_pos"][2]
                    up = upright(quat)
                    fallen = (up < TILT_THRESHOLD) or (height < HEIGHT_THRESHOLD)
                    print(f"t={state['step']/50:6.1f}s  height={height:.3f}m  "
                          f"upright={up:+.3f}  {'*** FALLEN ***' if fallen else 'standing'}")

            viewer.sync()


if __name__ == "__main__":
    main()
