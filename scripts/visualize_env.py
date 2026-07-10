#!/usr/bin/env python3
"""
Interactively visualize the Phase 6 RL environment (BipedLocalEnv) in the
MuJoCo viewer: observations, reward, and termination logic all wired up
and running against real physics, ahead of the actual Colab training run.

Run with `mjpython` (not plain `python`) on macOS - required for the
interactive `mujoco.viewer` GUI, same as `scripts/visualize.py`.

Usage:
    uv run mjpython scripts/visualize_env.py                # zero action
    uv run mjpython scripts/visualize_env.py --sine          # small sine test signal
    uv run mjpython scripts/visualize_env.py --random        # small random actions
    uv run mjpython scripts/visualize_env.py --dr            # use the training cfg (DR on) instead of the play cfg
    uv run mjpython scripts/visualize_env.py --seed 123
"""
import argparse
import os
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from mjlab_biped.config import BipedEnvCfg, make_play_env_cfg
from mjlab_biped.local_env import BipedLocalEnv


def make_action_fn(mode: str, rng: np.random.Generator):
    """Return a fn(step_count) -> (6,) action array for the chosen test signal."""
    if mode == "zero":
        return lambda t: np.zeros(6)
    if mode == "sine":
        # Small-amplitude, low-frequency sine per joint so the actuators'
        # response is clearly visible without immediately toppling the robot.
        freqs = np.array([0.3, 0.4, 0.5, 0.3, 0.4, 0.5])
        phases = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        amplitude = 0.15  # rad

        def sine_action(t):
            control_dt = 0.02
            time_s = t * control_dt
            return amplitude * np.sin(2 * np.pi * freqs * time_s + phases)

        return sine_action
    if mode == "random":
        amplitude = 0.2  # rad, small enough not to immediately destabilize

        def random_action(t):
            return rng.uniform(-amplitude, amplitude, size=6)

        return random_action
    raise ValueError(f"Unknown action mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=["zero", "sine", "random"], default="zero",
        help="Test action signal to drive the 6 actuators with (default: zero).",
    )
    parser.add_argument("--sine", action="store_const", const="sine", dest="action")
    parser.add_argument("--random", action="store_const", const="random", dest="action")
    parser.add_argument(
        "--dr", action="store_true",
        help="Use the full training BipedEnvCfg (domain randomization ON) "
             "instead of the play variant (DR off, the default here).",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    env_cfg = BipedEnvCfg() if args.dr else make_play_env_cfg()
    env = BipedLocalEnv(env_cfg=env_cfg, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    action_fn = make_action_fn(args.action, rng)

    obs = env.reset()

    print("=" * 70)
    print("Biped RL Environment Visualizer (Phase 6.E)")
    print("=" * 70)
    print(f"Env config:        {'training (DR ON)' if args.dr else 'play (DR OFF)'}")
    print(f"Action signal:     {args.action}")
    print(f"Actor obs dim:     {obs.shape[0]}")
    print(f"Velocity command:  {env.command}  (vx, vy, yaw_rate)")
    print(f"Episode length:    {env.env_cfg.termination_cfg.max_episode_steps} steps "
          f"({env.env_cfg.episode_length_s} s)")
    print()
    print("Controls:")
    print("  Mouse: Rotate view (left), Pan (right)")
    print("  Scroll: Zoom")
    print("  Space: Pause/Resume")
    print("  R: Manually trigger a reset")
    print("  Close window to exit")
    print()
    print("Started PAUSED. Press SPACE to run the environment loop.")
    print("Live stats print every 25 control steps, and on every reset/termination.")
    print("=" * 70)
    print()

    state = {"paused": True, "reset_requested": False}

    def key_callback(keycode):
        if 0 <= keycode < 256:
            ch = chr(keycode)
            if ch == " ":
                state["paused"] = not state["paused"]
                print("Env loop:", "RUNNING" if not state["paused"] else "PAUSED")
            elif ch.lower() == "r":
                state["reset_requested"] = True

    def print_stats(reward, info):
        print(
            f"  step={info['step_count']:4d}  reward={reward:+.4f}  "
            f"base_height={info['base_height']:.4f}  "
            f"fall_tilt={info['fall_tilt']}  fall_height={info['fall_height']}"
        )

    with mujoco.viewer.launch_passive(
        env.sim.model, env.sim.data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            if state["reset_requested"]:
                obs = env.reset()
                print(f"\n--- Manual reset --- command={env.command}\n")
                state["reset_requested"] = False

            if not state["paused"]:
                action = action_fn(env.step_count)
                obs, reward, terminated, truncated, info = env.step(action)

                if info["step_count"] % 25 == 0:
                    print_stats(reward, info)

                if terminated or truncated:
                    reason = "FALL" if terminated else "TIME_OUT"
                    print(f"\n--- Episode ended: {reason} --- step={info['step_count']}")
                    obs = env.reset()
                    print(f"--- Reset --- command={env.command}\n")

            viewer.sync()


if __name__ == "__main__":
    main()
