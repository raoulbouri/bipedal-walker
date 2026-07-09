#!/usr/bin/env python3
"""Visualize biped in MuJoCo with interactive controls."""

import mujoco
import mujoco.viewer
import numpy as np
import os
from pathlib import Path

def main():
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    model = mujoco.MjModel.from_xml_path("models/mjcf/biped.xml")
    data = mujoco.MjData(model)
    data.qpos[:] = model.keyframe(0).qpos
    mujoco.mj_forward(model, data)

    print("=" * 60)
    print("Biped MuJoCo Viewer")
    print("=" * 60)
    print(f"\nModel: biped.xml")
    print(f"Bodies: {model.nbody - 1}  Joints: {model.njnt}  DOFs: {model.nv}")
    print(f"Total mass: {np.sum(model.body_mass):.3f} kg")
    print(f"\nStarting pose: Standing keyframe")
    print(f"  Base position (z): {data.qpos[2]:.3f} m")
    print(f"  Base orientation (quat): {data.qpos[3:7]}")
    print(f"  Actuated joint angles (rad): {data.qpos[7:]}\n")

    print("Controls:")
    print("  Mouse: Rotate view (left), Pan (right)")
    print("  Scroll: Zoom")
    print("  Space: Pause/Resume")
    print("  Close window to exit\n")

    paused = {"value": True}

    def key_callback(keycode):
        if 0 <= keycode < 256 and chr(keycode) == " ":
            paused["value"] = not paused["value"]
            print("Physics:", "RUNNING" if not paused["value"] else "PAUSED")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        print("Started PAUSED. Press SPACE to toggle dynamics.\n")

        while viewer.is_running():
            if not paused["value"]:
                mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
