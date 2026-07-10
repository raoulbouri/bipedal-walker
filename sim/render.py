"""Rendering utilities for biped simulation trajectories."""

import numpy as np
import mujoco
import subprocess
import os


def render_trajectory_frames(model_path, qpos_trajectory, width=320, height=240):
    """
    Render a sequence of qpos states to RGB frames using offscreen rendering.

    Args:
        model_path: path to the MJCF model.
        qpos_trajectory: array-like, shape (n_frames, nq) - one qpos vector per frame.
        width, height: frame dimensions.

    Returns:
        np.ndarray of shape (n_frames, height, width, 3), dtype uint8.
    """
    qpos_trajectory = np.asarray(qpos_trajectory)
    n_frames = qpos_trajectory.shape[0]

    # Load the model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Create renderer once (expensive to create per-frame)
    renderer = mujoco.Renderer(model, height, width)

    frames = []
    try:
        for i in range(n_frames):
            # Set qpos from trajectory
            data.qpos[:] = qpos_trajectory[i]

            # Run forward kinematics to compute derived quantities (xpos, xquat, etc.)
            # needed by the renderer
            mujoco.mj_forward(model, data)

            # Update renderer scene with new state
            renderer.update_scene(data)

            # Render and collect frame
            rgb_frame = renderer.render()
            frames.append(rgb_frame)
    finally:
        # Close renderer
        renderer.close()

    # Stack all frames into a single array
    frames_array = np.stack(frames, axis=0)
    return frames_array


def save_video(frames, out_path, fps=50):
    """
    Encode a frames array to an .mp4 file using ffmpeg via subprocess (no imageio).

    Args:
        frames: np.ndarray, shape (n_frames, height, width, 3), dtype uint8.
        out_path: output file path (should end in .mp4).
        fps: frames per second.

    Raises:
        RuntimeError: if ffmpeg fails.
    """
    frames = np.asarray(frames, dtype=np.uint8)
    n_frames, height, width, channels = frames.shape
    assert channels == 3, f"Expected 3-channel RGB, got {channels}"

    # Ensure parent directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Prepare ffmpeg command
    # Format: rgb24 (raw RGB without alpha), dimensions WxH, frame rate fps
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-pix_fmt", "yuv420p",  # Output format for compatibility
        "-vcodec", "libx264",
        out_path,
    ]

    # Run ffmpeg, piping raw frame bytes to stdin
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Write all frame data as raw bytes
    frame_bytes = frames.tobytes()
    stdout, stderr = process.communicate(input=frame_bytes)

    # Check return code
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with return code {process.returncode}\n"
            f"stderr: {stderr.decode()}"
        )
