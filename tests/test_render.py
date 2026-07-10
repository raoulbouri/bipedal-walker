"""Tests for rendering module."""

import sys
from pathlib import Path

import numpy as np
import pytest
import tempfile
import os
import shutil

# Add parent directory to path to import sim module
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.render import render_trajectory_frames, save_video
from sim import BipedSim


class TestRenderTrajectoryFrames:
    """Tests for render_trajectory_frames function."""

    def test_frame_shape_and_dtype(self, model_path_jetson):
        """Test that rendered frames have correct shape and dtype."""
        # Load model to get qpos dimension
        import mujoco
        model = mujoco.MjModel.from_xml_path(model_path_jetson)

        # Create a small trajectory: same "stand" qpos repeated 5 times
        # (we can vary one joint slightly, but the test is just checking shape)
        nq = model.nq
        qpos_stand = np.zeros(nq)

        # Use a simple approach: just repeat the same qpos
        qpos_trajectory = np.tile(qpos_stand, (5, 1))

        # Render
        frames = render_trajectory_frames(
            model_path_jetson, qpos_trajectory, width=320, height=240
        )

        # Check shape and dtype
        assert frames.shape == (5, 240, 320, 3), f"Expected (5, 240, 320, 3), got {frames.shape}"
        assert frames.dtype == np.uint8, f"Expected uint8, got {frames.dtype}"

    def test_frames_non_degenerate(self, model_path_jetson):
        """Test that rendered frames contain actual visual content (not blank)."""
        import mujoco
        model = mujoco.MjModel.from_xml_path(model_path_jetson)
        nq = model.nq

        # Render a single frame with default qpos
        qpos = np.zeros(nq)
        qpos_trajectory = np.tile(qpos, (1, 1))

        frames = render_trajectory_frames(
            model_path_jetson, qpos_trajectory, width=320, height=240
        )

        # Check that frames are not all the same solid color or all zero
        # by checking that there is some pixel variation (std > 0)
        frame_std = frames.std()
        assert frame_std > 0, "Rendered frame appears blank (no pixel variation)"

    def test_frames_change_with_different_poses(self, model_path_jetson):
        """Test that different qpos produce visibly different rendered frames."""
        # Use BipedSim to generate a real trajectory
        sim = BipedSim(model_path_jetson)
        sim.reset(keyframe="stand")

        # Collect qpos at intervals as the robot moves
        qpos_snapshots = []
        ctrl = np.zeros(sim.model.nu)

        # Step the simulation and collect qpos periodically
        for step in range(30):
            sim.step(ctrl)
            if step % 5 == 0:  # Collect every 5 steps
                qpos_snapshots.append(sim.qpos())

        # We should have ~6-7 distinct poses
        qpos_trajectory = np.array(qpos_snapshots)

        # Render the trajectory
        frames = render_trajectory_frames(
            model_path_jetson, qpos_trajectory, width=320, height=240
        )

        # Check that first and last frames are different
        # (robot should have moved/fallen/changed state visibly)
        first_frame = frames[0]
        last_frame = frames[-1]

        frames_identical = np.array_equal(first_frame, last_frame)
        assert not frames_identical, (
            "First and last frames are pixel-identical; "
            "renderer is not reflecting qpos changes"
        )


class TestSaveVideo:
    """Tests for save_video function."""

    def test_save_video_creates_file(self, model_path_jetson):
        """Test that save_video produces a real, non-empty .mp4 file."""
        # Check ffmpeg is available
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not available")

        # Render a small set of frames
        import mujoco
        model = mujoco.MjModel.from_xml_path(model_path_jetson)
        nq = model.nq

        qpos = np.zeros(nq)
        qpos_trajectory = np.tile(qpos, (5, 1))

        frames = render_trajectory_frames(
            model_path_jetson, qpos_trajectory, width=320, height=240
        )

        # Save to a temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_output.mp4")

            # Call save_video
            save_video(frames, out_path, fps=30)

            # Check that file exists and has nonzero size
            assert os.path.exists(out_path), f"Output file {out_path} was not created"
            file_size = os.path.getsize(out_path)
            assert file_size > 0, f"Output file {out_path} is empty (size={file_size})"
