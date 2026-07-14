"""Phase 7.R.1: action clipping in mjlab_biped/mjlab_task.py.

mjlab_task.py cannot be imported on the Mac (no mjlab install -- see
docs/mjlab_adapter_notes.md), so ACTION_CLIP is extracted via AST/text,
never `import`ed, matching the pattern in test_mjlab_task_static.py.
Cross-checked against the real, compiled model's jnt_range (plain
`mujoco`, available in the lean Mac env) so this test fails if the
pipeline ever changes a joint's limits without updating the clip.
"""
import ast
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parent.parent
MJLAB_TASK_PATH = REPO_ROOT / "mjlab_biped" / "mjlab_task.py"

ACTUATED_JOINTS = ("hip_roll_l", "hip_roll_r", "knee_l", "knee_r", "ankle_l", "ankle_r")


def _extract_action_clip() -> dict:
    """Parse ACTION_CLIP = {...} out of mjlab_task.py via AST, without
    importing the module (it imports the real mjlab package)."""
    tree = ast.parse(MJLAB_TASK_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ACTION_CLIP" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("ACTION_CLIP assignment not found in mjlab_task.py")


def _real_jnt_range(joint_name: str, model_path: str) -> tuple:
    model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / model_path))
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    assert jid >= 0, f"joint '{joint_name}' not found in {model_path}"
    lo, hi = model.jnt_range[jid]
    return float(lo), float(hi)


def test_action_clip_present_for_all_actuated_joints():
    clip = _extract_action_clip()
    for joint in ACTUATED_JOINTS:
        assert joint in clip, f"ACTION_CLIP is missing an entry for '{joint}'"


def test_action_clip_matches_real_joint_ranges_biped_xml():
    """The clip bounds must exactly match biped.xml's own jnt_range --
    not a hand-copied approximation that can silently drift."""
    clip = _extract_action_clip()
    for joint in ACTUATED_JOINTS:
        expected = _real_jnt_range(joint, "models/mjcf/biped.xml")
        actual = tuple(clip[joint])
        assert actual == expected, (
            f"ACTION_CLIP['{joint}'] = {actual} does not match biped.xml's "
            f"real jnt_range {expected}"
        )


def test_action_clip_matches_real_joint_ranges_biped_warp_xml():
    """biped_warp.xml must share the same joint limits as biped.xml (both
    come from the same pipeline/URDF) -- verifies the clip is valid for
    the model actually used in training, not just the CPU eval model."""
    clip = _extract_action_clip()
    for joint in ACTUATED_JOINTS:
        expected = _real_jnt_range(joint, "models/mjcf/biped_warp.xml")
        actual = tuple(clip[joint])
        assert actual == expected, (
            f"ACTION_CLIP['{joint}'] = {actual} does not match "
            f"biped_warp.xml's real jnt_range {expected}"
        )


def test_action_cfg_wires_clip():
    """ACTION_CFG must actually pass clip=ACTION_CLIP to
    JointPositionActionCfg -- defining ACTION_CLIP without wiring it in
    would silently do nothing."""
    source = MJLAB_TASK_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "ACTION_CFG" for t in node.targets)
            and isinstance(node.value, ast.Call)
        ):
            kwarg_names = {kw.arg for kw in node.value.keywords}
            assert "clip" in kwarg_names, "ACTION_CFG does not pass a clip= argument"
            for kw in node.value.keywords:
                if kw.arg == "clip":
                    assert isinstance(kw.value, ast.Name) and kw.value.id == "ACTION_CLIP", (
                        "ACTION_CFG's clip= argument is not ACTION_CLIP"
                    )
            return
    raise AssertionError("ACTION_CFG assignment not found in mjlab_task.py")
