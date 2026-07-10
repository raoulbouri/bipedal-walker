"""Stage 1: Onshape URDF → MuJoCo-loadable URDF → raw MJCF."""
import xml.etree.ElementTree as ET
import numpy as np, math, mujoco
from pathlib import Path

root_dir = Path(__file__).parent.parent
SRC = str(root_dir / "models/urdf/assembly_simple.urdf")
MJURDF = str(root_dir / "models/urdf/biped_for_mujoco.urdf")
RAW = str(root_dir / "models/mjcf/biped_raw.xml")
MESHDIR = str(root_dir / "meshes/stl")

FOOT_LO = math.radians(12 - 30)
FOOT_HI = math.radians(47 - 30)

# Rename Onshape mate-connector names to hardware-consistent names (matches
# the naming used in the motor controller / joint encoder firmware on the
# Jetson, see ~/Work/biped python_st3215 package). hip_roll_l/r already
# matches hardware (hip motor) - no rename needed there.
#   hip_pitch_l/r (Onshape name) -> knee_l/r (hardware name): the physical
#     joint Onshape's mate-connector naming called "hip_pitch" is actuated
#     as the knee on the real hardware.
#   knee_l/r (Onshape name) -> ankle_l/r (hardware name): the physical joint
#     Onshape called "knee" is the lowest actuated joint, named "ankle" in
#     the motor controller / joint encoder firmware.
ACTUATED_RENAME = {
    "hip_pitch_l": "knee_l", "hip_pitch_r": "knee_r",
    "knee_l": "ankle_l", "knee_r": "ankle_r",
}
# Passive hardstop joint (foot tilt, no actuator, hardware-measured hardstops
# at 12deg/47deg with a 30deg export offset). Renamed from "ankle" to "foot"
# to avoid colliding with the actuated ankle_l/r above - hardware firmware's
# "ankle" refers to the actuated joint above, not this passive one.
FOOT_HARDSTOP = {"revolute_1": "foot_r", "revolute_1_1": "foot_l"}

tree = ET.parse(SRC)
robot = tree.getroot()

# MuJoCo compiler directives
mj = ET.Element("mujoco")
ET.SubElement(mj, "compiler", {
    "meshdir": MESHDIR, "strippath": "true", "balanceinertia": "true",
    "discardvisual": "false", "fusestatic": "false", "angle": "radian",
})
robot.insert(0, mj)

for mesh in robot.iter("mesh"):
    mesh.set("filename", mesh.get("filename").replace(".gltf", ".stl"))
for j in robot.findall("joint"):
    name = j.get("name")
    if name in FOOT_HARDSTOP:
        j.set("name", FOOT_HARDSTOP[name])
        j.set("type", "revolute")
        for tag, attrs in (("limit", {"lower": f"{FOOT_LO:.6f}", "upper": f"{FOOT_HI:.6f}",
                                       "effort": "0", "velocity": "20"}),
                           ("dynamics", {"damping": "0.02", "friction": "0.003"})):
            e = j.find(tag) or ET.SubElement(j, tag)
            for k, v in attrs.items():
                e.set(k, v)
    else:
        if name in ACTUATED_RENAME:
            j.set("name", ACTUATED_RENAME[name])
        if j.get("type") == "revolute":
            # light damping on actuated joints for sim stability
            d = j.find("dynamics") or ET.SubElement(j, "dynamics")
            if d.get("damping") is None:
                d.set("damping", "0.05")
            lim = j.find("limit")
            if lim is not None:
                lim.set("effort", "2.5")

tree.write(MJURDF, encoding="utf-8", xml_declaration=True)
print("wrote", MJURDF)

# --- compile + save MJCF ---
m = mujoco.MjModel.from_xml_path(MJURDF)
mujoco.mj_saveLastXML(RAW, m)
print("wrote", RAW)

d = mujoco.MjData(m); mujoco.mj_forward(m, d)
print(f"\nbodies={m.nbody-1} joints={m.njnt} dofs={m.nv} total_mass={sum(m.body_mass):.4f} kg")
print("\nbody hierarchy (name : mass g : has-joint):")
for i in range(1, m.nbody):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    par = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[i]) or "world"
    hj = "joint" if m.body_jntnum[i] > 0 else "-"
    print(f"  {nm:28} m={m.body_mass[i]*1000:7.2f}g  parent={par:20} {hj}")
print("\njoints:")
for i in range(m.njnt):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    lo, hi = m.jnt_range[i]
    lim = f"[{math.degrees(lo):.0f}, {math.degrees(hi):.0f}] deg" if m.jnt_limited[i] else "unlimited"
    print(f"  {nm:16} {lim}")
