"""Stage 1: Onshape URDF → MuJoCo-loadable URDF → raw MJCF."""
import xml.etree.ElementTree as ET
import numpy as np, math, mujoco
from pathlib import Path

root_dir = Path(__file__).parent.parent
SRC = str(root_dir / "models/urdf/assembly_simple.urdf")
MJURDF = str(root_dir / "models/urdf/biped_for_mujoco.urdf")
RAW = str(root_dir / "models/mjcf/biped_raw.xml")
MESHDIR = str(root_dir / "meshes/stl")

ANKLE_LO = math.radians(12 - 30)
ANKLE_HI = math.radians(47 - 30)
ANKLE = {"revolute_1": "ankle_r", "revolute_1_1": "ankle_l"}

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
    if j.get("name") in ANKLE:
        j.set("name", ANKLE[j.get("name")])
        j.set("type", "revolute")
        for tag, attrs in (("limit", {"lower": f"{ANKLE_LO:.6f}", "upper": f"{ANKLE_HI:.6f}",
                                       "effort": "0", "velocity": "20"}),
                           ("dynamics", {"damping": "0.02", "friction": "0.003"})):
            e = j.find(tag) or ET.SubElement(j, tag)
            for k, v in attrs.items():
                e.set(k, v)
    elif j.get("type") == "revolute":
        # light damping on actuated joints for sim stability
        d = j.find("dynamics") or ET.SubElement(j, "dynamics")
        if d.get("damping") is None:
            d.set("damping", "0.05")

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
