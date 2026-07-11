"""Stage 2: biped_raw.xml → RL-ready MJCF variants (with/without Jetson)."""
import xml.etree.ElementTree as ET

RAW = "models/mjcf/biped_raw.xml"

KP = 40.0
KV = 10.0
FORCE = 2.5
# Recalibrated 2026-07-09 (Phase 0): the original 0.2061 value left both feet
# floating 1-3mm above the floor at the stand keyframe (0 contacts, touch
# sensors read 0), found by Phase 0's foot-contact integration test. Feet are
# not perfectly level (small L/R asymmetry in leg geometry), so an exact
# zero-gap shift for one foot leaves the other still floating; this value
# is calibrated via mesh AABB corners so BOTH feet have a small (<1mm)
# interpenetration margin, guaranteeing contact/touch-sensor readings at rest.
STAND_HEIGHT = 0.2030
# Joint names match the motor controller / joint encoder firmware naming
# (see ~/Work/biped python_st3215 package): hip_roll (hip motor), knee
# (formerly Onshape's "hip_pitch" mate), ankle (formerly Onshape's "knee"
# mate - the lowest actuated joint).
ACTUATED = ["hip_roll_l", "knee_l", "ankle_l",
            "hip_roll_r", "knee_r", "ankle_r"]
# FOOT_BODIES keys are body names (from Onshape mesh/part naming), a separate
# MuJoCo namespace from joint names above - not part of the joint rename.
FOOT_BODIES = {"foot": "r", "foot_1": "l"}
BODY_FRICTION = "1.0 0.02 0.001"

MASS_BY_MESH = {
    "Motor": 0.065,
    "Upper_Leg_A": 0.004, "Upper_Leg_B": 0.004,
    "Tibia": 0.005, "Tube": 0.005, "Foot": 0.009,
    "Hip_Base": 0.005, "Hip_Joint_A": 0.002, "Hip_Joint_B": 0.002,
    "Part_1": 0.005,
}
NOMINAL_INERTIA = {"Motor": "2e-5 2e-5 1.5e-5"}
TORSO_BODY = "composite_part_1__1_"

TORSO_BASE = 0.090 + 0.009
JETSON = 0.180
VARIANTS = [
    ("models/mjcf/biped.xml",            TORSO_BASE + JETSON, "6e-4 5e-4 4e-4"),
    ("models/mjcf/biped_no_jetson.xml",  TORSO_BASE,          "2.1e-4 1.8e-4 1.4e-4"),
]

# Collision proxy specs for Warp variant (mesh name → capsule/box/sphere params)
COLLISION_PROXIES = {
    "Motor": {"type": "capsule", "size": "0.015 0.05"},
    "Upper_Leg_A": {"type": "capsule", "size": "0.012 0.08"},
    "Upper_Leg_B": {"type": "capsule", "size": "0.012 0.08"},
    "Tibia": {"type": "capsule", "size": "0.008 0.10"},
    "Tube": {"type": "capsule", "size": "0.008 0.10"},
    "Hip_Base": {"type": "box", "size": "0.02 0.05 0.02"},
    "Hip_Joint_A": {"type": "sphere", "size": "0.01"},
    "Hip_Joint_B": {"type": "sphere", "size": "0.01"},
    "Part_1": {"type": "capsule", "size": "0.008 0.06"},
}


def build(OUT, torso_mass, torso_inertia):
    tree = ET.parse(RAW)
    mj = tree.getroot()

    compiler = mj.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", "../../meshes/stl/")

    def find_body(name):
        for b in mj.iter("body"):
            if b.get("name") == name:
                return b
        return None

    find_body("root").insert(0, ET.Element("freejoint", {"name": "floating_base"}))

    def set_mass(body, new_mass, inertia=None):
        ine = body.find("inertial")
        if ine is None:
            return
        m_old = float(ine.get("mass"))
        ine.set("mass", f"{new_mass:.6g}")
        if inertia is not None:
            ine.set("diaginertia", inertia)
        elif m_old > 1e-4:
            di = [float(x) * (new_mass / m_old) for x in ine.get("diaginertia").split()]
            ine.set("diaginertia", " ".join(f"{x:.6g}" for x in di))

    pelvis = find_body(TORSO_BODY)
    set_mass(pelvis, torso_mass, torso_inertia)
    for b in mj.iter("body"):
        if b.get("name") == TORSO_BODY:
            continue
        for mesh in [g.get("mesh") for g in b.findall("geom") if g.get("mesh")]:
            if mesh in MASS_BY_MESH:
                set_mass(b, MASS_BY_MESH[mesh], NOMINAL_INERTIA.get(mesh))
                break

    for body, side in FOOT_BODIES.items():
        b = find_body(body)
        vis = next(g for g in b.findall("geom") if g.get("mesh") == "Foot")
        b.append(ET.Element("geom", {
            "name": f"foot_col_{side}", "type": "mesh", "mesh": "Foot",
            "pos": vis.get("pos", "0 0 0"), "quat": vis.get("quat", "1 0 0 0"),
            "contype": "1", "conaffinity": "1",
            "friction": "1.0 0.02 0.001", "rgba": "0.1 0.5 0.9 0.4", "group": "3"}))
        b.append(ET.Element("site", {
            "name": f"foot_site_{side}", "pos": vis.get("pos", "0 0 0"),
            "size": "0.03", "rgba": "0 0 0 0"}))

    # Enable whole-body collision (see CLAUDE.md for design rationale)
    skip_bodies = set(FOOT_BODIES.keys())
    for b in mj.iter("body"):
        if b.get("name") in skip_bodies:
            continue
        for g in b.findall("geom"):
            if g.get("group") == "1" and g.get("type") == "mesh":
                g.set("contype", "1")
                g.set("conaffinity", "1")
                g.set("friction", BODY_FRICTION)

    pelvis.append(ET.Element("site", {"name": "imu", "pos": "0 0 0", "size": "0.005",
                                      "rgba": "1 0 0 0"}))
    opt = ET.SubElement(mj, "option")
    opt.set("timestep", "0.002"); opt.set("integrator", "implicitfast")
    vis = ET.SubElement(mj, "visual")
    ET.SubElement(vis, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3",
                                     "specular": "0 0 0"})
    ET.SubElement(vis, "global", {"offwidth": "1280", "offheight": "960"})
    dfl = ET.SubElement(mj, "default")
    ET.SubElement(dfl, "joint", {"armature": "0.01", "frictionloss": "0.002"})
    asset = mj.find("asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "512"})
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d", "builtin": "checker",
        "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.15 0.2", "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid",
        "texrepeat": "6 6", "reflectance": "0.1"})
    wb = mj.find("worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 2", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "size": "0 0 0.05",
        "material": "grid", "contype": "1", "conaffinity": "1",
        "friction": "1.0 0.02 0.001"})

    act = ET.SubElement(mj, "actuator")
    for j in ACTUATED:
        ET.SubElement(act, "position", {"name": f"act_{j}", "joint": j,
            "kp": f"{KP}", "kv": f"{KV}", "forcerange": f"-{FORCE} {FORCE}"})

    sen = ET.SubElement(mj, "sensor")
    for j in ACTUATED + ["foot_l", "foot_r"]:
        ET.SubElement(sen, "jointpos", {"name": f"pos_{j}", "joint": j})
        ET.SubElement(sen, "jointvel", {"name": f"vel_{j}", "joint": j})
    ET.SubElement(sen, "framequat", {"name": "torso_quat", "objtype": "site", "objname": "imu"})
    ET.SubElement(sen, "gyro", {"name": "torso_gyro", "site": "imu"})
    ET.SubElement(sen, "accelerometer", {"name": "torso_acc", "site": "imu"})
    ET.SubElement(sen, "framepos", {"name": "torso_pos", "objtype": "site", "objname": "imu"})
    ET.SubElement(sen, "framelinvel", {"name": "torso_linvel", "objtype": "site", "objname": "imu"})
    ET.SubElement(sen, "frameangvel", {"name": "torso_angvel", "objtype": "site", "objname": "imu"})
    for side in ("l", "r"):
        ET.SubElement(sen, "touch", {"name": f"touch_{side}", "site": f"foot_site_{side}"})

    key = ET.SubElement(mj, "keyframe")
    qpos = f"0 0 {STAND_HEIGHT} 1 0 0 0 " + " ".join(["0"] * 8)
    ET.SubElement(key, "key", {"name": "stand", "qpos": qpos,
                               "ctrl": " ".join(["0"] * len(ACTUATED))})

    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="utf-8", xml_declaration=False)
    print(f"wrote {OUT}  (torso {torso_mass:.3f} kg)")


def add_warp_variant(OUT, torso_mass, torso_inertia):
    """Build biped_warp.xml: implicitfast integrator + primitive collision geoms."""
    tree = ET.parse(RAW)
    mj = tree.getroot()

    compiler = mj.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", "../../meshes/stl/")

    def find_body(name):
        for b in mj.iter("body"):
            if b.get("name") == name:
                return b
        return None

    find_body("root").insert(0, ET.Element("freejoint", {"name": "floating_base"}))

    def set_mass(body, new_mass, inertia=None):
        ine = body.find("inertial")
        if ine is None:
            return
        m_old = float(ine.get("mass"))
        ine.set("mass", f"{new_mass:.6g}")
        if inertia is not None:
            ine.set("diaginertia", inertia)
        elif m_old > 1e-4:
            di = [float(x) * (new_mass / m_old) for x in ine.get("diaginertia").split()]
            ine.set("diaginertia", " ".join(f"{x:.6g}" for x in di))

    pelvis = find_body(TORSO_BODY)
    set_mass(pelvis, torso_mass, torso_inertia)
    for b in mj.iter("body"):
        if b.get("name") == TORSO_BODY:
            continue
        for mesh in [g.get("mesh") for g in b.findall("geom") if g.get("mesh")]:
            if mesh in MASS_BY_MESH:
                set_mass(b, MASS_BY_MESH[mesh], NOMINAL_INERTIA.get(mesh))
                break

    # Add foot collision geoms (same as CPU variant)
    for body, side in FOOT_BODIES.items():
        b = find_body(body)
        vis = next(g for g in b.findall("geom") if g.get("mesh") == "Foot")
        b.append(ET.Element("geom", {
            "name": f"foot_col_{side}", "type": "mesh", "mesh": "Foot",
            "pos": vis.get("pos", "0 0 0"), "quat": vis.get("quat", "1 0 0 0"),
            "contype": "1", "conaffinity": "1",
            "friction": "1.0 0.02 0.001", "rgba": "0.1 0.5 0.9 0.4", "group": "3"}))
        b.append(ET.Element("site", {
            "name": f"foot_site_{side}", "pos": vis.get("pos", "0 0 0"),
            "size": "0.03", "rgba": "0 0 0 0"}))

    # Add primitive collision geoms for non-foot bodies (Warp variant only)
    skip_bodies = set(FOOT_BODIES.keys())
    for b in mj.iter("body"):
        if b.get("name") in skip_bodies:
            continue
        # Find visual mesh geoms and add collision proxies
        for g in b.findall("geom"):
            if g.get("group") == "1" and g.get("type") == "mesh":
                mesh_name = g.get("mesh")
                # Keep the visual mesh as-is, don't add collision to it
                # (it remains contype=0, conaffinity=0)

                # Add a primitive collision geom if we have a proxy for this mesh
                if mesh_name in COLLISION_PROXIES:
                    proxy = COLLISION_PROXIES[mesh_name]
                    geom_dict = {
                        "type": proxy["type"],
                        "size": proxy["size"],
                        "pos": g.get("pos", "0 0 0"),
                        "quat": g.get("quat", "1 0 0 0"),
                        "contype": "1",
                        "conaffinity": "1",
                        "friction": BODY_FRICTION,
                        "rgba": "0.5 0.5 0.5 0.1",
                        "group": "2"
                    }
                    # For non-capsule/non-box types, adjust size format
                    b.append(ET.Element("geom", geom_dict))

    pelvis.append(ET.Element("site", {"name": "imu", "pos": "0 0 0", "size": "0.005",
                                      "rgba": "1 0 0 0"}))
    opt = ET.SubElement(mj, "option")
    opt.set("timestep", "0.002")
    # 2026-07-11: switched from "implicit" to "implicitfast". Verified
    # live (real mjlab 1.5.0 / mujoco-warp 3.10.0.1, installed and run on
    # CPU) that mjlab's own integrator map (mjlab/sim/sim.py
    # _INTEGRATOR_MAP) only recognizes "euler" and "implicitfast" -- there
    # is no "implicit" option at all, so the prior setting would have
    # raised KeyError('implicit') on Colab the first time an env was
    # constructed. "implicitfast" also matches the CPU biped.xml's
    # integrator, which is strictly better for sim-to-real/CPU-parity
    # than the originally planned "implicit" fallback would have been.
    opt.set("integrator", "implicitfast")
    vis = ET.SubElement(mj, "visual")
    ET.SubElement(vis, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3",
                                     "specular": "0 0 0"})
    ET.SubElement(vis, "global", {"offwidth": "1280", "offheight": "960"})
    dfl = ET.SubElement(mj, "default")
    ET.SubElement(dfl, "joint", {"armature": "0.01", "frictionloss": "0.002"})
    asset = mj.find("asset")
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient",
        "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "512"})
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d", "builtin": "checker",
        "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.15 0.2", "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid",
        "texrepeat": "6 6", "reflectance": "0.1"})
    wb = mj.find("worldbody")
    ET.SubElement(wb, "light", {"pos": "0 0 2", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "size": "0 0 0.05",
        "material": "grid", "contype": "1", "conaffinity": "1",
        "friction": "1.0 0.02 0.001"})

    act = ET.SubElement(mj, "actuator")
    for j in ACTUATED:
        ET.SubElement(act, "position", {"name": f"act_{j}", "joint": j,
            "kp": f"{KP}", "kv": f"{KV}", "forcerange": f"-{FORCE} {FORCE}"})

    sen = ET.SubElement(mj, "sensor")
    for j in ACTUATED + ["foot_l", "foot_r"]:
        ET.SubElement(sen, "jointpos", {"name": f"pos_{j}", "joint": j})
        ET.SubElement(sen, "jointvel", {"name": f"vel_{j}", "joint": j})
    ET.SubElement(sen, "framequat", {"name": "torso_quat", "objtype": "site", "objname": "imu"})
    ET.SubElement(sen, "gyro", {"name": "torso_gyro", "site": "imu"})
    ET.SubElement(sen, "accelerometer", {"name": "torso_acc", "site": "imu"})
    ET.SubElement(sen, "framepos", {"name": "torso_pos", "objtype": "site", "objname": "imu"})
    ET.SubElement(sen, "framelinvel", {"name": "torso_linvel", "objtype": "site", "objname": "imu"})
    ET.SubElement(sen, "frameangvel", {"name": "torso_angvel", "objtype": "site", "objname": "imu"})
    for side in ("l", "r"):
        ET.SubElement(sen, "touch", {"name": f"touch_{side}", "site": f"foot_site_{side}"})

    key = ET.SubElement(mj, "keyframe")
    qpos = f"0 0 {STAND_HEIGHT} 1 0 0 0 " + " ".join(["0"] * 8)
    ET.SubElement(key, "key", {"name": "stand", "qpos": qpos,
                               "ctrl": " ".join(["0"] * len(ACTUATED))})

    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="utf-8", xml_declaration=False)
    print(f"wrote {OUT}  (torso {torso_mass:.3f} kg, implicitfast integrator)")


for out, tmass, tinertia in VARIANTS:
    build(out, tmass, tinertia)

# Add Warp variant (implicitfast integrator + primitive collisions)
add_warp_variant("models/mjcf/biped_warp.xml", TORSO_BASE + JETSON, "6e-4 5e-4 4e-4")
