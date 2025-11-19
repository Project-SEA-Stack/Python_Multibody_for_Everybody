import numpy as np 
import networkx as nx 

AXIS_2D = (0, 2)                 # X-Z plane → indices 0 and 2
AXIS_ALL= {0, 1, 2}
AXIS_OUT= list(AXIS_ALL - set(AXIS_2D))[0]
NaN2    = [np.nan, np.nan]       # 2D NaN vector for prismatic joints
# --------------- small rotation helpers
def _rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], float)

def _rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], float)

def _rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], float)

def _quat_wxyz_to_R(q):
    q = np.asarray(q, float).ravel()
    if q.size != 4:
        raise ValueError("Quaternion must be [w, x, y, z].")
    w, x, y, z = q
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero-norm quaternion.")
    w, x, y, z = w/n, x/n, y/n, z/n
    xx, yy, zz = x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),   2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),       1 - 2*(xx + yy)]
    ], float)

def _euler_ypr_to_R(yaw, pitch, roll, degrees=False):
        if degrees:
            yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])
        return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)  # Z-Y-X

def _centroidal_transform(body_dict, angle_degrees=False):
    """
    Return (R_BC, r_BC, about_com) for a body:

    R_BC : 3x3 rotation mapping C→B (so v_B = R_BC @ v_C)
    r_BC : 3-vector from B-origin to C-origin, *expressed in B*
            If no 'com' block, we assume C coincides with B → r_BC = 0.
    about_com : bool, True if the YAML inertia is declared about the COM.
                If the body has a key 'inertia_about' == 'frame' (or 'reference'),
                we treat it as about the B-frame origin (not COM). Otherwise COM.

    Notes:
    - Euler angles are interpreted as yaw–pitch–roll (Z–Y–X), degrees if
        file-level 'angle_degrees' is True.
    - If 'com' is missing, we set R_BC = I and r_BC = 0.
    """
    ori = body_dict.get("orientation", None)

    # 1) Rotation C relative to B
    if ori is None:
        R_BC = np.eye(3)
    elif len(ori) == 3:
        yaw, pitch, roll = np.asarray(ori, float).ravel()
        R_BC = _euler_ypr_to_R(yaw, pitch, roll, degrees=angle_degrees)
    elif len(ori) == 4:
        R_BC = _quat_wxyz_to_R(np.asarray(ori, float).ravel())
    else:
        raise ValueError("Body 'orientation' must have 3 (Euler) or 4 (quaternion) elements.")

    return R_BC

def _pick_world_name(existing_names: set[str]) -> str:
        base = "__world_anchor__"
        name = base
        k = 1
        while name in existing_names:
            name = f"{base}{k}"
            k += 1
        return name
        
def _build_core_lists(joint_records: list[dict], name2idx: dict[str, int], cg: dict[str, np.ndarray], 
                      joint_map: dict[str, str],bodies_by_name: dict[str, dict]):
    joints              = []
    types               = []
    parent_cg_to_joint  = []
    joint_to_child_cg   = []
    prismatic_direction = []

    ax_idx = list(AXIS_2D)
    for j in joint_records:
        p, c = j["body1"], j["body2"]
        p_idx, c_idx = name2idx[p], name2idx[c]

        j_type = "F" if j["type"].lower().startswith("float") else joint_map[j["type"].lower()]
        if j_type == "P":
            axis3 = np.array(j.get("axis", NaN2 + [np.nan]), float)

            # If parent body has an orientation, rotate axis by the *negative* orientation:
            # i.e., apply R^T to express axis in the model frame consistent with M4E planar block.
            parent_dict = bodies_by_name.get(p)
            if parent_dict is not None and parent_dict.get("orientation") is not None:
                R_BC    = _centroidal_transform(parent_dict)         # 3x3
                axis3   = (R_BC.T @ axis3.reshape(3, 1)).ravel()     # rotate by -orientation

            axis2d = axis3[ax_idx]
            n = np.linalg.norm(axis2d)
            if n == 0:
                raise ValueError("Prismatic axis has zero length in X-Y.")
            axis2d = (axis2d / n).tolist()
        else:
            axis2d = NaN2

        joint_glob = np.array(j["location"], float)
        # transform joint location to parent body frame always rotating by -orientation from bodies location always 
        R_BC        = _centroidal_transform(bodies_by_name[p])      # 3x3
        origin      = bodies_by_name[p].get('location')
        joint_glob  = origin + (joint_glob - origin) @ R_BC     # rotate by -orientation

        p_cg    = cg[p]
        c_cg    = cg[c]
        p2j     = (joint_glob - p_cg)[ax_idx].tolist()
        j2c     = j2c = NaN2 if j_type == "F" else (c_cg - joint_glob)[ax_idx].tolist()

        joints.append([p_idx, c_idx])
        types.append(j_type)
        parent_cg_to_joint.append(p2j)
        joint_to_child_cg.append(j2c)
        prismatic_direction.append(axis2d)

    return joints, types, parent_cg_to_joint, joint_to_child_cg, prismatic_direction

def _handle_axis(j_type, axis3):
        if j_type == "P":
            axis2d = axis3[list(AXIS_2D)]
            norm = np.linalg.norm(axis2d)
            if norm == 0:
                raise ValueError("Prismatic axis has zero length in X-Y.")
            return (axis2d / norm).tolist()
        return NaN2

# --------- helpers (local-only) ----------
def _vec3(v, default=(0.0, 0.0, 0.0)):
    if v is None: return np.asarray(default, float)
    a = np.asarray(v, float).ravel()
    return a if a.size == 3 else np.asarray(default, float)

def _linvel_xy(bdict: dict) -> np.ndarray:
    return _vec3(bdict.get("initial_linear_velocity"))[list(AXIS_2D)]

def _omega_z(bdict: dict) -> float:
    return float(_vec3(bdict.get("initial_angular_velocity"))[AXIS_OUT])

def _in_plane_angle_R(R: np.ndarray) -> float:
    """
    Return the rotation angle about the out-of-plane axis, inferred from the
    2×2 sub-block of the rotation matrix spanning AXIS_2D, consistent with
    rot_2D_wave = [[c, s], [-s, c]].

    Works regardless of whether the out-of-plane axis is X, Y, or Z.
    """
    a, b = AXIS_2D
    # 2×2 “in-plane” block in the order (a,b)
    c = R[a, a]
    s = R[a, b]

    return float(np.arctan2(s, c))

def _tilde2D(v2: np.ndarray) -> np.ndarray:
    v2 = np.asarray(v2, float).ravel()
    return np.array([-v2[1], v2[0]], dtype=float)  # Z-axis skew

def _iter_joint_records_in_order(G: nx.DiGraph, name2idx: dict[str, int]):
    # iterate parents in the already-ordered name2idx key order
    for parent in name2idx:  # insertion-ordered
        for _, _, dat in G.out_edges(parent, data="data"):
            yield dat

def _build_di_graph_from_joints(joint_records):
    G = nx.DiGraph()
    if joint_records:
        add_edge = G.add_edge
        for j in joint_records:
            add_edge(j["body1"], j["body2"], data=j)
    return G

def _prepare_indices_and_cg(G: nx.DiGraph, world_name: str | None, bodies: list[dict]):
    body_order = list(nx.topological_sort(G))
    if world_name and (world_name in body_order):
        body_order = [world_name] + [b for b in body_order if b != world_name]

    name2idx = {name: i for i, name in enumerate(body_order)}
    idx2name = {i: n for n, i in name2idx.items()}

    # cg lookup (world frame), and quick body lookup
    cg = {}
    bodies_by_name = {}
    for b in bodies:
        name                    = b["name"]
        bodies_by_name[name]    = b
        B_world                 = np.array(b.get("location", [0, 0, 0]), float)
        com                     = b.get("com")
        com_loc_B               = np.array(((com.get("location") if isinstance(com, dict) else com) or [0, 0, 0]), float)
        # R_WB                    = _centroidal_transform(b)
        # cg[name]                = B_world + R_WB @ com_loc_B # We apply the orientation through IC
        cg[name]                = B_world + com_loc_B

    return name2idx, idx2name, cg, bodies_by_name

def _norm_xy(v):
    x, z = v[list(AXIS_2D)]
    n = (x*x + z*z) ** 0.5
    return (x / n, z / n) if n > 0 else (1.0, 0.0)  # default axis if zero

def _actuation_value(fn, t=None, sym=None):
    """Return a (possibly symbolic) magnitude f(t)."""

    if not fn:
        return sym.Float(0.0)
    ftype = str(fn.get("type", "CONSTANT")).upper()

    if ftype == "CONSTANT":
        return sym.Float(fn.get("value", 0.0))

    if ftype == "SINE":
        A   = sym.Float(fn.get("amplitude", 1.0)) # TODO: divide by 2?
        fHz = sym.Float(fn.get("frequency", 0.0))  # Hz
        phi = sym.Float(fn.get("phase", 0.0))      # rad
        w   = 2 * sym.pi * fHz
        return A * sym.sin(w * t + phi)

    if ftype in ("POLY", "POLYNOMIAL"):
        coeffs = fn.get("coefficients", [])
        return sum(sym.Float(c) * (t ** i) for i, c in enumerate(coeffs))

    if ftype == "RAMP":
        y0    = sym.Float(fn.get("y0", 0.0))
        slope = sym.Float(fn.get("slope", 0.0))
        t0    = sym.Float(fn.get("t0", 0.0))
        return y0 + slope * sym.Max(0, t - t0)

    # Fallback
    return sym.Float(0.0)
