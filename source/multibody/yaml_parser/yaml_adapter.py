# yaml_adapter.py
from threading import local
from types import SimpleNamespace
from pathlib import Path
import yaml
import numpy as np
import networkx as nx 
from ..checks.body_check import bodies
from ..checks.points_forces_check import points_forces
import sympy as sym

# helpers
from .helper import _quat_wxyz_to_R, _centroidal_transform, _pick_world_name, _build_core_lists, \
                               _vec3, _tilde2D, _in_plane_angle_R, _linvel_xy, _omega_z, \
                                 _build_di_graph_from_joints, _prepare_indices_and_cg, _iter_joint_records_in_order, \
                                 _norm_xy, _actuation_value, AXIS_2D, AXIS_OUT

from .simulation_parser import add_simulation_info

class YAML2Example:
    """Convert *.model.yaml* to the lists required by MBDSystem.from_example()."""

    _JOINT_MAP = {"revolute": "R", "prismatic": "P"}  # we add 'F' internally when needed

    # --------------------------------------------------------------------- init
    def __init__(self, yaml_path: str | Path):
        self.ForcesPointsSym    = []
        self.BodyDataSym        = []

        # Initialize default axis 
        self.default_axis_t             = np.zeros(3)
        self.default_axis_t[AXIS_2D[0]] = 1.0
        self.default_axis_r             = np.zeros(3)
        self.default_axis_r[AXIS_OUT]   = 1.0

        self.raw = yaml.safe_load(Path(yaml_path).read_text())['model']

        # bodies explicitly anchored to world
        self.fixed_names = {
            b["name"] for b in self.raw["bodies"] if bool(b.get("fixed", False))
        }

        # world anchor name (auto-created only if required)
        self.world_name: str | None = None

        # Bodies and joints definition
        self.ensure_world_and_floats()
        self.build_body_graph()            # → self.G   (DiGraph)
        self.assign_body_indices()         # → self.name2idx, idx2name, self.cg
        self.emit_joint_order()            # → self.joint_records
        self.build_lists()                 # → core lists (joints/types/vectors)
        self.read_mass_and_inertia()       # → m0, J0 (Iy in B frame)
        self.parse_ic()

        # points & forces from spring_dampers
        self.parse_spring_dampers()

        # Fixed body loads (forces/torques)
        self.parse_body_loads() 

        # Motor loads
        self.parse_motors()

        # checks
        bodies(self.joints, self.types,
                     self.parent_cg_to_joint, self.joint_to_child_cg, self.prismatic_direction)
        points_forces(self.joints, self.types, self.Initial_Points, self.Force)

        # check the problem is reduced to 2D
        self._check_is_2d()
    
    # -------------------------------------------------- small rotation helpers
    

    def ensure_world_and_floats(self):
        """Anchor system using any declared fixed body.
        If none exist, synthesize a world anchor at origin.
        """
        bodies = self.raw.get("bodies", []) or []
        bodies_by_name = {b["name"]: b for b in bodies}

        # Build joint graph for connectivity test
        Gtmp = nx.DiGraph()
        Gtmp.add_nodes_from(bodies_by_name.keys())
        for j in self.raw.get("joints", []) or []:
            Gtmp.add_edge(j["body1"], j["body2"])

        comps = list(nx.connected_components(Gtmp.to_undirected()))
        comps_without_anchor = [C for C in comps if not (set(C) & self.fixed_names)]

        if not comps_without_anchor:
            return  # all components already anchored by fixed bodies

        # Choose world anchor: reuse an existing fixed body if there is one
        if self.fixed_names:
            self.world_name = sorted(self.fixed_names)[0]
        else:
            # No fixed bodies declared → synthesize a world anchor at origin
            existing = set(bodies_by_name.keys())
            self.world_name = _pick_world_name(existing)
            world = {"name": self.world_name, "location": [0, 0, 0], "fixed": True}
            self.raw.setdefault("bodies", []).append(world)
            bodies_by_name[self.world_name] = world
            self.fixed_names.add(self.world_name)

        # For each unanchored component, attach its roots to the chosen world
        for C in comps_without_anchor:
            roots = [n for n in C if Gtmp.in_degree(n) == 0 and n not in self.fixed_names]
            if not roots:
                roots = [sorted([n for n in C if n not in self.fixed_names])[0]]

            for r in roots:
                float_joint = {
                    "name": f"F_{self.world_name}_to_{r}",
                    "type": "floating",       # will map to 'F'
                    "body1": self.world_name,
                    "body2": r,
                    "location": bodies_by_name.get(r, {}).get("location", [0, 0, 0]),
                }
                self.raw.setdefault("joints", []).append(float_joint)
                self.raw["joints"].insert(0, float_joint)

    # ----------------------------------------- 2) build full body/joint graph
    def build_body_graph(self):
        self.G = _build_di_graph_from_joints(self.raw.get("joints", []))

    # ----------------------------------------- 3) numbering: parent < child
    def assign_body_indices(self):
        (self.name2idx,
         self.idx2name,
         self.cg,
         self._bodies_by_name) = _prepare_indices_and_cg(self.G, self.world_name, self.raw["bodies"])

    # ----------------------------------------- 4) joints in parent-child order
    def emit_joint_order(self):
        self.joint_records = list(_iter_joint_records_in_order(self.G, self.name2idx))

    # ---------------------------------------------------- 5) core kinematic lists
    def build_lists(self):
        (self.joints,
         self.types,
         self.parent_cg_to_joint,
         self.joint_to_child_cg,
         self.prismatic_direction) = _build_core_lists(
            self.joint_records, self.name2idx, self.cg, self._JOINT_MAP,
            self._bodies_by_name,
        )

    # ---------------------------------------------------- 6) mass & inertia (Iy)
    def _inertia_tensor_from_yaml(self, inertia_field):
        """
        Build 3x3 inertia tensor (about COM, expressed in the COM frame C)
        from common Chrono YAML forms:
          - dict with keys 'moments' (Ixx, Iyy, Izz) and optional 'products' (Ixy, Ixz, Iyz)
          - dict with scalar keys: ixx, iyy, izz, ixy, ixz, iyz
          - list/tuple of length 3 → principal moments only
        """
        if inertia_field is None:
            raise ValueError("Missing 'inertia' for a body.")

        Ixx = Iyy = Izz = None
        Ixy = Ixz = Iyz = 0.0  # default zero products

        if isinstance(inertia_field, dict):
            Ixx, Iyy, Izz = map(float, inertia_field["moments"])

            if "products" in inertia_field:
                Ixy, Ixz, Iyz = map(float, inertia_field["products"])

        # Standard inertia matrix sign convention: off-diagonals are -Ixy, -Ixz, -Iyz
        I = np.array([[ Ixx, -Ixy, -Ixz],
                      [-Ixy,  Iyy, -Iyz],
                      [-Ixz, -Iyz,  Izz]], float)
        return I

    def read_mass_and_inertia(self):
        """
        Build arrays m0 and J0 (I_y in the B-frame) aligned with the child (body2)
        order of self.joint_records.

        Steps per body:
        (a) Read 'mass'
        (b) Build I_C from YAML (about COM unless declared otherwise)
        (c) Get (R_BC, r_BC, about_com) from _centroidal_transform
        (d) If YAML inertia is about the B-frame (not COM): convert it to COM via
            parallel-axis inverse: I_B@COM = I_B@B - m * (||r||^2 I - r r^T),
            where r = r_BC (B→C in B).
            If it is already about COM: keep as-is.
        (e) Ensure inertia is expressed in the B-frame at COM: I_B@COM = R_BC I_C R_BC^T
        (f) Extract J_y = (I_B@COM)[1,1] for 2D (out-of-plane about Y).
        """
        m_list  = []
        Iy_list = []

        for j in self.joint_records:
            child = j["body2"]
            b = self._bodies_by_name.get(child)
            if b is None:
                raise KeyError(f"Body '{child}' not found in 'bodies' section.")

            # (a) mass
            if "mass" not in b:
                raise ValueError(f"Body '{child}' is missing 'mass'.")
            m = float(b["mass"])
            m_list.append(m)

            # (b) inertia as provided (in C by Chrono convention unless overridden)
            I_from_yaml = self._inertia_tensor_from_yaml(b.get("inertia"))

            # (c) centroidal transform
            R_BC = _centroidal_transform(b)

            # (d) if YAML inertia is about the B-origin, move it to COM in B
            I_at_COM = R_BC @ I_from_yaml @ R_BC.T

            # (e) (already expressed in B at COM)
            Iy_list.append(float(I_at_COM[1, 1]))

        self.m0 = np.array(m_list, float)
        self.J0 = np.array(Iy_list, float)


    # ---------------------------------------------------- 7) spring-dampers → points & forces
    def parse_spring_dampers(self):
        """Populate Initial_Points + Force from the YAML spring_dampers block."""
        self.Initial_Points = {"GR": [], "BD": {}}
        self.Force = {k: [] for k in ("PointsBD", "CG",
                                      "TensionSpring", "TensionDamper",
                                      "TorsionSpring", "TorsionDamper")}

        for sd in self.raw.get("rsdas", []):
            # —————————————————— A) rotational  (RSDA)
            b1 = self.name2idx[sd["body1"]]
            b2 = self.name2idx[sd["body2"]]
            k  = float(sd.get("spring_coefficient", 0))
            c  = float(sd.get("damping_coefficient", 0))
            if k:
                # Zero angle as initial condition
                self.Force["TorsionSpring"].append(((b1, b2), [0, k]))
            if c:
                self.Force["TorsionDamper"].append(((b1, b2), c))

            # —————————————————— B) tension  (TSDA)
        for sd in self.raw.get("tsdas", []):
            # 1. register / label the two connection points
            lbl1 = self._register_point(sd["body1"], sd["point1"], is_local=False)
            lbl2 = self._register_point(sd["body2"], sd["point2"], is_local=False)

            k  = float(sd.get("spring_coefficient", 0))
            c  = float(sd.get("damping_coefficient", 0))
            l0 = float(sd.get("free_length", 0))

            if k:
                self.Force["TensionSpring"].append(((lbl1, lbl2), [l0, k]))
            if c:
                self.Force["TensionDamper"].append(((lbl1, lbl2), c))

    # ------------ helper to create / reuse a point & return its label --------
    def _register_point(self, body_name: str, xyz, *, is_local: bool = True, tol: float = 1e-12):
        """
        Add a point to Initial_Points without duplicates.

        Storage invariant:
        - GR: absolute (world) coordinates [x,z] for world or fixed bodies
        - BD: relative-to-body-CG coordinates [x,z] for non-fixed bodies

        Parameters
        ----------
        body_name : str
            Name of the body the point is associated with.
        xyz : array-like
            Point coordinates in either local (body) or global (world) frame.
        coord_frame : {"local","global"}, default "local"
            Frame in which `xyz` is provided.
        tol : float
            Comparison tolerance for deduping.

        Returns
        -------
        str
            Label like "GR0i" or "BDbi".
        """
        x, z = np.asarray(xyz, float)[list(AXIS_2D)]
        x = float(x); z = float(z)

        is_world = (self.world_name is not None) and (body_name == self.world_name)
        is_fixed = (body_name in self.fixed_names)

        # ---------- Grounded (GR) branch: store ABSOLUTE ----------
        if is_world or is_fixed:
            coord_abs = [x, z]

            # De-duplicate in GR
            gr_list = self.Initial_Points.get("GR", [])
            for i, (gx, gz) in enumerate(gr_list):
                if abs(gx - coord_abs[0]) <= tol and abs(gz - coord_abs[1]) <= tol:
                    return f"GR0{i}"

            idx = len(gr_list)
            gr_list.append(coord_abs)
            self.Initial_Points["GR"] = gr_list
            return f"GR0{idx}"

        # ---------- Body-defined (BD) branch: store RELATIVE ----------
        body_idx = self.name2idx[body_name]
        if body_idx not in self.Initial_Points["BD"]:
            self.Initial_Points["BD"][body_idx] = []

        if is_local:
            coord_rel = [x, z]
        else:  # frame == "global" -> convert to local (relative to CG)
            cg = self.cg.get(body_name)
            if cg is None:
                raise KeyError(f"Missing CG for body '{body_name}' to convert global->local.")
            cg2 = np.asarray(cg, float)[list(AXIS_2D)]
            coord_rel = [x - float(cg2[0]), z - float(cg2[1])]

        # De-duplicate within BD[body_idx]
        bd_list = self.Initial_Points["BD"][body_idx]
        for i, (bx, bz) in enumerate(bd_list):
            if abs(bx - coord_rel[0]) <= tol and abs(bz - coord_rel[1]) <= tol:
                return f"BD{body_idx}{i}"

        idx = len(bd_list)
        bd_list.append(coord_rel)
        return f"BD{body_idx}{idx}"

    
    def parse_ic(self):
        """
        Build initial conditions vector self.ic = [Q ; QD] following JointSystem order.
        XY plane; rotation about Z.
        """
        def _cg_xy(body_name: str) -> np.ndarray:
            # World CG from self.cg, XY slice
            return np.asarray(self.cg[body_name], float).ravel()[list(AXIS_2D)]
        
        # --------- quick lookups ----------
        bodies_by_name = self._bodies_by_name  # set in _assign_body_indices
        rec_by_child   = { self.name2idx[rec["body2"]] : rec for rec in self.joint_records }

        # Children are 1..NB in your lists
        NB = max(c for _, c in self.joints) if self.joints else 0
        Q_vals, QD_vals = [], []

        for child in range(1, NB+1):
            # joint list is already in child order, but we’ll be explicit
            j_idx  = next(i for i, (_, c) in enumerate(self.joints) if c == child)
            j_type = self.types[j_idx]
            parent = self.joints[j_idx][0]

            child_name = self.idx2name[child]
            child_b    = bodies_by_name[child_name]

            # Parent kinematics (ground if parent==0)
            if parent == 0:
                rP = np.array([0.0, 0.0]); vP = np.array([0.0, 0.0]); wP = 0.0
            else:
                parent_name = self.idx2name[parent]
                parent_b    = bodies_by_name[parent_name]
                rP = _cg_xy(parent_name)
                vP = _linvel_xy(parent_b)
                wP = _omega_z(parent_b)

            # Child CG kinematics
            rC = _cg_xy(child_name)
            vC = _linvel_xy(child_b)

            if j_type == 'F':
                theta0      = 0.0
                if child_b.get("orientation") is not None:
                    R_B = _centroidal_transform(child_b)                  # uses body["orientation"]
                    theta0 = _in_plane_angle_R(np.asarray(R_B, float))

                omega = _omega_z(child_b)

                # X, Y, Theta ; XD, YD, ThetaD
                Q_vals.extend([float(rC[0]), float(rC[1]), float(theta0)])
                QD_vals.extend([float(vC[0]), float(vC[1]), -float(omega)])  # keep same sign convention as 'R'

            elif j_type == 'R':
                # theta0 from COM.orientation (if present) via _centroidal_transform
                theta0      = 0.0
                if child_b.get("orientation") is not None:
                    R_B = _centroidal_transform(child_b)
                    theta0 = _in_plane_angle_R(np.asarray(R_B, float))

                omega = _omega_z(child_b)

                Q_vals.append(float(theta0))
                QD_vals.append(float(omega))

            elif j_type == 'P':
                # Project along joint axis in XY using joint_records
                rec             = rec_by_child[child]
                joint_loc_xy    = _vec3(rec.get("location"))[list(AXIS_2D)]
                axis3           = _vec3(rec.get("axis"), default=(np.nan, np.nan, np.nan))
                Uxy             = axis3[list(AXIS_2D)]

                if not np.isfinite(Uxy).all() or np.linalg.norm(Uxy) == 0.0:
                    s = 0.0; sdot = 0.0
                else:
                    U   = Uxy / np.linalg.norm(Uxy)
                    p2j = joint_loc_xy - rP        # parentCG → joint (XY)
                    j2c = rC - joint_loc_xy        # joint → childCG (XY)

                    # With angles = 0 at t0, geometry consistent with your lists:
                    s    = float((rC - rP - p2j - j2c).dot(U))
                    sdot = float((vC - vP - wP * _tilde2D(p2j + j2c)).dot(U))

                Q_vals.append(s)
                QD_vals.append(sdot)

            else:
                raise ValueError(f"Unsupported joint type '{j_type}' while parsing ICs.")

        self.ic = np.array(Q_vals + QD_vals, dtype=float)

    def parse_body_loads(self):
        """
        Parse Chrono YAML body_loads (from self.raw.body_loads) and populate self.Force.

        Populates:
          - self.Force["PointsBD"]: dicts with {"body", "localID", "vector":[Fx,Fz]}
          - self.Force["CG"]:       dicts with {"body", "Mz"}  (includes CG torques and r×F from forces)
        """

        body_loads = self.raw.get("body_loads") or []
        if not body_loads:
            return   # nothing to do, just exit

        for e in body_loads:
            typ = e["type"].upper()
            body_idx = self.name2idx[e["body"]]

            if e.get("local_load", True):
                raise NotImplementedError("local_load: true not implemented in M4E yet.")

            # Load vector
            Lx, Lz = _vec3(e.get("load", [0.0, 0.0, 0.0]))[list(AXIS_2D)]

            if typ in ("TORQUE", "MOMENT"):
                # 2D torque: keep only z-component at CG
                self.Force["CG"].append([body_idx, 0., 0., float(Lz)])

                continue

            if typ == "FORCE":
                # Force (2D): use XY components
                Fx, Fz = float(Lx), float(Lz)

                # Application point (BD). Convert to local if given in global.
                pos_vec = _vec3(e.get("point", [0.0, 0.0, 0.0]))

                # Register BD point (returns 0-based localID)
                pointName = self._register_point(body_name=e["body"], xyz=pos_vec, is_local=e.get("local_point", True))
                # Check the total number of bodies is < 10 for this to work
                if len(self.idx2name) < 10:
                    localID = int(pointName[3:])  #  What if the number of bodies > 10? 
                elif len(self.idx2name) < 100:
                    localID = int(pointName[4:])
                else:
                    raise ValueError("Number of bodies exceeds 99 or negative, cannot parse localID from pointName.")

                # Store force at BD point
                self.Force["PointsBD"].append([body_idx, localID, Fx,Fz,0])

                continue

            # Ignore other types to keep it minimal

    def parse_motors(self):
        motors = self.raw.get("motors", None)
        if not motors:
            return  # nothing to do

        # time symbol for symbolic actuation
        t = sym.symbols('t', real=True)
        self.ForcesPointsSym.append(t)

        for m in motors:
            a_type = str(m.get("actuation_type", "")).upper()
            if a_type != "FORCE":
                raise ValueError("type not supported")

            mtype = str(m.get("type", "")).upper()

            # Bodies and placement
            b1      = self.name2idx[m["body1"]]
            b2      = self.name2idx[m["body2"]]
            loc_w   = _vec3(m.get("location", [0.0, 0.0, 0.0]))
            axis    = _vec3(m.get("axis", self.default_axis_t))  # relevant for linear & rotation

            spindle = m.get("spindle")  # relevant for rotation
            guide   = m.get("guide")    # relevant for linear

            if spindle is not None and spindle != 'FREE':
                raise NotImplementedError("Motor spindle constraints not implemented in M4E yet.")
            if guide is not None and guide != 'FREE':
                raise NotImplementedError("Motor guide constraints not implemented in M4E yet.")

            # Force/torque magnitude function (sympy expr or numeric)
            mag     = _actuation_value(m.get("actuation_function"), t=t, sym=sym)

            if mtype == "LINEAR":
                # project onto XY along the motor axis
                ux, uz  = _norm_xy(axis)
                Fx      = mag * sym.Float(ux)
                Fz      = mag * sym.Float(uz)

                # register BD points if body is not fixed/world
                if m["body1"] not in self.fixed_names:
                    pointName = self._register_point(body_name=m["body1"], xyz=loc_w, is_local=False)

                    if len(self.idx2name) < 10:
                        localID = int(pointName[3:])  #  What if the number of bodies > 10? 
                    elif len(self.idx2name) < 100:
                        localID = int(pointName[4:])
                    else:
                        raise ValueError("Number of bodies exceeds 99 or negative, cannot parse localID from pointName.")

                    self.Force["PointsBD"].append([b1, localID,  Fx,  Fz, sym.Integer(0)])

                if m["body2"] not in self.fixed_names:
                    pointName = self._register_point(body_name=m["body2"], xyz=loc_w, is_local=False)

                    if len(self.idx2name) < 10:
                        localID = int(pointName[3:])  #  What if the number of bodies > 10? 
                    elif len(self.idx2name) < 100:
                        localID = int(pointName[4:])
                    else:
                        raise ValueError("Number of bodies exceeds 99 or negative, cannot parse localID from pointName.")

                    self.Force["PointsBD"].append([b2, localID, -Fx, -Fz, sym.Integer(0)])


            elif mtype in ("ROTATION", "ROTATIONAL"):
                # project torque magnitude onto z via spindle/axis (2D keeps only z)
                axz         = axis[AXIS_OUT]
                My          = mag * sym.Float(axz)

                # equal & opposite CG torques
                if m["body1"] not in self.fixed_names:
                    self.Force["CG"].append([b1, 0, 0, My])

                if m["body2"] not in self.fixed_names:
                    self.Force["CG"].append([b2, 0, 0, -My])

            else:
                # Unknown motor type under FORCE actuation
                raise ValueError("type not supported")

    def _check_is_2d(self):
        """Ensure the problem is purely 2D: all CG Y=0 and rotations only about Y-axis."""
        # 1. CG positions must have Y=0
        for name, loc in self.cg.items():
            if not np.isclose(loc[AXIS_OUT], 0.0, atol=1e-12):
                raise ValueError(
                    f"Body '{name}' has nonzero Y-coordinate in CG: {loc}"
                )

        # Helper for axis checking
        def _is_axis_z(axis):
            #convert to euler angles if len of axis is 4
            if len(axis) == 4:
                R       = _quat_wxyz_to_R(axis)
                axis    = R[:,AXIS_OUT]  # Z-axis of the rotated frame

            axis = np.array(axis, float)
            if np.allclose(axis, 0.0):
                return True  # ignore zero vector cases
            
            axis                /= np.linalg.norm(axis)
            return (
                np.allclose(axis, self.default_axis_r, atol=1e-12) or
                np.allclose(axis, -self.default_axis_r, atol=1e-12)
            )

        # 2. Check body orientations (if defined)
        for b in self.raw.get("bodies", []):
            if "orientation" in b and b["orientation"] is not None:
                if not _is_axis_z(b["orientation"]):
                    raise ValueError(
                        f"Body '{b['name']}' orientation {b['orientation']} "
                        f"is not aligned with ±Z-axis."
                    )

        # 3. Check joint axes for revolute joints
        for j in self.raw.get("joints", []):
            if j["type"].lower().startswith("rev"):
                if not _is_axis_z(j.get("axis", self.default_axis_r)):
                    raise ValueError(
                        f"Revolute joint '{j['name']}' axis {j.get('axis')} "
                        f"is not aligned with ±Y-axis."
                    )

    # --------------------------------------------------- 8) public façade
    def as_example_module(self,animation_on, SaveMovieOn, plotTstep) -> SimpleNamespace:
        """Return a namespace ready for `MBDSystem.from_example()`."""
        return SimpleNamespace(
            joints                 = self.joints,
            types                  = self.types,
            parent_cg_to_joint     = self.parent_cg_to_joint,
            joint_to_child_cg      = self.joint_to_child_cg,
            prismatic_direction    = self.prismatic_direction,
            # minimal stubs so the class can be instantiated
            Reference_frame_Origin = [0, 0],
            Initial_Points         = self.Initial_Points,
            Force                  = self.Force,
            ForcesPointsSym        = self.ForcesPointsSym,
            BodyDataSym            = self.BodyDataSym,
            ForcesPointsNum        = np.ones(len(self.ForcesPointsSym)),
            BodyDataNum            = np.ones(len(self.BodyDataSym)),
            ic                     = self.ic,
            # NEW: physical properties in child/joint order
            m0                     = self.m0,
            J0                     = self.J0,
            # Animation features
            animation_on           = animation_on,
            SaveMovieOn            = SaveMovieOn,
            plotTstep              = plotTstep
        )


# ---------------------------------------------------------------- convenience
def load_yaml_as_example(path: str | Path,*,
                        animation_on: bool = True,
                        SaveMovieOn: str | None = None,
                        plotTstep: int = 10) -> SimpleNamespace:
    """One-liner wrapper:  ex = load_yaml_as_example("myfile.yaml")"""
    # Load model properties
    path_model  = path + ".model.yaml"
    ex          =  YAML2Example(path_model).as_example_module(animation_on, SaveMovieOn, plotTstep)

    # Load simulation properties
    path_sym    = path + ".simulation.yaml"

    # Append simulation info to the example
    add_simulation_info(path_sym, ex)

    return ex
