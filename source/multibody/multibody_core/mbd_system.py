"""
mbd_system.py

High-level multibody dynamics system definition and simulation interface.

This module provides the MBDSystem class, which encapsulates:

1. Symbolic construction of kinematics, forces, and energy expressions based on
   user-supplied joint, body, point, and force definitions or existing example modules.
2. Automatic lambdification of symbolic expressions into efficient NumPy-powered
   functions for position transforms, velocities, accelerations, mass matrix,
   right-hand side, and total energy.
3. A streamlined numerical integration workflow using SciPy's `solve_ivp`, with
   support for custom external force managers.

Public API:

- MBDSystem: Create a multibody system from raw data or an example, then symbolically
  build, lambdify, and integrate the equations of motion.
"""

from __future__ import annotations
import numpy as np
import sympy as sym
from types import ModuleType
from pathlib import Path
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
from time import time

# --- reuse the building blocks already in the package -------------
from .joints_system          import JointSystem, normalize_prismatic
from .kinematics_transformation import VelocityTransformation
from .points_forces_definition import points_force_finder
from .systems_energy          import systems_energy
from .rigid_body_integrator    import integrate_dynamics   # numeric RHS

# ------------------------------------------------------------------
@dataclass
class MbdSystem:
    """
    High-level multibody system combining symbolic model generation and numerical simulation.

    Constructs the full symbolic model (kinematics, constraints, forces, energies),
    compiles these to numeric functions, and provides an integration routine.

    Args:

        joints (list[tuple[int, int]]): Parent-child connectivity for each joint.
        types (list[str]): Joint types ('R', 'P', 'F', etc.).
        parent_cg_to_joint (list): Vectors from parent body CG to joint locations.
        joint_to_child_cg (list): Vectors from joint to child body CGs.
        prismatic_direction (list): Unit direction vectors for prismatic joints.
        Initial_Points (dict): Definitions of grounded and body-specific points.
        Force (dict): Force definitions (CG forces, point forces, springs, dampers).
        Reference_frame_Origin (tuple|list): Global origin coordinates.
        ForcesPointsSym (list[sym.Symbol]): Symbolic variables for force-point magnitudes.
        BodyDataSym (list[sym.Symbol]): Symbolic variables for body data (mass, inertia).
        ic (np.ndarray): Initial conditions array for generalized coordinates and speeds.
        g (float|sym.Symbol): Gravitational acceleration constant (default 9.81).
        gVec (np.ndarray): Per-body gravity scale factors (default ones).

    """
    # Raw input data, typically from an Examples module
    joints:                list[tuple[int, int]]
    types:                 list[str]
    parent_cg_to_joint:    list
    joint_to_child_cg:     list
    prismatic_direction:   list
    Initial_Points:        dict = field(default_factory=dict)
    Force:                 dict = field(default_factory=dict)
    Reference_frame_Origin: list | tuple | np.ndarray = field(default_factory=lambda: np.array([0, 0]))

    # Optional symbolic lists
    ForcesPointsSym:       list[sym.Symbol] = field(default_factory=list)
    BodyDataSym:           list[sym.Symbol] = field(default_factory=list)

    # Physical constants
    g: float | sym.Symbol   = 9.81
    gVec: np.ndarray        = None
    ic: np.ndarray  | None  = None  # default zero initial conditions

    # Internal symbolic containers (populated in __post_init__)
    joint_system: JointSystem       = field(init=False)
    Q:   list                       = field(init=False)
    QD:  list                       = field(init=False)
    QDD: list                       = field(init=False)
    NDOF:int                        = field(init=False)
    Body_col:list                   = field(init=False)
    
    Pos = JointLoc = R = RD = Vel = Acc = None
    paths = graph = None
    Points_All = Force_All = None
    Energy = None
    M = None
    m = None
    J = None
    
    # lambdified numeric wrappers
    Pos_func = JointLoc_func = R_func = RD_func = Vel_func = None
    Acc_func = Right_side_func = ReducedM_func = Energy_func = None
    BDpoints_func: dict = field(init=False)
    CGpoints_func = None
    GRpoints_func = None
    BDvel_func: dict = field(init=False)
    CGvel_func = None
    
    t_update: list[int] = field(init=False, default_factory=list)

    # ------------- alternate constructors -------------------------
    @classmethod
    def from_example(cls, ex: ModuleType) -> "MbdSystem":
        """
        Create an MbdSystem from a user-provided example module.

        The example module must define: joints, types, parent_cg_to_joint,
        joint_to_child_cg, prismatic_direction, Initial_Points, Force,
        Reference_frame_Origin, ForcesPointsSym, BodyDataSym, ic, g (optional),
        and gVec (optional).
        """

        return cls(
             joints                 = ex.joints,
             types                  = ex.types,
             parent_cg_to_joint     = ex.parent_cg_to_joint,
             joint_to_child_cg      = ex.joint_to_child_cg,
             prismatic_direction    = ex.prismatic_direction,
             Initial_Points         = getattr(ex, "Initial_Points", {}),
             Force                  = getattr(ex, "Force", {}),
             Reference_frame_Origin = getattr(ex, "Reference_frame_Origin", sym.Matrix([0, 0])),
             ForcesPointsSym        = getattr(ex, "ForcesPointsSym", []),
             BodyDataSym            = getattr(ex, "BodyDataSym", []),
             ic                     = getattr(ex, "ic", None),
             g                      = getattr(ex, "g", 9.81),
             gVec                   = np.asarray(getattr(ex, "gVec", np.ones((len(ex.joints),1))))
             )

    # ------------------- public API -------------------------------
    # -------------------- numeric workflow ------------------------
    def integrate(self, mainNumVars, m_vals, J_vals, *,
                  tspan:float|tuple, dt:float,
                  rtol=1e-8, atol=1e-8, algorithm="RK45",
                  external_manager=None):
        
        """
        Numerically integrate the equations of motion Q̈ = M⁻¹·RHS.

        Args:
            mainNumVars (Sequence[float]): Flattened initial values for [Q, QD,
                force-point parameters, body data].
            m_vals (Sequence[float]): Mass values per body.
            J_vals (Sequence[float]): Inertia values per body.
            tspan (float|tuple): End time or (start, end) for integration.
            dt (float): Time step for output.
            rtol, atol (float): Solver tolerances.
            solver (str): SciPy ODE solver name (e.g., "RK45").
            external_manager: Optional force manager for external coupling.

        Returns:
            OdeResult: Result object with time points `t` and solution `y`.
        """

        tspan = (0.0, float(tspan))

        # indices where `t` appears (used by integrate_dynamics), MUST INCLUDE REAL ASSUMPTION
        self.t_update = [i for i, s in enumerate(self.mainSymVars)
                         if s.has(sym.symbols("t", real=True))]

        # Define the MBD method to integrate the system
        fun = lambda t, y: integrate_dynamics(
                t, y, mainNumVars, m_vals, J_vals,
                self,external_manager=external_manager)

        t0_int = time()
        sol = solve_ivp(fun, tspan, self.ic,
                        t_eval   = np.arange(tspan[0], tspan[1]-1e-8, dt),
                        method   = algorithm, rtol=rtol, atol=atol)

        t_int = time() - t0_int
        print(f'Integrated {tspan[-1]:.2f} in {t_int:.3f} seconds')
        # Correct the line above
        return sol
    

    # ======================= INTERNALS ============================
    def __post_init__(self):
        """
        Internal initializer: build symbolic model and compile numeric functions.
        """
        # Ensure required subfields exist
        self.Initial_Points.setdefault("BD", {})
        self.Initial_Points.setdefault("GR", [])

        if self.gVec is None:
            self.gVec = np.ones((len(self.joints), 1))

        # Convert Reference_frame_Origin to sym.Matrix
        if not isinstance(self.Reference_frame_Origin, sym.Matrix):
            self.Reference_frame_Origin = sym.Matrix(self.Reference_frame_Origin)

        self._symbolic_build()      # prepare all SymPy objects
        self._compile_numeric()     # lambdify → NumPy callables
        self._ic_check()            # ensure initial conditions match joint definitions

    def _symbolic_build(self):
        """
        Symbolically construct kinematics, forces, and energy expressions.
        Mirrors Sections 1–4 of the procedural workflow.
        """

        # Joint system -------------------------------------------------
        t0_init     = time()
        pris_dir    = normalize_prismatic(self.prismatic_direction)
        
        self.joint_system = JointSystem.from_data(
                self.joints, self.types,
                self.parent_cg_to_joint, self.joint_to_child_cg, pris_dir)

        # symbolic coordinates ---------------------
        self.Q, self.QD, self.QDD, self.NDOF, self.Body_col = \
            self.joint_system.coordinate_finder()
        
        if self.ic is None:
            self.ic = np.zeros(len(self.Q) + len(self.QD))
        t_init = time() - t0_init
        print(f'Problem initialization finished in:\t\t {t_init:.3f} seconds')

        # symbolic position and velocity transformation matrices -------------
        t0_R = time()
        vt      = VelocityTransformation(self.joint_system, self.Reference_frame_Origin)
        
        self.Pos, self.JointLoc, self.R, self.RD, R_track = vt.compute_grounded()
        
        (self.Pos, self.JointLoc, self.R, self.RD, R_track,
         self.pris_body_to_parent_map, self.paths, self.graph) = \
            vt.compute_non_grounded(self.Pos, self.JointLoc,
                                    self.R, self.RD, R_track)

        # velocities / accelerations
        self.Vel    = self.R * sym.Matrix(self.QD)
        self.Acc    = self.R * sym.Matrix(self.QDD) + self.RD * sym.Matrix(self.QD)
        
        t_R = time() - t0_R
        print(f'R and RD matrices calculation finished in:\t {t_R:.3f} seconds')

        # points & forces ----------------------------------------------
        t0_EOM = time()
        NBodies     = len(self.joint_system.joints)
        
        (self.Points_All,
         self.Force_All,
         SpringPE,
         self.Vel_All) = points_force_finder(
                self.joint_system.joints, NBodies,
                self.Pos, self.Q, self.QD,
                self.Initial_Points, self.Force,
                self.JointLoc, self.pris_body_to_parent_map,
                self.Body_col, self.Vel, self.Reference_frame_Origin)

        # energies & EOM ----------------------------------------------
        self.Energy, self.M, self.m, self.J = systems_energy(
                NBodies, self.Q, self.QD, self.R,
                self.Points_All['CG'], self.g, self.gVec)
        
        self.Energy += SpringPE

        ReducedM        = self.R.T * self.M * self.R
        Fgravity_mat    = self.g * self.m.multiply_elementwise(sym.Matrix(self.gVec)) * sym.Matrix([0,-1,0]).T
        Fgravity        = Fgravity_mat.reshape(Fgravity_mat.rows * Fgravity_mat.cols,1)

        ForceExternal = (
            sym.Matrix.vstack(*self.Force_All["CG"])           +
            sym.Matrix.vstack(*self.Force_All["PointsBD"])     +
            sym.Matrix.vstack(*self.Force_All["TensionDamper"])+
            sym.Matrix.vstack(*self.Force_All["TensionSpring"])+
            sym.Matrix.vstack(*self.Force_All["TorsionDamper"])+
            sym.Matrix.vstack(*self.Force_All["TorsionSpring"])
        )
        self.ForceAllCombined   = ForceExternal + Fgravity
        Right_side_1            = - self.R.T * self.M * self.RD * sym.Matrix(self.QD)
        self.Right_side         = Right_side_1 + self.R.T * self.ForceAllCombined
        self.ReducedM           = ReducedM

        t_EOM = time() - t0_EOM
        print(f'Symbolic EOM computation finished in:\t\t {t_EOM:.3f} seconds')

        # used later when lambdifying
        self.mainSymVars     = self.Q + self.QD + \
                               self.ForcesPointsSym + \
                               self.BodyDataSym


    def _compile_numeric(self):
        """
        Lambdify symbolic expressions into NumPy callables.
        Mirrors Section 5 of the procedural workflow.
        """
        t0_lambdified = time()
        # lambdify core pieces (NumPy backend)
        modules = "numpy"
        self.Pos_func       = sym.lambdify(self.mainSymVars, self.Pos,      modules)
        self.JointLoc_func  = sym.lambdify(self.mainSymVars, self.JointLoc, modules)
        self.R_func         = sym.lambdify(self.mainSymVars, self.R,        modules)
        self.RD_func        = sym.lambdify(self.mainSymVars, self.RD,       modules)
        self.Vel_func       = sym.lambdify(self.mainSymVars, self.Vel,      modules)
        self.Acc_func       = sym.lambdify(self.mainSymVars + self.QDD, self.Acc, modules)
        
        # Similarly for Points_All:
        self.BDpoints_func  = {
            key: sym.lambdify((self.mainSymVars), self.Points_All['BD'][key], modules)
            for key in  self.Points_All['BD']
        }
        self.CGpoints_func  = sym.lambdify(self.mainSymVars, self.Points_All['CG'], modules)
        self.GRpoints_func  = sym.lambdify(self.mainSymVars, self.Points_All['GR'], modules)
        self.JOpoints_func  = sym.lambdify(self.mainSymVars, self.Points_All['JO'], modules)
        
        # For velocities:
        self.BDvel_func  = {
            key: sym.lambdify((self.mainSymVars), self.Vel_All['BD'][key], modules)
            for key in  self.Points_All['BD']
        }
        self.CGvel_func  = sym.lambdify(self.mainSymVars, self.Vel_All['CG'], modules)
        
        # EOM
        self.Force_func     = sym.lambdify(self.mainSymVars + list(self.m) + list(self.J),
                                 self.ForceAllCombined, modules)
        self.M_func         = sym.lambdify(self.mainSymVars + list(self.m.T)+list(self.J.T),
                                           self.M, modules) 
        self.ReducedM_func  = sym.lambdify(self.mainSymVars + list(self.m.T)+list(self.J.T),
                                           self.ReducedM, modules)
        self.Right_side_func= sym.lambdify(self.mainSymVars + list(self.m.T)+list(self.J.T),
                                           self.Right_side, modules)
        self.Energy_func    = sym.lambdify(self.mainSymVars + list(self.m.T)+list(self.J.T),
                                           self.Energy, modules)
        
        t_lambdified = time() - t0_lambdified
        print(f'Compiling EOM expressions finished in:\t\t {t_lambdified:.3f} seconds')


    def _ic_check(self, tol: float = 1e-7):
        """
        Internal check to ensure that for float joints with numerically defined
        geometry, the initial conditions match the joint definition.

        If geometry is symbolic, no changes are applied.
        """
        ic_corrected = self.ic.copy()

        for jointID,type in enumerate(self.types):
            if type != 'F':
                continue  # only care about float joints

            # Build the geometry vector from parent and child offsets
            geom_vec = self.parent_cg_to_joint[jointID]

            # Skip if geometry has symbolic values
            if any(isinstance(val, sym.Basic) for val in geom_vec):
                continue

            expected_pos = np.array(geom_vec, dtype=float)
            idx = self.Body_col[jointID]  # first 2 are x,z

            # If ic not provided this fails --> IC should always be provided
            x_ic, z_ic = ic_corrected[idx[0]], ic_corrected[idx[1]]
            if (abs(x_ic - expected_pos[0]) > tol or
                abs(z_ic - expected_pos[1]) > tol):

                print(
                    f"\n[Info] Updating float body {jointID+1} IC from "
                    f"(x={x_ic}, z={z_ic}) → "
                    f"(x={expected_pos[0]}, z={expected_pos[1]}) "
                    f"relative to Reference_frame_Origin: "
                    f"(x={self.Reference_frame_Origin[0]}, z={self.Reference_frame_Origin[1]}) .\n"
                )
                ic_corrected[idx[0]] = expected_pos[0]
                ic_corrected[idx[1]] = expected_pos[1]

        # Update the stored ICs in-place
        self.ic = ic_corrected



