# hydro_adapter.py
# Nonlinear hydrodynamics adapter for ExternalForcesManager.
#
# This adapter is aligned with `hydro_linear_mckf.py` in terms of:
#   - data loading (xarray.Dataset or dict-like) vs Capytaine on-the-fly
#   - 6-DOF -> 2D reduction convention: per body keep [Fx, Fz, My] = [0, 2, 4]
#   - buoyancy arm rotation: pitch about +Y (right-handed)
#
# The time-domain Cartesian force model implemented is:
#   F_hydro(t) = F_ex(t) - C * xdot - K * x + Fb
#
# NOTE (requested): hydrostatic term currently uses K @ x (no equilibrium subtraction).

from __future__ import annotations

from typing import Any, Dict, Optional, List
import numpy as np
import xarray as xr
from pathlib import Path

from multibody import ExternalForcesManager as manager

try:
    import capytaine as cpt  # Optional, only needed for on-the-fly hydro
except Exception:
    cpt = None


def _rot_y(theta: float) -> np.ndarray:
    """3D rotation about Y (right-handed)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]])


def _as_col(x: Any) -> np.ndarray:
    """Convert an array-like vector into a (n,1) column vector."""
    a = np.asarray(x)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        # If it's already a column, keep it; if it's a row, transpose
        if a.shape[1] == 1:
            return a
        if a.shape[0] == 1:
            return a.T
        # If it's neither row nor column, flatten to a column
        return a.reshape(-1, 1)
    raise ValueError("Cannot convert to column vector: unexpected ndim.")


class HydroInterface:
    """
    External-forces adapter that matches ExternalForcesManager's contract:
      - update(t, q, qd, mainNumVars)
      - forces() -> list[CartForce]
      - optionally exposes .added_inertia (Cartesian added mass)
    """
    name = "Hydro"

    def __init__(
        self,
        mbd_sys: Any,
        numericalValuesTuple: tuple,
        is_2D: bool,
        *,
        data: Optional[Any] = None,                         # Path A: xarray.Dataset or dict-like
        omega_r: Optional[np.ndarray] = None,               # Path B: frequency vector
        body_inputs: Optional[Dict[int, Any]] = None,       # Path B: dict of Capytaine bodies (or compatible)
        wave_amplitude: float = 1.0,
        equilibrium_pos: Optional[np.ndarray] = None,       # kept for parity; not used in K term (by request)
        operating_omega: Optional[float] = None,
        **kwargs,                                           # allow forward-compat / legacy aliases
    ) -> None:
        
        # Basic setup
        self.is_2D                          = is_2D
        self._mbd                           = mbd_sys
        self.mainNumVars, self.m0, self.J0  = numericalValuesTuple
        self.body_inputs                    = body_inputs

        # Store parameters
        self.wave_amplitude = float(wave_amplitude)
        self.rho            = float(kwargs.get("rho", 1025.0))
        self.g              = float(kwargs.get("g", mbd_sys.g))
        nq                  = len(mbd_sys.Q)

        # buoyancy metadata (per body)
        self.volumes: List[float]               = []
        self.buoyancy_arm: List[np.ndarray]     = []        # COM->COB in global at equilibrium
        self.CoB:List[np.ndarray]               = []        # Center of buoyancy in global coordinates at equilibrium
        self.Fb_3N: List[np.ndarray]            = []        # 3DOF buoyancy force per body in Cartesian ordering [Fx, Fz, My]
        self.omega_grid: Optional[np.ndarray]   = None

        # hydrodynamics data fields
        self.added_mass             = None
        self.radiation_damping      = None
        self.hydrostatic_stiffness  = None
        self.excitation_force       = None
        self.influenced_dof         = None
        self.save_dir               = None

        # Alias support (some scripts used waveheight)
        if "save_dir" in kwargs and kwargs["save_dir"] is not None:
            self.save_dir = str(kwargs["save_dir"])
            self.file_name = str(kwargs.get("file_name", "bem.nc"))
        if "load_dir" in kwargs and "file_name" in kwargs and kwargs["load_dir"] is not None:
            path = Path(kwargs["load_dir"])
            data = xr.open_dataset(path / kwargs["file_name"])
            data = cpt.io.xarray.merge_complex_values(data)

        if "waveheight" in kwargs and kwargs["waveheight"] is not None:
            wave_amplitude = float(kwargs["waveheight"])/2.0

        if "T_ramp" in kwargs and kwargs["T_ramp"] is not None:
            self.T_ramp = float(kwargs["T_ramp"])
        else:
            self.T_ramp = 0.0  # no ramp by default        

        # Equilibrium generalized coordinates (kept for interface parity)
        if equilibrium_pos is None:
            equilibrium_pos = np.asarray(self.mainNumVars[: len(mbd_sys.Q)], dtype=float)
        elif len(equilibrium_pos) != nq:
            raise ValueError("equilibrium_pos length must match number of generalized coordinates")
        
        if operating_omega is not None:
            self.omega = float(operating_omega)
        else:
            raise ValueError("Provide `operating_omega` for time-domain simulation.")

        equilibrium_pos      = np.hstack([equilibrium_pos, np.zeros(len(mbd_sys.QD), dtype=float)])
        self.equilibrium_pos = equilibrium_pos.reshape(-1)

        # Equilibrium Cartesian positions for hydrostatic restoring (F_hs = K @ (x - x_eq))
        self.CG_equib = np.asarray(self._mbd.Pos_func(*self.equilibrium_pos), dtype=float)

        # Prepare FloatingBody objects and compute buoyancy metadata
        prepared_bodies = self._prepare_bodies_for_bem() if body_inputs is not None else {}
        all_bodies = self._compute_with_capytaine(prepared_bodies)
        
        # Load or compute hydrodynamic data
        if data is not None:
            self._load_data(data)
        else:
            if cpt is None:
                raise ImportError("capytaine is required to compute hydrodynamics when `data` is not provided.")
            if omega_r is None or body_inputs is None:
                raise ValueError("Provide `data` or (`omega_r` and `body_inputs`).")
            
            self.omega_grid = np.asarray(omega_r, dtype=float).flatten()
            self._compute_bem(all_bodies, self.save_dir)

        # Cached outputs for ExternalForcesManager
        self._cached_forces = []
        self.added_inertia  = None  # set in update()

        # Store forces
        self.integration_t  = []
        self.F_excitation   = []
        self.F_total        = []

        # reorder matrices as "influenced_dof", "radiating_dof"
        self.added_mass             = self.added_mass.transpose("omega","influenced_dof", "radiating_dof")
        self.radiation_damping      = self.radiation_damping.transpose("omega","influenced_dof", "radiating_dof")
        self.hydrostatic_stiffness  = self.hydrostatic_stiffness.transpose("influenced_dof", "radiating_dof")

    # ------------------------------------------------------------------
    # required by the adapter contract
    # ------------------------------------------------------------------
    def update(self, t, q, qd, mainNumVars):
        """
        Compute and cache hydrodynamic forces and added mass at (t, state).
        ExternalForcesManager will convert cached forces to generalized space.
        """
        # Cartesian positions and velocities (3*nb,1) in the MBD embedding: [x, z, theta] per body
        CGpos = _as_col(np.asarray(self._mbd.Pos_func(*mainNumVars), dtype=float))
        CGvel = _as_col(np.asarray(self._mbd.Vel_func(*mainNumVars), dtype=float))

        F_total, Madd = self._compute_forces_and_added_mass(t, CGpos, CGvel)

        # Expose Cartesian added mass so ExternalForcesManager reduces it to joint space
        self.added_inertia = Madd

        # Convert to CartForce list
        self._cached_forces = manager._force2cart(F_total, self._mbd)

    def forces(self):
        return list(self._cached_forces)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _compute_forces_and_added_mass(self, t: float, CGpos: np.ndarray, CGvel: np.ndarray):
        """
        Returns:
          F_total: (3*nb,1) time-domain Cartesian force vector [Fx,Fz,My] per body
          Madd   : (3*nb,3*nb) Cartesian added-mass matrix
        """
        nb = len(self._mbd.NDOF)

        # Interpolate hydrodynamic tensors at operating frequency
        M_full = self._interp("added_mass", self.omega)
        C_full = self._interp("radiation_damping", self.omega)
        K_full = self._get("hydrostatic_stiffness")  # usually frequency-independent
        Ex_full = self._interp("excitation_force", self.omega)  # complex

        # Reduce if dataset is 6DOF and the simulation embedding is planar (3 dof per body)
        if not self.is_2D:
            Madd = self._reduce_2D(M_full, nb)
            C = self._reduce_2D(C_full, nb)
            K = self._reduce_2D(K_full, nb)
            Ex = self._reduce_vec_2D(Ex_full, nb)  # (3*nb,1), complex
        else:
            Madd = np.asarray(M_full, dtype=float)
            C = np.asarray(C_full, dtype=float)
            K = np.asarray(K_full, dtype=float)
            Ex = _as_col(np.asarray(Ex_full))  # keep complex dtype if present

        # Time-domain regular-wave excitation (real part)
        r = self.ramp_function(t, self.T_ramp)
        F_ex = r * np.real(Ex * self.wave_amplitude * np.exp(-1j * self.omega * t))

        # Buoyancy (static): force up, moment = r x F (y-moment retained)
        Fb = self._buoyancy_force_2D(CGpos)

        # Radiation damping and hydrostatic restoring
        F_rad   = C @ CGvel
        F_hs    = K @ (CGpos - self.CG_equib)  

        # Total force applied to the nonlinear integrator
        F_total = F_ex - F_rad - F_hs + Fb

        # Store forces for post-processing / debugging
        self.integration_t.append(t)
        self.F_excitation.append(F_ex)
        self.F_total.append(F_total)

        return _as_col(F_total), Madd

    def _buoyancy_force_2D(self, CGpos: np.ndarray) -> np.ndarray:
        """Return (3*nb,1) buoyancy vector in [Fx,Fz,My] per body."""
        nb = len(self._mbd.NDOF)

        blocks = []
        for i in range(nb):
            vol = float(self.volumes[i]) if i < len(self.volumes) else 0.0
            arm0 = self.buoyancy_arm[i] if i < len(self.buoyancy_arm) else np.zeros(3)
            arm0[1] = 0.0  # ensure arm is in x-z plane for 2D reduction
            arm0[0] = 0.0 # NOTE: temporary test

            F_b = vol * self.rho * self.g  # upward buoyancy magnitude

            # theta lives in the 3rd entry per body in the planar embedding
            theta = float(CGpos[3 * i + 2, 0])
            arm = (_rot_y(theta) @ arm0.reshape(3, 1)).ravel()
            arm = arm0

            Fv = np.array([0.0, 0.0, F_b])
            Mv = np.cross(arm, Fv)
            My = float(Mv[1])

            blocks.append(np.array([0.0, F_b, My]))

        if not blocks:
            return np.zeros((0, 1))
        return _as_col(np.concatenate(blocks))

    # -----------------------------
    # Data loading / computation
    # -----------------------------
    def _load_data(self, data: Any) -> None:
        """Normalize xarray.Dataset or dict-like into adapter attributes."""
        # xarray style
        for name in list(data.data_vars):
            setattr(self, name, data[name])

        # Extract variables from coords with fallbacks
        coords          = dict(getattr(data, "coords", {}))
        self.rho        = float(getattr(data, "rho", getattr(data.coords, "rho", 1025.0)))
        self.g          =  float(getattr(data, "g", getattr(data.coords, "g", 9.81)))
        self.omega_grid = coords.get("omega", None)

    def _prepare_bodies_for_bem(self) -> Dict[int, "cpt.FloatingBody"]:
        """Create and configure FloatingBody objects from body_inputs descriptors."""
        eq_vars    = self.equilibrium_pos.copy()
        CGpos_2d   = np.asarray(self._mbd.CGpoints_func(*eq_vars), dtype=float)  # (NB, 2)
        all_cgs_3d = np.insert(CGpos_2d, 1, 0.0, axis=1)                          # (NB, 3)
        first_cg   = all_cgs_3d[0]

        prepared = {}
        for i, descriptor in self.body_inputs.items():
            # Backward compatibility: accept pre-built FloatingBody as-is
            if isinstance(descriptor, cpt.FloatingBody):
                prepared[i] = descriptor
                continue

            cg       = all_cgs_3d[i - 1]
            mesh_ref = descriptor.get("mesh_reference", "absolute")

            body = cpt.FloatingBody(mesh=descriptor["mesh"], name=descriptor["name"])

            if mesh_ref == "body_cg":
                body.translate(cg)
            elif mesh_ref == "first_body_cg":
                body.translate(first_cg)
            # "absolute": use mesh as-is

            body.center_of_mass  = cg.copy()
            body.rotation_center = cg.copy()
            body.add_all_rigid_body_dofs()

            inertia_diag = np.asarray(descriptor["inertia_diag"], dtype=float)
            body.inertia_matrix = xr.DataArray(
                data=np.diag(inertia_diag),
                dims=["influenced_dof", "radiating_dof"],
                coords={
                    "influenced_dof": list(body.dofs),
                    "radiating_dof":  list(body.dofs),
                },
                name="inertia_matrix",
            )

            prepared[i] = body

        print("\n[HydroAdapter] Prepared bodies:")
        for i, body in prepared.items():
            print(f"  [{i}] name={body.name!r}, CoM={body.center_of_mass}")

        return prepared

    def _compute_with_capytaine(self, prepared_bodies: Dict[int, Any]) -> Any:
        self.volumes.clear()
        self.buoyancy_arm.clear()

        all_bodies = None
        for i, body in prepared_bodies.items():
            body.compute_hydrostatics()
            all_bodies = body if all_bodies is None else (all_bodies + body)

            arm    = np.asarray(body.center_of_buoyancy - body.center_of_mass, dtype=float)
            arm[1] = 0.0
            self.buoyancy_arm.append(arm)
            self.volumes.append(float(body.volume))
            self.CoB.append(np.asarray(body.center_of_buoyancy, dtype=float))

        return all_bodies

    def _compute_bem(self, all_bodies, save_dir=None) -> None:
        # Setup radiation and diffraction problems for all bodies and frequencies
        rad = [
            cpt.RadiationProblem(body=all_bodies, radiating_dof=dof, omega=w,rho=self.rho, g=self.g)
            for dof in all_bodies.dofs
            for w in self.omega_grid
        ]

        dif     = [cpt.DiffractionProblem(body=all_bodies, wave_direction=0.0, omega=w,rho=self.rho, g=self.g) for w in self.omega_grid]
        solver  = cpt.BEMSolver()
        results = solver.solve_all(rad + dif)
        dataset = cpt.assemble_dataset(results)
        self._load_data(dataset)

        if save_dir is not None:
            eq_vars     = self.equilibrium_pos.copy()
            CGs2D       = np.asarray(self._mbd.CGpoints_func(*eq_vars), dtype=float)
            CGs         = np.insert(CGs2D, 1, 0.0, axis=1)

            # Save hydrostactic data too if save_dir is provided
            V           = np.asarray(self.volumes, dtype=float).reshape(-1)
            CoB         = np.asarray(self.CoB, dtype=float)
            Nb          = V.size
            body_names  = [
                (b.name if isinstance(b := self.body_inputs[i + 1], cpt.FloatingBody) else b["name"])
                for i in range(Nb)
            ]

            dataset = dataset.assign_coords(body_name=body_names, axis=["x", "y", "z"]).assign(
                volume=xr.DataArray(V, dims=("body_name",)),
                center_of_buoyancy=xr.DataArray(CoB, dims=("axis", "body_name")),
                center_of_mass=xr.DataArray(CGs, dims=("axis", "body_name"))
            )
            self.save_bem_data(save_dir, dataset, self.file_name)

    # -----------------------------
    # Helpers: get/interp and reduce
    # -----------------------------
    def _get(self, name: str) -> np.ndarray:
        arr = getattr(self, name)
        try:
            return np.asarray(arr.values)
        except Exception:
            return np.asarray(arr)

    def _interp(self, name: str, omega: float) -> np.ndarray:
        arr = getattr(self, name)

        # --- xarray path
        if hasattr(arr, "coords") and "omega" in arr.coords:
            omega_grid = np.asarray(arr.coords["omega"].values)

            # Case 1: single-frequency dataset → direct indexing
            if omega_grid.size == 1:
                return np.asarray(arr.isel(omega=0).values)

            # Case 2: exact match → snap to grid
            idx = np.argmin(np.abs(omega_grid - omega))
            if np.isclose(omega_grid[idx], omega):
                return np.asarray(arr.isel(omega=idx).values)

            # Case 3: genuine interpolation
            return np.asarray(arr.interp(omega=omega, method="linear").values)

        # --- fallback (already evaluated data)
        return self._get(name)


    @staticmethod
    def _reduce_index(nb: int) -> np.ndarray:
        # (Fx, Fy, Fz, Mx, My, Mz) -> keep (Fx, Fz, My) == indices [0,2,4]
        dofs = np.array([0, 2, 4])
        return (6 * np.arange(nb)[:, None] + dofs).reshape(-1)

    def _reduce_2D(self, A: np.ndarray, nb: int) -> np.ndarray:
        idx = self._reduce_index(nb)
        A = np.asarray(A)
        if A.ndim == 2:
            return A[np.ix_(idx, idx)]
        if A.ndim == 3:
            I, J = np.ix_(idx, idx)
            return A[:, I, J]
        raise ValueError("Expected a 2D or 3D matrix for reduction.")

    def _reduce_vec_2D(self, v: Any, nb: int) -> np.ndarray:
        idx = self._reduce_index(nb)
        v = np.asarray(v)

        # Typical shapes after xarray interp at scalar omega:
        #  - (6*nb,) or (6*nb,1) or (1,6*nb)
        if v.ndim == 1:
            return v[idx].reshape(-1, 1)
        if v.ndim == 2:
            if v.shape[1] == 1:
                return v[idx, :]
            if v.shape[0] == 1:
                return v[:, idx].T
            # If it's an unexpected 2D shape, flatten and reduce
            return v.reshape(-1, 1)[idx, :]
        raise ValueError("Expected a vector-like array for reduction.")
    
    @staticmethod
    def ramp_function(t, T_ramp):
        t       = np.asarray(t)
        r       = np.ones_like(t)
        mask    = t < T_ramp

        r[mask] = 0.5 * (1.0 - np.cos(np.pi * t[mask] / T_ramp))

        return np.float64(r)
    
    @staticmethod
    def save_bem_data(save_dir: str, dataset: Any, file_name: str = "bem.nc") -> None:
        if cpt is None:
            raise ImportError("capytaine is required to save BEM data.")

        path = Path(save_dir)
        if path.suffix != ".nc":
            path.mkdir(parents=True, exist_ok=True)
            path = path / file_name
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        ds = dataset

        # --- fix: convert pandas Categorical coords/vars to plain strings
        try:
            import pandas as pd

            # Coords
            for cname in list(getattr(ds, "coords", {})):
                try:
                    idx = ds.coords[cname].to_index()
                    if isinstance(idx, pd.CategoricalIndex) or pd.api.types.is_categorical_dtype(idx.dtype):
                        ds = ds.assign_coords({cname: ds.coords[cname].astype(str)})
                except Exception:
                    # If it doesn't have an index or fails, skip
                    pass

            # Data variables (rare, but safe)
            for vname in list(getattr(ds, "data_vars", {})):
                try:
                    if pd.api.types.is_categorical_dtype(ds[vname].dtype):
                        ds[vname] = ds[vname].astype(str)
                except Exception:
                    pass

        except Exception:
            # If pandas isn't available or something odd happens, proceed without conversion.
            pass

        # Capytaine helper to serialize complex-valued datasets safely
        cpt.io.xarray.separate_complex_values(ds).to_netcdf(path)

