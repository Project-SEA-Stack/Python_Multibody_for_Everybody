# multibody/linearization/hydro_linear_mckf.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List, Union
from networkx import omega
from networkx import omega
import numpy as np
import sympy as sym
import xarray as xr

import capytaine as cpt
from pathlib import Path

import sympy as sym


def rot_2D_wave(angle):
    """
    Returns a 2x2 rotation matrix for a rotation about the y-axis
    with the sign convention changed (per your specification).
    """
    c = sym.cos(angle)
    s = sym.sin(angle)
    return sym.Matrix([[c,0,s],
                [0, 1, 0],
                [-s, 0, c]])


class HydroLinearMCKF:
    """
    Frequency-domain hydrodynamics adapter with force dict convention:
      returns (M(ω), C(ω), K, F) where F = {"dc": F_dc_sym, "phasor": Fhat(ω)}

    - F_dc_sym is a SymPy Cartesian (3N,1) DC force vector (buoyancy) that depends on q via theta_i.
    - Fhat(ω) is complex excitation phasor in Cartesian (3N,1) or (nω,3N,1).

    Cartesian ordering per body: [Fx, Fz, My].
    """

    name = "HydroLinearMCKF"

    def __init__(
        self,
        mbd_sys: Any,
        numericalValuesTuple: tuple,
        is_2D: bool = False,
        *,
        data: Optional[Any] = None,
        omega_r: Optional[np.ndarray] = None,
        body_inputs: Optional[Dict[int, Any]] = None, # TODO: this is not optional? --> needs testing
        wave_amplitude: float = 1.0,
        equilibrium_pos: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> None:
    
        # Unpack basics
        self.is_2D                          = bool(is_2D)
        self._mbd                           = mbd_sys
        self.mainNumVars, self.m0, self.J0  = numericalValuesTuple
        self.body_inputs                    = body_inputs

        # Store parameters
        self.wave_amplitude = float(wave_amplitude)
        self.rho            = float(kwargs.get("rho", 1025.0))
        self.g              = float(kwargs.get("g", mbd_sys.g))

        # buoyancy metadata (per body)
        self.volumes: List[float]           = []
        self.arm_global: List[np.ndarray]   = []        # COM->COB in global at equilibrium
        self.arm_body: List[np.ndarray]     = []        # COM->COB in body frame @ θ=0
        self.Fb_3N: List[np.ndarray]        = []        # 3DOF buoyancy force per body in Cartesian ordering [Fx, Fz, My]

        # hydrodynamics data fields
        self.added_mass             = None
        self.radiation_damping      = None
        self.hydrostatic_stiffness  = None
        self.excitation_force       = None
        self.influenced_dof         = None
        self.save_dir               = None

        if "save_dir" in kwargs and kwargs["save_dir"] is not None:
            self.save_dir   = str(kwargs["save_dir"])
            self.file_name  = kwargs.get("file_name", "bem.nc")
        if "load_dir" in kwargs and "file_name" in kwargs and kwargs["load_dir"] is not None:
            path            = Path(kwargs["load_dir"])
            data            = xr.open_dataset(path / kwargs["file_name"])
            data            = cpt.io.xarray.merge_complex_values(data)

        if equilibrium_pos is None:
            equilibrium_pos     = np.asarray(self.mainNumVars[: len(mbd_sys.Q)], dtype=float)

        equilibrium_pos         = np.hstack([equilibrium_pos, np.zeros(len(mbd_sys.QD), dtype=float)])
        self.equilibrium_pos    = equilibrium_pos.reshape(-1)

        # Prepare FloatingBody objects and compute buoyancy metadata
        prepared_bodies = self._prepare_bodies_for_bem() if body_inputs is not None else {}
        all_bodies      = self._compute_with_hydrostatics(prepared_bodies) # NOTE: this may be overwritting directly passed data
        
        # Load data from .nc file or compute with capytaine if not provided
        if data is not None:
            self._load_data(data, omega_r)
        else:
            if cpt is None:
                raise ImportError("capytaine is required if `data` is not provided.")
            if omega_r is None or body_inputs is None:
                raise ValueError("Provide `data` or (`omega_r` and `body_inputs`).")
            self._compute_bem(all_bodies, np.asarray(omega_r, dtype=float),self.save_dir)

        # build symbolic DC force once
        self._Fdc_sym = self.Fb_3N

        # Ensure ordering of xarray according to "omega","influenced_dof", "radiating_dof
        self.added_mass             = self.added_mass.transpose("omega","influenced_dof", "radiating_dof")
        self.radiation_damping      = self.radiation_damping.transpose("omega","influenced_dof", "radiating_dof")
        self.hydrostatic_stiffness  = self.hydrostatic_stiffness.transpose("influenced_dof", "radiating_dof")

    # ------------------------------------------------------------
    # Adapter API
    # ------------------------------------------------------------
    def frequency_domain_MCKF(
        self, omega: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Returns:
          M(ω), C(ω), K, F where:
            F["dc"]     = SymPy Matrix (3N,1)
            F["phasor"] = complex ndarray, shape (3N,1) or (nω,3N,1)
        """
        omega_is_scalar = np.isscalar(omega)
        omega_arr       = float(omega) if omega_is_scalar else np.asarray(omega, dtype=float).reshape(-1)

        nb  = len(self._mbd.NDOF)

        M6  = self._interp("added_mass", omega_arr)
        C6  = self._interp("radiation_damping", omega_arr)
        K6  = self._mat("hydrostatic_stiffness")
        Ex6 = self._interp("excitation_force", omega_arr)

        Fhat6 = self._excitation_to_vector(Ex6, dof=6 * nb) * self.wave_amplitude

        if not self.is_2D:
            M = self._reduce_mat(M6, nb)
            C = self._reduce_mat(C6, nb)
            K = self._reduce_mat(K6, nb)
            Fhat = self._reduce_vec(Fhat6, nb)
        else:
            M = np.asarray(M6, dtype=float)
            C = np.asarray(C6, dtype=float)
            K = np.asarray(K6, dtype=float)
            Fhat = self._as_col_or_omega_col(Fhat6)

        if omega_is_scalar and M.ndim == 3:
            M, C, Fhat = M[0], C[0], Fhat[0]

        F = {"dc": self._Fdc_sym, "phasor": Fhat}
        return M, C, K, F
    
    def assemble_time_forces_from_linear_solution(
        self,
        t: np.ndarray,
        y: np.ndarray,
        mbd_sys: Any,
        mainNumVars: np.ndarray,
        *,
        omega0: float,
        R0: np.ndarray,
        ramp_T: float = 0.0,
        include_buoyancy: bool = True,
        return_forces_inJC: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Assemble time-domain hydrodynamic forces using joint states from the linear integrator.
        This i s a post-processing step that maps the integrated joint trajectories to Cartesian 
        forces using the frequency-domain MCKF data and the planar embedding defined by R0.

        """

        nq = R0.shape[1]
        t  = np.asarray(t, dtype=float).ravel()        # defensive: accept list or ndarray
        y  = np.asarray(y, dtype=float)                 # defensive: accept list or ndarray
        if y.shape[0] != 2 * nq:                        # (2*nq, nt) expected; transpose if (nt, 2*nq)
            y = y.T
        nt = t.size

        q  = y[:nq, :]      # (nq, nt)
        qd = y[nq:, :]     # (nq, nt)

        # Get frequency-domain Cartesian tensors + excitation phasor at omega0
        M, C, K, F  = self.frequency_domain_MCKF(float(omega0))
        Fhat        = np.asarray(F["phasor"], dtype=complex).reshape(-1, 1)  # (ncart,1)

        C = np.asarray(C, dtype=float)
        K = np.asarray(K, dtype=float)

        ncart = R0.shape[0]

        # Precompute ramp + complex exponential
        r       = self._ramp(t, ramp_T).reshape(nt, 1, 1)                   # (nt,1,1)
        exp_it  = np.exp(-1j * float(omega0) * t).reshape(nt, 1, 1)         # (nt,1,1)

        # Map joint -> Cartesian (planar embedding)
        # x(t)  = R0 q(t)
        # xd(t) = R0 qd(t)
        # NOTE: the following may not be correct, review
        x   = (R0 @ q).T.reshape(nt, ncart, 1)      # (nt,ncart,1)
        xd  = (R0 @ qd).T.reshape(nt, ncart, 1)   # (nt,ncart,1)

        # Excitation force (time domain)
        F_ex = r * np.real(Fhat.reshape(1, ncart, 1) * exp_it)              # (nt,ncart,1)

        # Radiation damping and hydrostatic restoring
        F_rad = -np.einsum("ij,tjk->tik", C, xd)                             # C @ xd
        F_hs  = -np.einsum("ij,tjk->tik", K, x)                              # K @ x

        # Buoyancy (optional)
        # F_b = np.zeros((nt, ncart, 1), dtype=float)
        F_b = self.Fb_3N.reshape(1, ncart, 1) if include_buoyancy else np.zeros((nt, ncart, 1), dtype=float) 

        # Total
        F_total = F_ex + F_rad + F_hs + F_b

        out = {
            "F_exc": F_ex,
            "F_rad": F_rad,
            "F_hs": F_hs,
            "F_b": F_b,
            "F_total": F_total,
        }

        if return_forces_inJC:
            # Q_total(t) = R0^T F_total(t)
            Q_total = np.einsum("ji,tik->tjk", R0, F_total)               # (nt,nq,1)
            out["Q_total"] = Q_total

        return out

    # ------------------------------------------------------------
    # data loading / capytaine compute
    # ------------------------------------------------------------
    def _load_data(self, data: Any, omega_r: np.ndarray) -> None:
        user_rho = self.rho  # preserve user-specified density before nc may differ

        # xarray-like
        for name in list(data.data_vars):
            setattr(self, name, data[name])

        # Extract variables from coords with fallbacks
        coords          = dict(getattr(data, "coords", {}))
        nc_rho          = float(getattr(data, "rho", getattr(data.coords, "rho", 1025.0)))
        self.g          = float(getattr(data, "g", getattr(data.coords, "g", 9.81)))
        self.omega      = coords.get("omega", None)
        wave_direction  = coords.get("wave_direction", None)

        # Rescale rho-proportional BEM quantities when user rho differs from the nc rho.
        # All dimensional Capytaine outputs (added mass, damping, forces, stiffness) are
        # proportional to the fluid density used during the BEM solve (nc_rho). If the
        # simulation uses a different density (user_rho), every coefficient must be rescaled
        # by user_rho / nc_rho so that forces, inertia, and stiffness are consistent.
        # NOTE: this is only a temporary solution. We are not normalizing coefficients as wecSim does.
        rho_scale = user_rho / nc_rho
        if abs(rho_scale - 1.0) > 1e-6:
            _rho_fields = (
                'added_mass', 'radiation_damping',
                'diffraction_force', 'Froude_Krylov_force', 'excitation_force',
                'hydrostatic_stiffness',
            )
            for fname in _rho_fields:
                val = getattr(self, fname, None)
                if val is not None:
                    setattr(self, fname, val * rho_scale)

        self.rho = user_rho  # keep user-specified rho (do not override with nc value)

        self.coords_exc = {
            'omega': (['omega'], self.omega.values, {'units': 'rad/s'}),
            'wave_direction': (['wave_direction'], wave_direction.values, {'units': 'rad'}),
            'influenced_dof': (['influenced_dof'], self.influenced_dof, []),
        }

        return

    def _prepare_bodies_for_bem(self) -> Dict[int, cpt.FloatingBody]:
        """Create and configure FloatingBody objects from body_inputs descriptors."""
        sumDOF     = self.equilibrium_pos.shape[0]
        eq_vars    = np.hstack((self.equilibrium_pos.copy(), self.mainNumVars[sumDOF:]))
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

    def _compute_with_hydrostatics(self, prepared_bodies: Dict[int, cpt.FloatingBody]) -> Any:
        self.volumes.clear()
        self.arm_global.clear()

        all_bodies = None
        Fb_3N      = []

        for i, body in sorted(prepared_bodies.items()):  # sort by body index so Fb_3N matches MBD Cartesian ordering
            body.compute_hydrostatics()
            all_bodies = body if all_bodies is None else (all_bodies + body)

            # TODO: remember to add option for arm rotation for nonlinear buoyancy
            arm    = np.asarray(body.center_of_buoyancy - body.center_of_mass, dtype=float)
            arm[1] = 0.0
            # arm[0] = 0.0
            self.arm_global.append(arm)
            self.volumes.append(float(body.volume))

            Fz  = self.rho * self.g * body.volume * np.array([0.0, 0.0, 1.0])
            My  = np.cross(arm, Fz)
            f_b = np.array([0.0, Fz[2], My[1]], dtype=float)
            Fb_3N.append(f_b)

        if Fb_3N:
            self.Fb_3N = np.hstack(Fb_3N).reshape(-1, 1)

        return all_bodies
    
    def _compute_bem(self, all_bodies, omega_r: np.ndarray, save_dir=None) -> None:
        # Setup radiation and diffraction problems for all bodies and frequencies
        rad = [
            cpt.RadiationProblem(body=all_bodies, radiating_dof=dof, omega=w,rho=self.rho, g=self.g)
            for dof in all_bodies.dofs
            for w in omega_r
        ]

        dif     = [cpt.DiffractionProblem(body=all_bodies, wave_direction=0.0, omega=w,rho=self.rho, g=self.g) for w in omega_r]
        solver  = cpt.BEMSolver()
        results = solver.solve_all(rad + dif)
        dataset = cpt.assemble_dataset(results)
        self._load_data(dataset, omega_r)

        if save_dir is not None:
            # eq_vars     = self.equilibrium_pos.copy()
            sumDOF     = self.equilibrium_pos.shape[0]
            eq_vars    = np.hstack((self.equilibrium_pos.copy(), self.mainNumVars[sumDOF:]))
            CGs2D       = np.asarray(self._mbd.CGpoints_func(*eq_vars), dtype=float)
            CGs         = np.insert(CGs2D, 1, 0.0, axis=1)
            CoB         = np.asarray(self.arm_global, dtype=float) + CGs  # (3,NB)

            # Save hydrostactic data too if save_dir is provided
            V           = np.asarray(self.volumes, dtype=float).reshape(-1)
            CoB         = np.asarray(CoB, dtype=float)
            Nb          = V.size
            body_names  = [
                (b.name if isinstance(b := self.body_inputs[i + 1], cpt.FloatingBody) else b["name"])
                for i in range(Nb)
            ]

            dataset = dataset.assign_coords(body_name=body_names, axis=["x", "y", "z"]).assign(
                volume=xr.DataArray(V, dims=("body_name",)),
                center_of_buoyancy=xr.DataArray(CoB.T, dims=("axis", "body_name")),
                center_of_mass=xr.DataArray(CGs.T, dims=("axis", "body_name"))
            )

            self.save_bem_data(save_dir, dataset, self.file_name)

    # ------------------------------------------------------------
    # DOF mapping
    # ------------------------------------------------------------
    def _extract_independent_dofs(self) -> None:
        """
        Extract and map independent degrees of freedom from capytaine data.
        
        Maps 6DOF per body from data.coords to independent DOFs based on self._mbd.Q.
        Creates human-readable DOF names in format: body_name_dof_name.
        
        DOF name mappings:
        - X -> surge
        - Z -> heave
        - theta -> pitch
        - s -> heave (NOTE: 's' represents either heaving, surging, or combination; 
                     currently identified as heave motion)
        """
        # Create body number to name mapping
        body_name_map = {}
        if self.body_inputs is not None:
            for body_num, body in self.body_inputs.items():
                body_name_map[body_num] = (
                    body.name if isinstance(body, cpt.FloatingBody) else body["name"]
                )
        
        # Define DOF name mappings
        dof_type_map = {
            'X': 'surge',
            'Z': 'heave',
            'theta': 'pitch',
            's': 'heave'  # NOTE: 's' can represent heaving, surging, or combination;
                          # currently identified as heave motion
        }
        
        # Extract independent DOFs from mbd system
        independent_dof_list = []
        
        for dof in self._mbd.Q:
            # Parse DOF string to get type and body number
            # DOF format: {prefix}_{body_number}
            dof_str = str(dof)
            parts = dof_str.rsplit('_', 1)  # Split from right to get body number
            
            if len(parts) == 2:
                dof_type = parts[0]
                body_num = int(parts[1])
                
                # Get body name
                body_name = body_name_map.get(body_num, f"body{body_num}")
                
                # Get mapped DOF name
                dof_name = dof_type_map.get(dof_type, dof_type)
                
                # Create full DOF identifier
                full_dof_name = f"{body_name}_{dof_name}"
                independent_dof_list.append(full_dof_name)
        
        # Store the result
        self.influenced_dof = independent_dof_list

    # ------------------------------------------------------------
    # array helpers
    # ------------------------------------------------------------
    def _mat(self, name: str) -> np.ndarray:
        arr = getattr(self, name)
        try:
            return np.asarray(arr.values)
        except Exception:
            return np.asarray(arr)

    def _interp(self, name: str, omega: Union[float, np.ndarray]) -> np.ndarray:
        arr = getattr(self, name)
        try:
            # Single omega point: linear interpolation requires ≥2 points — return directly
            if "omega" in arr.coords and arr.coords["omega"].size == 1:
                raw = arr.values  # shape (1, ...) with omega as first dim
                if np.isscalar(omega):
                    return raw.squeeze(axis=0)   # drop omega dim → (...)
                else:
                    return raw                    # keep omega dim → (1, ...)
            return np.asarray(arr.interp(omega=omega, method="linear").values)
        except Exception:
            return np.asarray(arr)

    @staticmethod
    def _reduce_index(nb: int) -> np.ndarray:
        # 6DOF -> 3DOF: (Fx,Fy,Fz,Mx,My,Mz) -> (Fx,Fz,My)
        dofs = np.array([0, 2, 4])
        return (6 * np.arange(nb)[:, None] + dofs).reshape(-1)

    def _reduce_mat(self, A: np.ndarray, nb: int) -> np.ndarray:
        idx = self._reduce_index(nb)
        A = np.asarray(A)
        if A.ndim == 2:
            return A[np.ix_(idx, idx)]
        if A.ndim == 3:
            return A[:, idx][:, :, idx]
        raise ValueError(f"Expected 2D or 3D matrix, got shape {A.shape}")

    def _reduce_vec(self, v: np.ndarray, nb: int) -> np.ndarray:
        idx = self._reduce_index(nb)
        v = self._as_col_or_omega_col(v)
        if v.ndim == 2:
            return v[idx, :]
        if v.ndim == 3:
            return v[:, idx, :]
        raise ValueError(f"Expected 2D or 3D vector, got shape {v.shape}")

    @staticmethod
    def _as_col_or_omega_col(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v)
        if v.ndim == 1:
            return v.reshape(-1, 1)
        if v.ndim == 2:
            if v.shape[1] == 1:
                return v
            return v[:, 0:1]
        if v.ndim == 3:
            return v[:, :, 0:1]
        raise ValueError(f"Cannot coerce force array shape {v.shape}")

    @classmethod
    def _excitation_to_vector(cls, Ex: np.ndarray, dof: int) -> np.ndarray:
        Ex = np.asarray(Ex)
        if Ex.ndim == 1:
            if Ex.size != dof:
                raise ValueError(f"Excitation size {Ex.size} != dof {dof}")
            return Ex.reshape(dof, 1)
        if Ex.ndim == 2:
            if Ex.shape[0] == dof:
                return Ex[:, 0:1]
            if Ex.shape[1] == dof:
                return Ex[0:1, :].T
        if Ex.ndim == 3:
            # (nω, dof, wave_dir) or (nω, wave_dir, dof)
            if Ex.shape[1] == dof:
                return Ex[:, :, 0:1]
            if Ex.shape[2] == dof:
                return Ex[:, 0:1, :].transpose(0, 2, 1)
        raise ValueError(f"Unrecognized excitation shape {Ex.shape} for dof={dof}")
    
    @staticmethod
    def save_bem_data(save_dir: str, dataset: Any, file_name: str) -> None:
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

    @staticmethod
    def _ramp(t: np.ndarray, T: float) -> np.ndarray:
        """Cosine ramp used in LinearizationManager.ramp()."""
        t = np.asarray(t, dtype=float)
        if T is None or float(T) <= 0.0:
            return np.ones_like(t)
        T = float(T)

        r = np.ones_like(t)
        r[t <= 0.0] = 0.0
        mask = (t > 0.0) & (t < T)
        r[mask] = 0.5 * (1.0 - np.cos(np.pi * t[mask] / T))
        r[t >= T] = 1.0
        return r
