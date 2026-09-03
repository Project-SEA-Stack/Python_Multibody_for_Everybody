# multibody/linearization/linearization_main.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import numpy.typing as npt
import sympy as sym
from scipy.integrate import solve_ivp
from time import time

from multibody.linearization.linearize_eom import linearize_mbd
from multibody.linearization.linear_integrator import linear_integrator
from ._helper_linearization import detect_space, as_col, broadcast_omega_mat, broadcast_omega_vec


@runtime_checkable
class LinearizationAdapter(Protocol):
    """
    Adapter returns frequency-domain M,C,K plus a force dict containing:
      - F["dc"]:    SymPy Matrix (ncart,1) DC Cartesian force as a function of q,qd (typically q only)
      - F["phasor"]: complex ndarray phasor excitation (nω,ncart,1) or (ncart,1)

    The manager:
      - uses only F["phasor"] for forcing (mapped by R0.T)
      - linearizes only F["dc"] into eq_residual and K_dc, C_dc (via Jacobians of R^T F_dc)
    """
    name: str

    def frequency_domain_MCKF(
        self, omega: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        ...


@dataclass
class FrequencyDomainTotals:
    M: np.ndarray            # (nω,nq,nq) or (nq,nq)
    C: np.ndarray            # (nω,nq,nq) or (nq,nq)
    K: np.ndarray            # (nq,nq)
    Fhat: np.ndarray         # (nω,nq,1) or (nq,1) complex
    eq_residual: np.ndarray  # (nq,1) real


@dataclass
class CompiledOperatingPoint:
    omega0: float
    M: np.ndarray
    C: np.ndarray
    K: np.ndarray
    Fhat: np.ndarray
    eq_residual: np.ndarray
    ramp_T: float


class LinearizationManager:
    """
    Final workflow:
      1) assemble_frequency_domain(ω): builds joint-space M(ω),C(ω),K, Fhat(ω) and eq_residual
         - DC handled via adapter-provided F["dc"] (symbolic) -> contributes to eq_residual, K_dc, C_dc
         - phasor excitation handled via F["phasor"] -> contributes only to forcing term (mapped with R0.T)
      2) compile_operating_point(ω0): caches M0,C0,K0,Fhat0,eq_residual and ramp_T
      3) time integration uses compiled constants:
            M0 qdd + C0 qd + K0 q = eq_residual + ramp(t)*Re{Fhat0 e^{i ω0 t}}
    """

    def __init__(
        self,
        mbd_sys: Any,
        q0: npt.ArrayLike,
        mainNumVars: npt.ArrayLike,
        m0: npt.ArrayLike,
        J0: npt.ArrayLike,
        *,
        print_sym_matrices: bool = False,
    ) -> None:
        self.mbd = mbd_sys

        # Symbols & sizes
        self.q_syms     = list(self.mbd.Q)
        self.qd_syms    = list(self.mbd.QD)
        self.nq         = len(self.q_syms)

        # Initial checks and setup
        self.q0 = np.asarray(q0, dtype=float).reshape(-1)
        if self.q0.size != self.nq:
            raise ValueError(f"q0 must have length nq={self.nq}, got {self.q0.size}")

        self.mainNumVars = np.asarray(mainNumVars, dtype=float).reshape(-1)
        if self.mainNumVars.size != len(self.mbd.mainSymVars):
            raise ValueError(
                f"mainNumVars length {self.mainNumVars.size} must match len(mbd.mainSymVars) "
                f"{len(self.mbd.mainSymVars)}"
            )

        # equilibrium numeric vector (q=q0, qd=0)
        self.eq_vars                                = self.mainNumVars.copy()
        self.eq_vars[: self.nq]                     = self.q0
        self.eq_vars[self.nq : self.nq + self.nq]   = 0.0

        # equilibrium substitution for symbolic eval
        self._subs_base = dict(zip(self.q_syms, self.q0)) | dict(zip(self.qd_syms, [0.0] * self.nq))

        # R0, RD0, NOTE: dxdq0 = R0 at equilibrium
        self.R0     = np.asarray(self.mbd.R_func(*self.eq_vars), dtype=float)
        self.RD0    = np.asarray(self.mbd.RD_func(*self.eq_vars), dtype=float)

        # Linearize MBD once
        self.linMBD = linearize_mbd(self.mbd, self.q0, m0, J0, print_sym_matrices=print_sym_matrices)
        self.M_mbd  = np.asarray(self.linMBD.Mbar(*self.eq_vars), dtype=float)
        self.C_mbd  = np.asarray(self.linMBD.Cbar(*self.eq_vars), dtype=float)
        self.K_mbd  = np.asarray(self.linMBD.Kbar(*self.eq_vars), dtype=float)
        self.F_mbd0 = np.asarray(self.linMBD.Fbar(*self.eq_vars), dtype=float).reshape(self.nq, 1)

        # Adapters
        self.adapters: Dict[str, LinearizationAdapter] = {}

        # Cache for each adapter DC linearization: {name: (Q0_dc, K_dc, C_dc)}
        self._dc_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Compiled operating point
        self.compiled: Optional[CompiledOperatingPoint] = None

    # ---------------- registration ----------------
    def register(self, adapter: LinearizationAdapter) -> None:
        '''Register a linearization adapter. Adapters must have unique names.'''

        if adapter.name in self.adapters:
            raise KeyError(f"Adapter '{adapter.name}' already registered.")
        self.adapters[adapter.name] = adapter
        self._dc_cache.pop(adapter.name, None)


    def _global2joint_frequency(
        self,
        M: np.ndarray,
        C: np.ndarray,
        K: np.ndarray,
        Fhat: np.ndarray,
        *,
        n_omega: int,
        squeeze_if_scalar: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        # normalize omega dimension
        M_w     = broadcast_omega_mat(M, n_omega)
        C_w     = broadcast_omega_mat(C, n_omega)
        Fhat    = as_col(Fhat)
        F_w     = broadcast_omega_vec(Fhat, n_omega) if Fhat.ndim == 2 else broadcast_omega_vec(Fhat, n_omega)

        # K must be omega-independent (2D)
        K = np.asarray(K)
        if K.ndim == 3:
            if K.shape[0] == 1:
                K = K[0]
            else:
                raise ValueError("omega-dependent K not supported")

        R0, RD0 = self.R0, self.RD0

        # Mq(ω) = R0^T M(ω) R0
        Mq  = np.einsum("ab,wbc,cd->wad", R0.T, M_w, R0)

        # Cq(ω) = R0^T ( C(ω) R0 + M(ω) RD0 )
        tmp = np.einsum("wbc,cd->wbd", C_w, R0) + np.einsum("wbc,cd->wbd", M_w, RD0)
        Cq  = np.einsum("ab,wbc->wac", R0.T, tmp)

        # Kq = R0^T K R0
        Kq  = R0.T @ K @ R0

        # Fq(ω) = R0^T Fhat(ω)
        Fq  = np.einsum("ab,wbc->wac", R0.T, F_w)

        if squeeze_if_scalar and n_omega == 1:
            return Mq[0], Cq[0], Kq, Fq[0]
        return Mq, Cq, Kq, Fq

    # ---------------- DC linearization from adapter-provided F["dc"] ----------------
    def _dc_terms_for_adapter(self, name: str, Fdc_sym: sym.Matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        From adapter DC Cartesian force Fdc_sym(q,qd) (ncart,1), compute:
          Q0_dc = (R^T Fdc)|q0
          K_dc  = -∂(R^T Fdc)/∂q |q0
          C_dc  = -∂(R^T Fdc)/∂qd|q0
        Returned in joint coordinates (nq,nq) and (nq,1).
        """
        if name in self._dc_cache:
            return self._dc_cache[name]

        # generalized DC
        Qdc     = self.mbd.R.T * sym.Matrix(Fdc_sym)  # (nq,1) sym

        # Jacobians
        dQdq    = Qdc.jacobian(self.q_syms)
        dQdqd   = Qdc.jacobian(self.qd_syms)

        # stiffness/damping contributions
        Kdc_sym = -dQdq
        Cdc_sym = -dQdqd  # if nqd==0 this will be empty

        # lambdify & eval
        Qdc_func    = sym.lambdify(self.mbd.mainSymVars, Qdc, modules="numpy")
        Kdc_func    = sym.lambdify(self.mbd.mainSymVars, Kdc_sym, modules="numpy")
        Cdc_func    = sym.lambdify(self.mbd.mainSymVars, Cdc_sym, modules="numpy")

        Q0          = np.asarray(Qdc_func(*self.eq_vars), dtype=float).reshape(self.nq, 1)
        Kdc         = np.asarray(Kdc_func(*self.eq_vars), dtype=float).reshape(self.nq, self.nq)
        Cdc         = np.asarray(Cdc_func(*self.eq_vars), dtype=float).reshape(self.nq, self.nq)

        # Store and return
        self._dc_cache[name] = (Q0, Kdc, Cdc)
        return Q0, Kdc, Cdc

    # ---------------- frequency-domain assembly (single branch) ----------------
    def assemble_frequency_domain(self, omega: Union[float, npt.ArrayLike]) -> FrequencyDomainTotals:
        # if omega is a vector then we expand M,C,Fhat along a new leading dimension, but K is always 2D (omega-independent)
        omega_is_scalar = np.isscalar(omega)
        if omega_is_scalar:
            n_omega     = 1
            squeeze     = True
            omega_arg   = float(omega)
        else:
            omega_arg   = np.asarray(omega, dtype=float).reshape(-1)
            n_omega     = int(omega_arg.size)
            squeeze     = False

        # Start with MBD contributions
        if squeeze:
            M_tot       = self.M_mbd.copy()
            C_tot       = self.C_mbd.copy()
            Fhat_tot    = np.zeros((self.nq, 1), dtype=complex)
        else:
            M_tot       = np.repeat(self.M_mbd[None, :, :], n_omega, axis=0)
            C_tot       = np.repeat(self.C_mbd[None, :, :], n_omega, axis=0)
            Fhat_tot    = np.zeros((n_omega, self.nq, 1), dtype=complex)

        K_tot       = self.K_mbd.copy()
        eq_residual = self.F_mbd0.copy()

        # Accumulate adapter contributions
        for name, ad in self.adapters.items():
            M, C, K, F = ad.frequency_domain_MCKF(omega_arg)

            if not isinstance(F, dict) or "dc" not in F or "phasor" not in F:
                raise ValueError(f"Adapter '{name}' must return force dict with keys ['dc','phasor'].")

            Fdc     = F["dc"]
            Fhat    = F["phasor"]

            # DC contributions (eq residual + Kdc + Cdc) ONLY from Fdc
            # NOTE: Fdc is assumed to be constant at equilibrium, if Fdc is function of coordinates modify in definition
            if isinstance(Fdc, sym.MatrixBase) or isinstance(Fdc, np.ndarray):
                Q0_dc, K_dc, C_dc = self._dc_terms_for_adapter(name, Fdc)

                # Add DC contributions to totals:
                eq_residual = eq_residual + Q0_dc
                K_tot       = K_tot + K_dc
                
                # C_dc is allowed (rare); add to C_tot (omega-independent)
                if squeeze:
                    C_tot = C_tot + C_dc
                else:
                    C_tot = C_tot + np.repeat(C_dc[None, :, :], n_omega, axis=0)
            else:
                # If adapter provides no symbolic DC, treat as zero (by convention you always provide it)
                pass

            # Hydrodynamic M,C,K and excitation phasor
            nb      = len(self.mbd.NDOF)
            space   = detect_space(np.asarray(K), nb, self.nq)

            if space == "global":
                Mq, Cq, Kq, Fq = self._global2joint_frequency(
                    np.asarray(M, dtype=float),
                    np.asarray(C, dtype=float),
                    np.asarray(K, dtype=float),
                    np.asarray(Fhat, dtype=complex),
                    n_omega=n_omega,
                    squeeze_if_scalar=squeeze,
                )
            else:
                # joint-space adapter (rare). Normalize dims
                Warning('Remember to add R.T * M * RD terms to C if using joint-space adapter!')
                if squeeze:
                    Mq = np.asarray(M, dtype=float)
                    Cq = np.asarray(C, dtype=float)
                    Kq = np.asarray(K, dtype=float)
                    Fq = np.asarray(Fhat, dtype=complex).reshape(self.nq, 1)
                else:
                    Mq = self._broadcast_omega_mat(np.asarray(M, dtype=float), n_omega)
                    Cq = self._broadcast_omega_mat(np.asarray(C, dtype=float), n_omega)
                    Kq = np.asarray(K, dtype=float)
                    if Kq.ndim == 3:
                        if Kq.shape[0] == 1:
                            Kq = Kq[0]
                        else:
                            raise ValueError(f"Adapter '{name}' returned omega-dependent K in joint-space.")
                    Fq = self._broadcast_omega_vec(np.asarray(Fhat, dtype=complex).reshape(self.nq, 1), n_omega)

            # Sum (phasor contributes only to forcing)
            M_tot = M_tot + Mq
            C_tot = C_tot + Cq
            K_tot = K_tot + Kq
            Fhat_tot = Fhat_tot + Fq

        return FrequencyDomainTotals(M=M_tot, C=C_tot, K=K_tot, Fhat=Fhat_tot, eq_residual=eq_residual)

    # ---------------- compile + time forcing + integration ----------------
    @staticmethod
    def ramp(t: float, T: float) -> float:
        if T <= 0.0:
            return 1.0
        if t <= 0.0:
            return 0.0
        if t >= T:
            return 1.0
        return float(0.5 * (1.0 - np.cos(np.pi * t / T)))

    def compile_operating_point(
        self,
        omega0: float,
        *,
        eq_tol: float = 1e-8,
        eq_mode: str = "zero_if_small",  # "zero_if_small" | "warn" | "raise" | "keep"
        ramp_T: float = 0.0,
    ) -> CompiledOperatingPoint:
        
        # Compute the frequency-domain totals at the operating frequency omega0, including eq_residual and DC contributions from adapters
        totals  = self.assemble_frequency_domain(float(omega0))
        M0      = np.asarray(totals.M, dtype=float).reshape(self.nq, self.nq)
        C0      = np.asarray(totals.C, dtype=float).reshape(self.nq, self.nq)
        K0      = np.asarray(totals.K, dtype=float).reshape(self.nq, self.nq)
        Fhat0   = np.asarray(totals.Fhat, dtype=complex).reshape(self.nq, 1)
        req     = np.asarray(totals.eq_residual, dtype=float).reshape(self.nq, 1)

        # If residual is nonzero, we have an equilibrium mismatch. Handle according to eq_mode:
        rnorm = float(np.linalg.norm(req))
        if eq_mode == "zero_if_small":
            if rnorm < eq_tol:
                req[:] = 0.0
        elif eq_mode == "warn":
            if rnorm >= eq_tol:
                print(f"[LinearizationManager] WARNING: ||eq_residual||={rnorm:.3e} >= {eq_tol:.3e}")
        elif eq_mode == "raise":
            if rnorm >= eq_tol:
                raise ValueError(f"Equilibrium residual ||r_eq||={rnorm:.3e} >= {eq_tol:.3e}")
        elif eq_mode == "keep":
            pass
        else:
            raise ValueError(f"Unknown eq_mode='{eq_mode}'")

        self.compiled = CompiledOperatingPoint(
            omega0=float(omega0), M=M0, C=C0, K=K0, Fhat=Fhat0, eq_residual=req, ramp_T=float(ramp_T)
        )
        return self.compiled

    def has_compiled(self) -> bool:
        return self.compiled is not None

    def force_time(self, t: float) -> np.ndarray:
        '''Evaluate the time-domain forcing at time t using the compiled operating point and the ramp function.'''
        if self.compiled is None:
            raise RuntimeError("Call compile_operating_point(...) before evaluating time forcing.")
        
        r       = self.ramp(float(t), self.compiled.ramp_T)
        exp_it  = np.exp(-1j * self.compiled.omega0 * float(t))
        f_osc   = np.real(self.compiled.Fhat * exp_it)

        return (self.compiled.eq_residual + r * f_osc).reshape(self.nq, 1)

    def integrate_linear_system(
        self,
        *,
        tspan: float,
        dt: Optional[float] = None,
        method: str = "RK45",
        omega0: Optional[float] = None,
        eq_tol: float = 1e-8,
        eq_mode: str = "zero_if_small",
        **solver_kwargs,
    ):
        # Check inputs correctness
        if self.compiled is None:
            if omega0 is None:
                raise RuntimeError("Provide omega0 or call compile_operating_point(...) before integrating.")
            self.compile_operating_point(float(omega0), ramp_T=self.ramp_T, eq_tol=eq_tol, eq_mode=eq_mode)

        if self.mainNumVars[self.nq : self.nq + self.nq].any() != 0.0:
            raise ValueError("Initial conditions must have zero velocity (mainNumVars[nq:nq+nq] == 0).")

        # Define the initial conditions from self.mainNumVars
        q0v     = self.mainNumVars[:self.nq].copy()
        qd0v    = np.zeros(self.nq)
        y0      = np.hstack((q0v, qd0v))

        # Time handling
        t0, tf = float(0.), float(tspan)
        t_eval = None

        if tf <= t0:
            raise ValueError("tspan must satisfy tf > t0")
        if dt is not None and dt > 0:
            t_eval = np.arange(t0, tf + dt, dt)

        # Compiled path does not depend on t_update, but keep signature compatible
        t_update = getattr(self.mbd, "t_update", [])

        # Define solver tolerances
        rtol = solver_kwargs.pop("rtol", 1e-6)
        atol = solver_kwargs.pop("atol", 1e-9)

        # Define the rhs function for solve_ivp using the linear integrator
        rhs = lambda t, y: linear_integrator(self, t, y, self.eq_vars, t_update=t_update)

        # Solve and time the integration
        t0_int  = time()
        sol     = solve_ivp(rhs, (t0, tf), y0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)
        t_end   = time() - t0_int
        print(f"Integration completed in {t_end:.3f} seconds.")

        return sol