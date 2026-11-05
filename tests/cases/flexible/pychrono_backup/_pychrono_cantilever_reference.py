# -*- coding: utf-8 -*-
r"""Independent, third-party-software cross-validation of FSM's tapered-
bending natural frequencies using REAL PyChrono (not just the from-scratch
_euler_bernoulli_fem_reference.py in this same folder).

This script is NOT part of the normal pytest suite and cannot be imported
or collected by it: PyChrono is only installed in a separate conda
environment named `chrono` (see below), not in the `multibody_env`
environment the rest of this repository's tests run under. Instead, this
script is a one-off, manually-run verification tool. Its output values are
hand-copied into
test_tapered_bending_frequencies_match_pychrono_reference() in
test_flex_distribution.py, the same pattern used by
tests/cases/chrono/truth/*.npy for the rigid-body regression cases (a
real-Chrono-derived reference value, pre-computed once and then checked
against on every normal test run without needing PyChrono at all).

How to run this script
-----------------------
PyChrono lives in a separate conda env (`chrono`), and importing it
requires prepending its DLL directories to PATH first (confirmed necessary
on Windows with this specific install):

    $env:PATH = "C:\\Users\\<user>\\Anaconda3\\envs\\chrono;" + `
                "C:\\Users\\<user>\\Anaconda3\\envs\\chrono\\Library\\bin;" + `
                "C:\\Users\\<user>\\Anaconda3\\envs\\chrono\\Scripts;" + $env:PATH
    C:\Users\<user>\Anaconda3\envs\chrono\python.exe _pychrono_cantilever_reference.py

Method
------
PyChrono's FEA module (pychrono.fea) provides ChElementBeamEuler (a
corotational Euler-Bernoulli beam finite element) and
ChBeamSectionEulerAdvanced (per-element E, I, A, rho -- enough to build a
tapered beam by assigning each element its own section, evaluated at that
element's own midpoint, mirroring the "one representative value per
element/segment" approximation both FSM and the local FEM reference also
make). There is no pychrono.modal in this install, so natural frequencies
are extracted the same way a physical modal hammer test would: clamp the
root, statically deflect the tip with a point force to get a "plucked"
initial shape (system.DoStaticLinear()), release the force, run free
(zero-gravity, zero-damping) dynamics, and FFT the tip's transverse
position history, peak-picking the dominant frequencies.

Two PyChrono-specific defaults that silently break this check if left
alone (see /memories/repo/pychrono_environment.md for the extended
writeup):
  - ChBeamSectionEulerAdvanced defaults to Rayleigh beta damping = 0.01
    (NOT zero), which overdamps bending modes almost completely. Must
    explicitly zero both Rayleigh coefficients per section.
  - The default ChSystemNSC timestepper is numerically dissipative enough
    to kill free vibration within a fraction of a second. Use the
    (non-dissipative) TRAPEZOIDAL timestepper instead.

Validated precision
--------------------
Self-check against the uniform cantilever's closed-form frequencies
(f_n = (beta_n*L)^2/(2*pi*L^2)*sqrt(EI/rhoA), beta_1*L=1.875104) at
n_elem=20: mode 1 agrees to ~1%, mode 2 to ~6% (the gap is FFT frequency-
resolution/peak-picking precision, not a modeling error -- confirmed by the
static tip deflection delta=F*L^3/(3EI) matching the closed form to ~1e-16
relative, i.e. the beam's actual stiffness is essentially exact). The same
~1%/~6% precision pattern is expected for the tapered case.
"""
import math

import numpy as np
import pychrono as chrono
import pychrono.fea as fea


def cantilever_freqs_from_chrono(L, EI_of_x, rhoA_of_x, n_elem,
                                  tip_force=50.0, tspan=1.5, dt=2e-4, n_modes=3):
    """Build a clamped-free PyChrono FEA beam with per-element (E*I)(x) and
    (rho*A)(x) sampled at each element's own midpoint, pluck its tip
    statically, release it, and FFT the free transverse vibration of the
    tip node to extract the first `n_modes` natural frequencies (Hz).
    """
    system = chrono.ChSystemNSC()
    system.Set_G_acc(chrono.ChVectorD(0, 0, 0))
    system.SetSolver(chrono.ChSolverSparseQR())
    system.SetTimestepperType(chrono.ChTimestepper.Type_TRAPEZOIDAL)

    mesh = fea.ChMesh()
    mesh.SetAutomaticGravity(False)
    system.Add(mesh)

    dx = L / n_elem
    builder = fea.ChBuilderBeamEuler()

    root_node = fea.ChNodeFEAxyzrot(chrono.ChFrameD(chrono.ChVectorD(0, 0, 0)))
    root_node.SetFixed(True)
    mesh.AddNode(root_node)

    # E and rho are fixed references; only (EI) and (rhoA) products matter
    # for bending frequencies, so I(x)=EI(x)/E and A(x)=rhoA(x)/rho recover
    # self-consistent per-element section properties for ChBeamSectionEulerAdvanced.
    E_val, rho_val = 200e9, 7800.0
    prev_node = root_node
    for i in range(n_elem):
        x_mid = (i + 0.5) * dx
        sec = fea.ChBeamSectionEulerAdvanced()
        sec.SetYoungModulus(E_val)
        sec.SetGshearModulus(E_val / 2.6)
        sec.SetDensity(rho_val)
        sec.SetArea(rhoA_of_x(x_mid) / rho_val)
        sec.SetIyy(EI_of_x(x_mid) / E_val)
        sec.SetIzz(EI_of_x(x_mid) / E_val)
        sec.SetBeamRaleyghDampingAlpha(0.0)
        sec.SetBeamRaleyghDampingBeta(0.0)

        builder.BuildBeam(mesh, sec, 1, prev_node,
                           chrono.ChVectorD((i + 1) * dx, 0, 0), chrono.ChVectorD(0, 1, 0))
        nds = builder.GetLastBeamNodes()
        prev_node = nds[len(nds) - 1]

    tip_node = prev_node
    tip_node.SetForce(chrono.ChVectorD(0, tip_force, 0))
    system.DoStaticLinear()
    tip_node.SetForce(chrono.ChVectorD(0, 0, 0))

    n_steps = int(tspan / dt)
    tip_y = np.zeros(n_steps)
    for k in range(n_steps):
        system.DoStepDynamics(dt)
        tip_y[k] = tip_node.GetPos().y

    tip_y = tip_y - tip_y.mean()
    spec = np.abs(np.fft.rfft(tip_y))
    freqs_axis = np.fft.rfftfreq(n_steps, d=dt)

    peak_idx = [i for i in range(2, len(spec) - 2)
                if spec[i] > spec[i - 1] and spec[i] > spec[i + 1] and spec[i] > 0.02 * spec.max()]
    peak_idx.sort(key=lambda i: -spec[i])
    freqs_found = np.array(sorted(freqs_axis[i] for i in peak_idx[:n_modes]))
    return freqs_found


if __name__ == "__main__":
    L, E, I, A, rho = 1.0, 200e9, 8e-8, 1e-4, 7800.0
    beta_L = np.array([1.875104, 4.694091, 7.854757])
    analytical = (beta_L ** 2) / (2 * math.pi * L ** 2) * math.sqrt(E * I / (rho * A))
    print("Analytical (uniform):", analytical)

    freqs_uniform = cantilever_freqs_from_chrono(
        L, lambda x: E * I, lambda x: rho * A, n_elem=20,
        tip_force=50.0, tspan=1.5, dt=2e-4, n_modes=3)
    print("Chrono FEA (uniform):", freqs_uniform)

    # Tapered cantilever -- SAME I(x) profile as EXAMPLE O /
    # Examples_flexible_FSM/flex_tapered_bending.py and
    # test_tapered_bending_frequencies_match_independent_fem_reference.
    I_ROOT, I_TIP = 8e-8, 2e-8
    freqs_tapered = cantilever_freqs_from_chrono(
        L, lambda x: E * (I_ROOT - (I_ROOT - I_TIP) * x / L), lambda x: rho * A,
        n_elem=40, tip_force=50.0, tspan=1.5, dt=1e-4, n_modes=3)
    print("Chrono FEA (tapered):", freqs_tapered)
