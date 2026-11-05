# source/multibody/flexible/_flex_distribution_helper.py
"""
@author: Sahand Sabet

Symbolic property-distribution support for FSM's Table 2 (flex_E, flex_I,
flex_A, flex_rho): lets any of those four fields be a `sympy` expression in
ONE shared symbol (the member's local axial coordinate, 0 at the root, L at
the tip) instead of today's scalar-or-length-n_seg-list, for members whose
material/section properties genuinely vary continuously along their length,
rather than being piecewise-constant per segment. A tapered beam/bar is one
common example of this, but the mechanism itself is general -- ANY function
of position is accepted, tapering (monotonic) or not.

Opt-in, per Table-2 row, via a new parallel list `flex_x_dist`: `None` (the
default -- today's behavior, completely unchanged) or a single `sympy.Symbol`
declaring which symbol that row's distributed properties use. See
`multibody.checks.validate_row` for the full contract.

Physics note (the reason this module exists instead of just averaging E/I
per segment): the quantity that is physically additive along a member's
length is COMPLIANCE density, `1/(E(x)*I(x))` (bending) or `1/(E(x)*A(x))`
(axial) -- not E or I separately. Averaging E and I independently and then
plugging into the constant-property formula `k = EI/dx` is a DIFFERENT,
biased quantity (by Jensen's inequality, since 1/u is convex, it always
under-estimates the true compliance, i.e. over-estimates stiffness). So
every stiffness-relevant pair is combined into ONE expression first
(`E(x)*I(x)` or `E(x)*A(x)`), and that combined expression's reciprocal is
integrated directly, per HALF-CELL (not per whole segment -- the two halves
of a segment with a non-constant distribution are generally not equal). The same logic applies to
mass: `rho(x)*A(x)` is integrated as one combined product, not `rho`/`A`
averaged separately and multiplied after.

All integration here is purely NUMERIC (`scipy.integrate.quad` on a
`sympy.lambdify`-compiled function) -- no symbolic antiderivative is ever
attempted. This sidesteps the closed-form-vs-no-closed-form branch entirely,
at the (negligible, for smooth integrands) cost of exactness compared to a
closed-form symbolic result.

Validation (contract checks, positivity, quadrature convergence) lives in
`source/multibody/checks/flex_distribution_check.py`, not here -- this
module only ever computes physical quantities, it never validates its own
inputs/outputs.

Known, intentional limitation (not fixed here -- see userDocumentation.md):
a non-uniform segment's TRUE mass centroid can drift slightly from its
assumed geometric midpoint. `resolve_mass_and_inertia`'s J is computed
rigorously about the geometric midpoint (consistent with how
`parent_cg_to_joint`/`joint_to_child_cg` already place every segment's CG
today), not about that true, slightly-offset centroid. Fully correcting
that would require shifting the segment's own kinematic placement, which
cascades into `_mounting_offset`/`dx_vec` -- out of scope for this module.
"""
import numpy as np
import sympy as sym
from scipy.integrate import quad

from ..checks.flex_distribution_check import check_strictly_positive, check_quadrature_converged


def _as_expr(value):
    """Promote a plain scalar to a trivial sympy expression; pass sympy expressions through."""
    return value if isinstance(value, sym.Basic) else sym.Float(value)


def _quad(f, lo, hi, row_label, label):
    """scipy.integrate.quad, then hand its own (value, err) pair to the
    checks module to judge convergence -- the integration itself stays here."""
    value, err = quad(f, lo, hi)
    check_quadrature_converged(value, err, row_label, label)
    return float(value)


def resolve_compliance_half_cells(E_val, prop_val, x_dist_sym, n_seg, dx, row_label, prop_name):
    """
    Integrate the combined compliance density `1/(E(x)*prop(x))` over each
    of the 2*n_seg half-cells (prop_val = flex_I for bending, flex_A for
    axial). Only ever called for a row where x_dist_sym is not None (the
    caller is responsible for branching to the existing constant-property
    formula otherwise).

    Returns
    -------
    (left, right) : two length-n_seg lists of half-cell compliances (plain
    floats). left[i]/right[i] are segment i's own root-side/tip-side half,
    each spanning dx/2.
    """
    combined = _as_expr(E_val) * _as_expr(prop_val)
    L_total = n_seg * dx
    # Validate the DENOMINATOR (E*prop) stays strictly positive BEFORE ever
    # forming its reciprocal -- checking 1/combined directly would hit a
    # genuine division-by-zero/inf at a degenerate point, surfacing as a
    # confusing "not finite" error instead of the clear "must stay strictly
    # positive" message this is meant to give.
    check_strictly_positive(combined, x_dist_sym, 0.0, L_total, f"E*{prop_name}", row_label)
    f = sym.lambdify(x_dist_sym, 1 / combined, "numpy")

    left, right = [], []
    for i in range(n_seg):
        x0, x_mid, x1 = i * dx, i * dx + dx / 2.0, (i + 1) * dx
        left.append(_quad(f, x0, x_mid, row_label, f"compliance (segment {i + 1}, root-side half)"))
        right.append(_quad(f, x_mid, x1, row_label, f"compliance (segment {i + 1}, tip-side half)"))
    return left, right


def resolve_mass_and_inertia(rho_val, A_val, x_dist_sym, n_seg, dx, row_label):
    """
    Integrate the combined mass density `rho(x)*A(x)` over each full
    segment for mass, and the second central moment (about that segment's
    own GEOMETRIC midpoint -- see module docstring's "known limitation")
    for inertia. Only ever called when x_dist_sym is not None.

    Returns
    -------
    (m_list, J_list) : two length-n_seg lists of plain floats.
    """
    density_expr = _as_expr(rho_val) * _as_expr(A_val)
    L_total = n_seg * dx
    check_strictly_positive(density_expr, x_dist_sym, 0.0, L_total, "rho*A", row_label)
    density_f = sym.lambdify(x_dist_sym, density_expr, "numpy")

    m_list, J_list = [], []
    for i in range(n_seg):
        x0, x1 = i * dx, (i + 1) * dx
        x_mid = 0.5 * (x0 + x1)
        m_i = _quad(density_f, x0, x1, row_label, f"mass (segment {i + 1})")
        if m_i <= 0.0:
            raise ValueError(f"row {row_label}: segment {i + 1} integrated to non-positive mass ({m_i:.6g}).")

        def _inertia_integrand(xx, _f=density_f, _xm=x_mid):
            return (xx - _xm) ** 2 * _f(xx)

        J_i = _quad(_inertia_integrand, x0, x1, row_label, f"inertia (segment {i + 1})")
        if J_i < -1e-12:
            raise ValueError(f"row {row_label}: segment {i + 1} integrated to negative inertia ({J_i:.6g}).")
        m_list.append(m_i)
        J_list.append(max(J_i, 0.0))
    return m_list, J_list


def resolve_display_samples(value, x_dist_sym, n_seg, dx):
    """
    Midpoint-sampled, DISPLAY-ONLY representative value per segment, for
    Table 2's printed/echoed form -- NEVER used for the actual physics
    (which always integrates the combined expression directly, never a
    standalone per-property value). Only meaningful when `value` is a
    sympy expression; a plain scalar is returned unchanged (not broadcast
    into a list -- callers that need a length-n_seg list handle that
    themselves, same as today's convention for an ordinary, non-distributed property).
    """
    if not isinstance(value, sym.Basic):
        return value
    f = sym.lambdify(x_dist_sym, value, "numpy")
    return [float(f((i + 0.5) * dx)) for i in range(n_seg)]
