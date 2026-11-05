# source/multibody/checks/flex_distribution_check.py
"""
@author: Sahand Sabet

Validation checks for FSM's Table 2 continuous-property-distribution
feature (flex_x_dist): the structural/contract check on a row's declared
symbol (`validate_row`), the physical sanity check that a stiffness- or
mass-relevant combined expression stays strictly positive and finite across
a member's length (`check_strictly_positive`), and the numerical-quality
check that a `scipy.integrate.quad` result actually converged
(`check_quadrature_converged`).

The actual numeric integration itself stays in
`source/multibody/flexible/_flex_distribution_helper.py` -- this module
only ever validates inputs/outputs, it never computes a physical quantity
(no `scipy.integrate.quad` call lives here).
"""
import numpy as np
import sympy as sym

_N_POSITIVITY_SAMPLES = 9   # sample points across a row's full [0, L] used to validate a
                            # stiffness/mass-relevant combined product stays strictly positive
_QUAD_REL_ERR_TOL = 1e-6    # quad's own error estimate, relative to the integral's magnitude,
                            # above which the result is distrusted rather than silently accepted


def validate_row(x_dist_sym, properties, row_label):
    """
    Enforce the flex_x_dist contract for one Table 2 row. `properties` is a
    dict of name -> value, scoped to ONLY the properties that row's own
    deformation_mode actually uses (bending: E, I, rho, A; axial: E, A, rho
    -- I is never read for an axial row by _expand_axial_row, so it is never
    validated either, even if it happens to hold a sympy expression).

    Contract (exactly these cases, nothing looser):
      - x_dist_sym is None: NONE of `properties` may be a sympy expression.
      - x_dist_sym is a sympy.Symbol: every sympy-expression property must
        use EXACTLY that symbol (not a different one); none may be a
        per-segment list (mixing a list with a symbolic distribution is not
        supported -- every property on a row with a distribution is scalar or
        symbolic, never a list); and AT LEAST ONE property must actually use the
        declared symbol (a declared-but-unused symbol is rejected as a
        near-certain mistake).
      - x_dist_sym is anything else (number, string, list, bool, a sympy
        expression that isn't a plain Symbol, ...): rejected immediately.

    Raises ValueError on any violation. Returns nothing.
    """
    if x_dist_sym is None:
        offenders = [name for name, v in properties.items() if isinstance(v, sym.Basic)]
        if offenders:
            raise ValueError(
                f"row {row_label}: {offenders} contain a sympy expression, but "
                f"flex_x_dist is None for this row -- set flex_x_dist[row] to "
                f"the symbol those expressions use."
            )
        return

    if not isinstance(x_dist_sym, sym.Symbol):
        raise ValueError(
            f"row {row_label}: flex_x_dist must be None or a single "
            f"sympy.Symbol, got {x_dist_sym!r}."
        )

    symbol_used = False
    for name, value in properties.items():
        if isinstance(value, (list, tuple)):
            raise ValueError(
                f"row {row_label}: has a distribution (flex_x_dist={x_dist_sym}), but "
                f"{name!r} is a per-segment list -- mixing a per-segment list "
                f"with a symbolic distribution is not supported. Use a sympy "
                f"expression in {x_dist_sym}, or a constant scalar, instead."
            )
        if isinstance(value, sym.Basic):
            extra_symbols = value.free_symbols - {x_dist_sym}
            if extra_symbols:
                raise ValueError(
                    f"row {row_label}: {name!r} uses symbol(s) {extra_symbols}, "
                    f"but this row's declared flex_x_dist is {x_dist_sym} -- "
                    f"every distributed property on one row must use the SAME "
                    f"single symbol."
                )
            if value.free_symbols:
                symbol_used = True

    if not symbol_used:
        raise ValueError(
            f"row {row_label}: flex_x_dist={x_dist_sym} is declared, but no "
            f"property actually depends on it -- every property resolved to "
            f"a plain constant. Set flex_x_dist[row] = None instead, or make "
            f"at least one property a real function of {x_dist_sym}."
        )


def check_strictly_positive(expr, x_dist_sym, lo, hi, label, row_label):
    """
    Sample a combined expression at _N_POSITIVITY_SAMPLES points across
    [lo, hi] and require every value to be finite, real, and strictly
    positive -- quadrature itself will not error on a sign change or a
    near-zero dip, it will just silently hand back a number that looks
    plausible but is physically wrong (infinite/negative compliance, or
    negative mass), so this check has to be explicit.

    Raises ValueError on any violation. Returns nothing -- callers lambdify
    the expression themselves for the actual integration (this function
    never hands back a reusable callable, so it never computes physics).
    """
    f = sym.lambdify(x_dist_sym, expr, "numpy")
    samples = np.linspace(lo, hi, _N_POSITIVITY_SAMPLES)
    values = np.asarray(f(samples), dtype=complex)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"row {row_label}: {label} is not finite over [{lo:.6g}, {hi:.6g}].")
    if np.any(np.abs(values.imag) > 1e-12):
        raise ValueError(f"row {row_label}: {label} is not real-valued over [{lo:.6g}, {hi:.6g}].")
    if np.any(values.real <= 0):
        raise ValueError(
            f"row {row_label}: {label} must stay strictly positive over "
            f"[{lo:.6g}, {hi:.6g}] -- found a non-positive value (a degenerate "
            f"distribution -- e.g. a cross-section shrinking to zero -- or a sign error)."
        )


def check_quadrature_converged(value, err, row_label, label):
    """
    Reject a `scipy.integrate.quad` result whose own error estimate is large
    relative to its magnitude, rather than silently trusting it. Takes the
    `(value, err)` pair `quad` itself already returned -- the actual
    integration stays with the caller, this only judges its quality.
    """
    if err > _QUAD_REL_ERR_TOL * max(abs(value), 1e-300):
        raise ValueError(
            f"row {row_label}: numerical integration of {label} did not "
            f"converge cleanly (value={value:.6g}, estimated error={err:.3g}) "
            f"-- check the expression for near-singular or highly oscillatory behavior."
        )
