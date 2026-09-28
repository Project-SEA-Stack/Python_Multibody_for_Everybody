# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 09:22:57 2025

@author: adiazfl
"""

import numpy as np
import matplotlib.patches as patches

def _spr_init(R: float, n: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a base (un‑scaled, un‑rotated) spring profile.
    Returns (x, y, length).
    """
    r       = 0.7 * R
    step    = 2 * (R - r)
    ang_R   = np.linspace(3*np.pi/2, np.pi/2, 20)
    ang_r   = np.linspace(np.pi/2, -np.pi/2, 20)

    x       = np.empty(0)
    y       = np.empty(0)
    c_R     = 0.0

    for i in range(n):
        c_R += step
        R_x = R * np.cos(ang_R)
        R_y = c_R + R * np.sin(ang_R)
        c_r = c_R + R - r
        r_x = R * np.cos(ang_r)
        r_y = c_r + r * np.sin(ang_r)

        x = np.hstack((x, R_x, r_x))
        y = np.hstack((y, R_y, r_y))

        if i == n-1:
            c_R += step
            R_x = R * np.cos(ang_R)
            R_y = c_R + R * np.sin(ang_R)

            # add end‑straight segments and shift so y[0]==0
            x = np.hstack(([0.0], x, R_x, [0.0]))
            y = np.hstack((y[0] - R, y, R_y, y[-1] + 3*R))
            y -= np.min(y)
            break

    length = abs(y[-1] - y[0])
    return x, y, length


def spring_coords(
    pt1: tuple[float,float],
    pt2: tuple[float,float],
    R: float,
    num_coils: int
) -> tuple[np.ndarray,np.ndarray]:
    """
    Get the (x,y) coordinates of a 2D spring from pt1→pt2.

    pt1, pt2 : (x,y) endpoints
    R        : base spring ``radius``
    num_coils: number of coils
    """
    p1 = np.array(pt1, dtype=float)
    p2 = np.array(pt2, dtype=float)

    # 1) generate base spring
    spr0_x, spr0_y, spr0_len = _spr_init(R, num_coils)

    # 2) scale
    dist  = np.linalg.norm(p2 - p1)
    scale = dist / spr0_len
    max_scale = 2.0

    if scale > max_scale:
        u      = spr0_len
        e      = (p2 - p1) / dist
        pt_aux = 0.5 * u * e * (scale - max_scale)
        x_bs   = spr0_x
        y_bs   = spr0_y * 2.0
    else:
        pt_aux = np.zeros(2)
        x_bs   = spr0_x
        y_bs   = spr0_y * scale

    # 3) to polar
    theta = np.arctan2(y_bs, x_bs)
    r     = np.hypot(x_bs, y_bs)

    # 4) rotate to align with p1→p2
    alpha       = np.pi/2 - np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
    theta_rot   = theta - alpha
    x_rot       = r * np.cos(theta_rot)
    y_rot       = r * np.sin(theta_rot)

    # 5) translate and stitch on the endpoints
    x_final = x_rot + p1[0] + pt_aux[0]
    y_final = y_rot + p1[1] + pt_aux[1]

    x_full = np.hstack((p1[0], x_final, p2[0]))
    y_full = np.hstack((p1[1], y_final, p2[1]))

    return x_full, y_full


def damper_plot(pt1, pt2, ax, color='m', linewidth=1.2):
    """
    Draw a little “box‐and‐line” damper between pt1 and pt2 on the given Axes.
    """
    p1 = np.array(pt1, float)
    p2 = np.array(pt2, float)
    module   = np.linalg.norm(p2 - p1)
    if module == 0:
        return

    direction = (p2 - p1) / module
    dir_perp  = np.array([ direction[1], -direction[0] ])

    # box fractions & sizes (same as your MATLAB L1…L5)
    L1, L2, L3, L4, L5 = 0.48, 0.5, 2*0.1, 2*0.05, 0.05

    P3  = p1 + L1 * direction * module
    P4  = p1 + L2 * direction * module
    P5  = P3 - L3 * dir_perp
    P6  = P3 + L3 * dir_perp
    P7  = P4 - L4 * dir_perp
    P8  = P4 + L4 * dir_perp
    P9  = P5 + L5 * direction
    P10 = P6 + L5 * direction

    segments = [
        (p1,  P3),
        (P4,  p2),
        (P5,  P6),
        (P7,  P8),
        (P5,  P9),
        (P6,  P10),
    ]

    for i, (A, B) in enumerate(segments):
        xs, ys = [A[0], B[0]], [A[1], B[1]]
        ax.plot(xs, ys,
                color=color,
                linewidth=linewidth,
                label='Damper' if i == 0 else None)
        
    

def torsion_spring(joint, R, ax, color='k', linewidth=1.2):
    """
    Draw a little circular “box” at `joint` of radius R on Axes `ax`,
    exactly like MATLAB’s rectangle(..., 'Curvature',[1,1]).
    """
    circ = patches.Circle(
        (joint[0], joint[1]),
        radius=R,
        fill=False,
        edgecolor=color,
        linewidth=linewidth
    )
    ax.add_patch(circ)

def torsion_damper(joint, R, ax, color='m', linewidth=1.2):
    """
    Draw a “damper” symbol at `joint` of radius R:
      1) a circle (like torsion spring)
      2) four little lines offset +/- angles of 10° around the circle
    """
    x0, y0 = joint
    # 1) circle
    circ = patches.Circle((x0, y0), radius=R,
                         fill=False, edgecolor=color, linewidth=linewidth)
    ax.add_patch(circ)

    # 2) damper box lines
    L3, L4, L5 = 0.1, 0.05, 0.02
    direction = np.array([0.0, 1.0])
    dir_perp  = np.array([1.0, 0.0])

    # two anchor points on circle at ±10°
    a1 = np.deg2rad(-10)
    a2 = np.deg2rad( 10)
    P3 = np.array([x0, y0]) + R * np.array([np.cos(a1), np.sin(a1)])
    P4 = np.array([x0, y0]) + R * np.array([np.cos(a2), np.sin(a2)])

    # big vertical line at P3
    P5 = P3 - L3 * dir_perp
    P6 = P3 + L3 * dir_perp
    # small vertical line at P4
    P7 = P4 - L4 * dir_perp
    P8 = P4 + L4 * dir_perp
    # tiny horizontal caps
    P9  = P5 + L5 * direction
    P10 = P6 + L5 * direction

    for i, (A, B) in enumerate([(P5,P6), (P7,P8), (P5,P9), (P6,P10)]):
        ax.plot([A[0], B[0]], [A[1], B[1]], color=color,
                linewidth=linewidth,
                label='Torsion Damper' if i == 0 else None)

def resolve_point(code: str, CG, BD, GR):
    """tiny helper to turn 'BD12_3'/'GR0_3'/'CG1_1' (required delimited format)
    into (x,y)."""
    pt = code[:2]
    rest = code[2:]
    if "_" not in rest:
        raise ValueError(
            f"'{code}' uses the legacy point-string format. "
            f"Use the delimited form instead, e.g. 'BD12_3' (body=12, point=3)."
        )
    body_str, idx_str = rest.split("_")
    body, idx = int(body_str), int(idx_str)
    if pt == "CG":
        return tuple(CG[body-1])
    if pt == "BD":
        arr = BD[body].reshape(-1,2)
        return tuple(arr[idx])
    if pt == "GR":
        return tuple(GR[idx])
    return (np.nan, np.nan)