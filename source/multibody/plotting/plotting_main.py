# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 15:12:39 2025

@author: adiazfl
"""

import numpy as np
import matplotlib.pyplot as plt
from .plotting_helper import *
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
# from Plotting.Bound_finder import bound_finder
# from pathlib import Path
from functools import partial
import sympy as sym

__all__ = [
    "plot_multibody_system",
]


def plt_template(ax: plt.Axes, dark: bool = False) -> None:
    """
    Apply light or dark styling to a Matplotlib Axes to match MATLAB’s PlotUtil.

    Adjusts background color, tick parameters, title and axis label colors,
    grid appearance, and the default color cycle.

    Parameters
    ----------
    ax : plt.Axes
        The target Axes to style.
    dark : bool, optional
        If True, use dark‐mode colors; otherwise use light‐mode. Default is False.
    """
    fontsize = 14

    light_fg = "#141414"
    light_bg = "#F2F2F2"
    dark_fg  = "#F2F2F2"
    dark_bg  = "#141414"

    if dark:
        fg, bg = dark_fg, dark_bg
        colour_cycle = ["#FA6A6A", "#FFBF00", "#7EF9FF"]
    else:
        fg, bg = light_fg, light_bg
        colour_cycle = ["#B5852A", "#009878", "#6439A0"]

    ax.set_facecolor(bg)
    ax.tick_params(colors=fg, direction="in")
    ax.title.set_color(fg)
    ax.title.set_fontsize(fontsize+2)
    ax.xaxis.label.set_color(fg)
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_color(fg)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.grid(True, color=fg, alpha=0.5, linestyle=":", linewidth=0.7)
    ax.set_prop_cycle(color=colour_cycle)
    ax.figure.set_facecolor(bg)
    # ax.legend(bbox_to_anchor=(1,1), fontsize=fontsize, frameon=False)
    

def plot_multibody_system(
    main_num_vars: np.ndarray,
    MBD,
    *,
    xfm = None,
    t_val: float | None = None,
    frame_bounds: tuple[float, float, float, float] | None = None,
    dark_mode: bool = False,
    ax: plt.Axes | None = None,
    new_figure: bool = True
) -> tuple[plt.Figure, plt.Axes]:
    """
    Draw the current configuration of a planar multibody system.

    Renders bodies’ centers of gravity, joints, body‐defined points,
    ground points, forces (CG forces, point forces), springs, dampers,
    and optional mooring lines—all on a single static plot.

    Parameters
    ----------
    main_num_vars : np.ndarray
        Numeric array of values for every symbolic variable in `MBD.mainSymVars`.
        If `MBD.t_update` is not None and `t_val` is provided, the corresponding
        entry in this array is overwritten with `t_val`.
    MBD : object
        A multibody‐dynamics container with attributes:

        - `mainSymVars`, `CGpoints_func`, `BDpoints_func`, `GRpoints_func`,
          `JOpoints_func`
        - `Force` (dict of force definitions)
        - `t_update` (index of time variable or None)
        
    xfm : optional
        Transformation/adapter for external libraries (e.g. MoorPy) used when plotting
        mooring lines.
    t_val : float or None, optional
        Current time in seconds. If provided, used to update the time symbol
        in `main_num_vars` and shown in the plot title.
    frame_bounds : tuple of 4 floats or None, optional
        Axis limits as `(xmin, xmax, ymin, ymax)`. If None, limits are computed
        from the union of all plotted points with a 10% padding.
    dark_mode : bool, optional
        If True, applies dark‐mode styling. Default is False.
    ax : plt.Axes or None, optional
        The Axes to draw on. If None or `new_figure` is True, a new Figure/Axes
        pair is created.
    new_figure : bool, optional
        If True, always create a fresh Figure/Axes even if `ax` is provided.
        Default is True.

    Returns
    -------
    fig : plt.Figure
        The Figure containing the multibody plot.
    ax : plt.Axes
        The Axes with all drawn elements.
    """
    # flag check to initialize plotting for external libraries
    if xfm is not None and xfm.xfmFlag == 0:
        xfm.init_plot()
        xfm.xfmFlag = 1
        
    # Unpack MBD:
    main_sym_vars = MBD.mainSymVars
    CG_fun = MBD.CGpoints_func
    BD_fun = MBD.BDpoints_func
    GR_fun = MBD.GRpoints_func
    JO_fun = MBD.JOpoints_func
    Force_dict = MBD.Force
    
    # 1) time update
    if t_val is not None and MBD.t_update:
        main_num_vars               = main_num_vars.copy()
        main_num_vars[MBD.t_update] = t_val

    # 2) build a substitution map for *all* your symbols
    subs_map = { sym: main_num_vars[i]
                 for i, sym in enumerate(main_sym_vars) }

    def to_float(expr):
        # replace every symbol in expr with its current numeric value
        return float(expr.subs(subs_map)) if isinstance(expr, sym.Expr) else float(expr)

    # 2) evaluate
    CG      = np.asarray(CG_fun(*main_num_vars), float)
    JO      = np.asarray(JO_fun(*main_num_vars), float)
    raw_gr  = GR_fun(*main_num_vars)
    GR      = np.vstack(raw_gr).astype(float) if raw_gr else np.empty((0, 2))   
    BD      = {k: np.asarray(v(*main_num_vars), float)
                   for k, v in BD_fun.items()}
    
    # 3) figure setup
    if new_figure or ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        plt_template(ax, dark_mode)
    else:
        fig = ax.figure
        ax.clear()

    if frame_bounds:
        ax.set_xlim(frame_bounds[0], frame_bounds[1])
        ax.set_ylim(frame_bounds[2], frame_bounds[3])
    else:
        all_pts = np.vstack((CG, GR)) if GR.size else CG
        pad = 0.1 * np.ptp(all_pts, axis=0)
        ax.set_xlim(all_pts[:,0].min()-pad[0], all_pts[:,0].max()+pad[0])
        ax.set_ylim(all_pts[:,1].min()-pad[1], all_pts[:,1].max()+pad[1])

    ax.set_aspect("equal")
    ax.set_xlabel("X"); ax.set_ylabel("Z")
    if t_val is not None:
        ax.set_title(f"t = {t_val:.2f}s")

    # 4) draw CG
    cmap = plt.cm.get_cmap("tab20", len(CG))
    for i, (x, z) in enumerate(CG):
        ax.scatter(
            x, z,
            s=40,
            marker="D",
            color=cmap(i),
            edgecolors="k",
            label=f"CG body {i+1}"
        )
        ax.text(x, z, rf'CG$_{{{i+1}}}$', va='bottom', ha='right')

    # before your loops
    jo_plotted = False
    bd_plotted = False
    
    # 5) joints & connecting lines
    # we plot CG→JO and parent→JO
    for j_idx, joint in enumerate(MBD.joint_system.joints):
        xj, zj = JO[j_idx]
        if np.isnan(xj) or np.isnan(zj):
            continue
        child_cg = CG[joint.child-1]
        ax.plot([child_cg[0], xj],[child_cg[1], zj],
                color=cmap(joint.child-1), lw=2.5)
        if joint.parent != 0:
            par_cg = CG[joint.parent-1]
            ax.plot([par_cg[0], xj],[par_cg[1], zj],
                    color=cmap(joint.parent-1), lw=2.5)
        lbl = "Joints" if not jo_plotted else None
        ax.scatter(xj, zj, s=40,
                   facecolors=cmap(joint.child-1),
                   edgecolors="k",marker='o',label=lbl)
        jo_plotted = True
        
        if joint.joint_type.value in ('R','P'):
            p, c = joint.parent, joint.child
            t    = joint.joint_type.value
            ax.text(xj, zj, rf'Jo$({t}_{p},{t}_{c})$', va='bottom', ha='left')


    # 6) body points & ground
    for body, mat in BD.items():
        c = cmap(body-1)
        cg = CG[body-1]
        k = 0
        for (x,z) in mat.reshape(-1,2):
            ax.plot([cg[0], x],[cg[1], z], "--", lw=1, color=c)
            lbl = "Body Points" if not bd_plotted else None
            ax.scatter(x, z, marker="x", color=c,label=lbl)
            bd_plotted = True
            ax.text(x, z, rf'BD$_{{{body},{k}}}$', va='top', ha='right')
            k += 1

    if GR.size:
        ax.scatter(GR[:,0], GR[:,1], s=60, marker="s", color="k", label="GR")
        for m,(xg,zg) in enumerate(GR):
            ax.text(xg, zg, rf'GR$_{{{m}}}$', va='top', ha='left')  
    
    # Plotting forces
    # 1) Determine a reasonable arrow‐scaling based on the domain size
    all_x = np.hstack((CG[:,0], GR[:,0] if GR.size else [], *[mat[:,0] for mat in BD.values()]))
    all_z = np.hstack((CG[:,1], GR[:,1] if GR.size else [], *[mat[:,1] for mat in BD.values()]))
    domain_size = max(np.ptp(all_x), np.ptp(all_z), 1.0)
    # make arrow length ~5% of domain
    arrow_scale = domain_size * 0.05

    # 2) build the CG‐force array
    cg_forces = []
    cg_bodies = []
    for entry in Force_dict.get("CG", []):
        # entry = [body, Fx_sym, Fy_sym, Mz_sym]
        body = entry[0]
        Fx_sym, Fy_sym = entry[1], entry[2]
        # substitute t→t_val if symbolic, else cast directly
        Fx = to_float(Fx_sym)
        Fy = to_float(Fy_sym)
        cg_bodies.append(body-1)
        cg_forces.append([Fx, Fy])
    cg_forces = np.array(cg_forces, float)
    
    if cg_forces.size:
        ax.quiver(
            CG[cg_bodies,0], CG[cg_bodies,1],             # arrow origins
            cg_forces[:,0], cg_forces[:,1],  # (u,v) = (Fx, Fy)
            angles='xy', scale_units='xy', scale=1/arrow_scale,
            color='r', width=0.005, label='CG Forces'
        )

    # 3) Body‑Defined Point forces (Force_dict["PointsBD"] entries are [body, ptID, Fx, Fy, Mz])
    bd_entries = Force_dict.get("PointsBD", [])
    if bd_entries:
        pts = []
        fbd = []
        for body, ptID, Fx, Fy, _Mz in bd_entries:
            body = int(body)      # 1‑based
            pt   = int(ptID)
            coords = BD[body].reshape(-1,2)[pt]
            pts.append(coords)
            fbd.append([to_float(Fx), to_float(Fy)])
        pts = np.array(pts)
        fbd = np.array(fbd)
        ax.quiver(
            pts[:,0], pts[:,1],
            fbd[:,0], fbd[:,1],
            angles='xy', scale_units='xy', scale=1/arrow_scale,
            color='b', width=0.005, label='Point Forces'
        )
    
    
    # 7) springs & dampers (optional)
    # tension springs
    if Force_dict.get("TensionSpring"):
        for pts, _ in Force_dict["TensionSpring"]:
            pt1 = resolve_point(pts[0], CG, BD, GR)
            pt2 = resolve_point(pts[1], CG, BD, GR)
            xs, ys = spring_coords(pt1, pt2, 0.05, 10)
            ax.plot(xs, ys, "-k", lw=1.2)
    # tension dampers
    if Force_dict.get("TensionDamper"):
        for pts, _ in Force_dict["TensionDamper"]:
            pt1 = resolve_point(pts[0], CG, BD, GR)
            pt2 = resolve_point(pts[1], CG, BD, GR)
            damper_plot(pt1, pt2, ax)
    # torsion springs
    if Force_dict.get("TorsionSpring"):
        for (ent, *_) in Force_dict["TorsionSpring"]:
            body1, body2 = ent[:2]
            j_idx = next(i for i,j in enumerate(MBD.joint_system.joints)
                         if (j.parent,j.child)==(body1,body2))
            torsion_spring(JO[j_idx], 0.03, ax)
            torsion_spring(JO[j_idx], 0.03*0.75, ax)
    # torsion dampers
    if Force_dict.get("TorsionDamper"):
        for (ent, *_) in Force_dict["TorsionDamper"]:
            body1, body2 = ent[:2]
            j_idx = next(i for i,j in enumerate(MBD.joint_system.joints)
                         if (j.parent,j.child)==(body1,body2))
            torsion_damper(JO[j_idx], 0.05, ax)
            
    if Force_dict.get("Moorings"):
        # Plot initialization created in MBD2MoorDyn through manager
        if t_val is None:
            print('printing time is not provided. Skipping mooring line plot')
            return
        
        # Plot the results at t = 0 in red
        ms = xfm.get_adapter("moordyn").plot
        ms.plot2d(ax=ax ,color="red", time = t_val)
        # print(f"Passed current time to MoorDyn is:\t{t_val}")
 
    plt_template(ax, dark_mode)
    

    return fig, ax

def bound_finder(frame_tol, t, y ,mainNumVars, MBDsys):
    """
    Compute fixed plotting bounds by scanning every CG and BD point over time.

    Parameters
    ----------
    frame_tol : float
        extra padding to add on each side
    t : array_like
        time points
    y : array_like
        solver output: [Q, QD] at each time step
    mainNumVars : array_like
        the full set of numeric arguments
    MBDsys : MBDSystem
        your multibody system instance

    Returns
    -------
    min_x, max_x, min_z, max_z : tuple of floats
        the tightest bounds plus the requested padding
    """
    # how many coordinates in Q + QD?
    n_states = y.shape[0]
    
    # 1) Pre‑seed the mins/maxs with the ground points
    raw_gr = MBDsys.GRpoints_func(*mainNumVars)

    if raw_gr:
        GR_all = np.vstack(raw_gr)
        x_min = GR_all[:,0].min()
        x_max = GR_all[:,0].max()
        z_min = GR_all[:,1].min()
        z_max = GR_all[:,1].max()
    else:
        GR_all = np.array([])
        x_min, x_max = np.inf, -np.inf
        z_min, z_max = np.inf, -np.inf
    
    # 2) Time–loop
    for i, ti in enumerate(t):
        # put **both** Q and QD back into the first 2*n_states slots
        mainNumVars[:n_states] = y[:, i]
    
        # eval bodies‑only points
        CG_all = MBDsys.CGpoints_func(*mainNumVars)  # (n_bodies, 2)
        raw_JO = MBDsys.JOpoints_func(*mainNumVars)  # (n_joints,2)
        # keep only rows where both x and z are finite
        good    = np.isfinite(raw_JO).all(axis=1)
        JO_all  = raw_JO[good]
        
        BD_list = []

        for bd_func in MBDsys.BDpoints_func.values():
            pts = np.array(bd_func(*mainNumVars))   # shape (n_i,2) or (0,2)
            if pts.size:
                BD_list.append(pts)
        
        if BD_list:
            BD_all = np.vstack(BD_list)            # shape (sum n_i, 2)
        else:
            BD_all = np.zeros((0,2))               # no BD points defined
    
        # gather all Xs and Zs this step
        xs = np.hstack((CG_all[:,0], BD_all[:,0], JO_all[:,0], GR_all[:,0] if GR_all.size else []))
        zs = np.hstack((CG_all[:,1], BD_all[:,1], JO_all[:,1], GR_all[:,1] if GR_all.size else []))
    
        # update running mins/maxs
        cur_min_x, cur_max_x = xs.min(), xs.max()
        cur_min_z, cur_max_z = zs.min(), zs.max()
    
        if cur_min_x < x_min: x_min = cur_min_x
        if cur_max_x > x_max: x_max = cur_max_x
        if cur_min_z < z_min: z_min = cur_min_z
        if cur_max_z > z_max: z_max = cur_max_z
    
    # 3) apply padding
    min_x = x_min - frame_tol
    max_x = x_max + frame_tol
    min_z = z_min - frame_tol
    max_z = z_max + frame_tol
    
    # now pass (min_x, max_x, min_z, max_z) into your PlotMultibodySystem2 call
    # add tolerance
    return (min_x,max_x,min_z,max_z)


def animate_multibody(tvec, y, MBD, mainNumVars,
                      frame_bounds=None, save_path=None, fps=20, 
                      loop = False, xfm=None):   
    """
    Create and optionally save an animation of the multibody simulation.

    Uses `plot_multibody_system` internally to render each frame.

    Parameters
    ----------
    tvec : array‐like, shape (n_frames,)
        Time values for each frame.
    y : array‐like, shape (n_states, n_frames)
        Simulation state array; rows are state variables, columns are time steps.
    MBD : object
        Multibody‐dynamics container used by `plot_multibody_system`.
    mainNumVars : array‐like
        Template array of numeric values for all symbolic variables. Updated
        each frame with `y` and (if needed) `t`.
    frame_bounds : tuple of 4 floats, optional
        Axis limits passed to the first call of `plot_multibody_system`.
    save_path : str or Path, optional
        If provided, path to write out the animation (via PillowWriter).
    fps : int, optional
        Frames per second for playback and saving. Default is 20.
    loop : bool, optional
        If True, the animation will loop when replayed. Default is False.
    xfm : optional
        Transformation/adapter for external plotting (e.g. MoorPy).

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The FuncAnimation object, which can also be used to save or display
        the animation outside this function.
    """

    frame_tol = 1
    frame_bounds = bound_finder(frame_tol, tvec, y ,mainNumVars, MBD)
    # --------------------------------------------------------------------
    # 1)  Create figure/axis once
    # --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8,5))
    ax.grid(True)
    
    # --------------------------------------------------------------------
    # 2)  The *wrapper* that will be called every frame
    # --------------------------------------------------------------------
    def draw_frame(frame,                                # <-- comes from FuncAnimation
                   mainNumVars, MBD, xfm,            # <-- fixed by partial
                   time_all, y_all,                    # <--        "
                   plot_fn, plot_kwargs):
        """
        Update mainNumVars with y[:,i] and tvec[i], then call plot_multibody_system.
        Returns a list of Artists when blitting, otherwise None.
        """
        y = y_all[:,frame]
        
        # 2.1) update dynamic slots
        current_time            = time_all[frame]
        Vars_copy               = mainNumVars.copy()
        Vars_copy[:len(y)]      = y   # Q and QD
        Vars_copy[MBD.t_update] = current_time       # the symbolic 't'
    
        # 2.2) clear or reuse the existing Axes
        ax.cla()
    
        # 2.3) delegate the real drawing to your existing helper
        plot_fn(
            main_num_vars   = Vars_copy,
            t_val           = current_time,
            ax              = ax,
            new_figure      = False,
            **plot_kwargs
            )
        ax.grid(True)
    
        # 2.4)  Return artists only if you plan to use blit=True
        return ax.collections + ax.lines          # <-- or []
    
    # --------------------------------------------------------------------
    # 3)  Pre‑fill every argument that does *not* change each frame
    # --------------------------------------------------------------------
    static_args = dict(
        MBD             = MBD,
        frame_bounds    = frame_bounds,
        dark_mode       = False,
        xfm             = xfm
        )
    
    frame_plotter = partial(
        draw_frame,
        mainNumVars = mainNumVars,           # shared NumPy array
        MBD         = MBD,
        xfm         = xfm,
        time_all    = tvec,
        y_all       = y,
        plot_fn     = plot_multibody_system,
        plot_kwargs = static_args
        )
    
    # --------------------------------------------------------------------
    # 4)  Build the animation (FuncAnimation only sees the frame index)
    # --------------------------------------------------------------------
    anim = FuncAnimation(
        fig,
        frame_plotter,          # the callback
        frames      = range(len(tvec)),
        interval    = 1000 / fps,  # ms between frames
        blit        = False,            # True only if draw_frame returns artists
        repeat      = loop
        )
    
    plt.show()
    
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=fps))
    
    plt.show()
    plt.close(fig)
    return anim

    



