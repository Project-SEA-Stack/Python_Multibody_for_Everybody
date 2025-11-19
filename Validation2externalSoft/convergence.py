import os
import numpy as np
import matplotlib.pyplot as plt
from multibody import plot


def compare_npy_arrays(arr_m4e, arr_chrono, time, eps=1e-4):
    """
    Error metrics comparing two numpy arrays of shape (T, NB, ND).
    The arrays represent the same physical quantities (e.g., positions of bodies)
    Each array has T time steps, NB bodies, and ND degrees of freedom (DOFs) per body.

    """
    if arr_m4e.shape != arr_chrono.shape:
        raise ValueError(f"Shape mismatch: {arr_m4e.shape} vs {arr_chrono.shape}")

    ################ Error metrics from Okada 2022 (modified) ################
    ############################################################################
    # Get number of steps time steps, N
    N = len(time)

    # Errors per body and DOF at each time step
    max_ref                 = np.max(np.abs(arr_chrono), axis=0) # Normalization factors from Gonzalez 2006
    rel_L2_per_DOF          = np.abs(arr_m4e - arr_chrono) / max_ref
    NRMSE_per_body_per_DOF  = np.linalg.norm(rel_L2_per_DOF / N, axis=0)
    NRMSE_per_body          = np.linalg.norm(NRMSE_per_body_per_DOF,axis=-1)

    return {
        'NRMSE_per_body': np.atleast_1d(NRMSE_per_body),
        'NRMSE_per_body_per_DOF': np.atleast_2d(NRMSE_per_body_per_DOF),
        'L2_per_DOF': np.atleast_3d(rel_L2_per_DOF),
    }


def _try_load_time_vector(folder: str):
    """Attempt to load a time vector from common file names in `folder`."""
    for name in ("t_chrono.npy", "time_chrono.npy", "t_M4E.npy"):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            try:
                return np.load(p)
            except Exception:
                pass
    return None


if __name__ == "__main__":
    # tolerances  = [(1e-4,1e-7),(1e-6,1e-9),(1e-8,1e-11),(1e-10,1e-13),(1e-12,1e-15)] #M4E
    tolerances  = [1e-2,1e-3, 5e-4, 2.5e-4, 1.25e-4] #Chrono
    # folder      = "integrator_comparison_chaos\chrono_a0"  # Change to your folder name
    folder      = "integrator_comparison\\chrono_a0"  # Change to your folder name
    folder2      = "integrator_comparison"  # Change to your folder name
    base_dir    = os.path.join("Validation2externalSoft\\results", folder)
    base_dir2    = os.path.join("Validation2externalSoft\\results", folder2)

    r_list     = []
    err_list   = []
    # for i, (rtol, atol) in enumerate(tolerances):
    for i, timeStep in enumerate(tolerances):
        # print(f'\Loading with rtol={rtol}, atol={atol}')

        # Load all files
        # file_M4E    = os.path.join(base_dir2, f"r_M4E_{i}.npy")
        file_M4E    = os.path.join(base_dir, f"r_chrono_{i}.npy")
        arr_m4e     = np.load(file_M4E)

        r_list.append(arr_m4e)

    for j in range(len(r_list)-1):
        err = compare_npy_arrays(r_list[j], r_list[-1], base_dir)
        RMS = err['NRMSE_per_body']
        # print(f"Comparison between rtol={tolerances[j]} and rtol={tolerances[-1]}: {RMS}")
        err_list.append(RMS)

    # plot error list vs rtol as log-log plot
    # rtol_list   = [t[0] for t in tolerances[:-1]]
    rtol_list   = [t for t in tolerances[:-1]]
    rtol_np     = np.asarray(rtol_list)
    err_np      = np.asarray(err_list)
    alpha       = np.polyfit(np.log(rtol_np), np.log(err_np), 1)[0]
    print(f"Error convergence rate: {alpha}") # theoretical convergence rtol^0.83.

    plt.figure(figsize=(11, 9))
    ax = plt.gca()
    

    for i in range(len(err_np[0])):
        ax.loglog(rtol_np, [err_np[j][i] for j in range(len(err_np))],
                marker='o', label=f'Body {i+1}')

    ax.set_xlabel('rtol')
    ax.set_ylabel('RMS error [m]')
    ax.set_title('RMS error against rtol=1e-12 solution')
    ax.grid(True, which="both", ls="--")
    ax.legend()
    ax.invert_xaxis()
    plot.plt_template(ax, dark=False)
    plt.tight_layout()
    plt.show()


    tolerances_MAE  = [(1e-4,1e-7),(1e-6,1e-9),(1e-8,1e-11),(1e-10,1e-13),(1e-12,1e-15)] #M4E
    xlabel = [1,2,3,4,5] + [1,2,3,4,5]
    symbols = ['o','o','o','o','o'] + ['x','x','x','x','x']
    colors = ['r','r','r','r','r'] + ['b','b','b','b','b']
    labels = [f'Chrono with dt: {a:.2e}' for a in tolerances] + [f'M4E with rtol: {t[0]:.2e}' for t in tolerances_MAE]
    for i, (rtol, atol) in enumerate(tolerances_MAE):
        # print(f'\Loading with rtol={rtol}, atol={atol}')

        # Load all files
        file_M4E    = os.path.join(base_dir2, f"r_M4E_{i}.npy")
        # file_M4E    = os.path.join(base_dir, f"r_chrono_{i}.npy")
        arr_m4e     = np.load(file_M4E)

        r_list.append(arr_m4e)

    # plot all eleements of r_list
    compare_distance = []
    plt.figure(figsize=(11, 9))
    ax = plt.gca()
    
    for i in range(len(r_list)):
        ax.plot(xlabel[i], r_list[i][-1,1,1],symbols[i],label=labels[i], color=colors[i],markersize=10)
        ax.set_ylabel('Final position x [m]')
        ax.set_title('Final position comparison for equivalent tolerances')

        compare_distance.append(r_list[i][-1,1,1])

    plt.legend()
    plot.plt_template(ax, dark=False)
    plt.tight_layout()
    plt.show()

    compare_distance_np = np.asarray(compare_distance).reshape((-1,5)).T

    np.set_printoptions(precision=2)
    print(np.diff(compare_distance_np,axis=1))    