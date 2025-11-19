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
    for name in ("t_M4E.npy", "t_chrono.npy", "time_chrono.npy"):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            try:
                return np.load(p)
            except Exception:
                pass
    return None


if __name__ == "__main__":
    # Specify file paths to load results from
    saveImage   = 0
    folder      = "yaml_spiderfloat"  # Change to your folder name
    base_dir    = os.path.join("Validation2externalSoft\\results", folder)
    file_M4E    = os.path.join(base_dir, "r_M4E.npy")
    file_chrono = os.path.join(base_dir, "r_chrono.npy")

    # Load data from .npy files
    # r2plot = [0, 1 , 2]  # Bodies to plot (0-indexed); set to [] to skip plotting
    r2plot = [3,4,5,6]
    arr_m4e     = np.load(file_M4E)
    arr_chrono  = np.load(file_chrono)
    t_vec       = _try_load_time_vector(base_dir)
    arr_m4e     = arr_m4e[:, r2plot, :]
    arr_chrono  = arr_chrono[:, r2plot, :]
    
    # Compute error metrics and print results
    result = compare_npy_arrays(arr_m4e, arr_chrono,t_vec)
    print('Error metrics (Chrono = truth):')
    print('NRMSE_per_body:',
        " ".join(f"\t{v:.3e}" for v in result['NRMSE_per_body']))
    print('NRMSE per body and DOF:',
          " ".join(f"body {i+1} \t{v[0]:.3e}, \t{v[1]:.3e}" for i, v in enumerate(result['NRMSE_per_body_per_DOF'])))

    # plot each body DOFs IN A SEPARATE SUBPLOT
    fig,ax = plt.subplots(arr_m4e.shape[1], 1, figsize=(11, 12))
    if arr_m4e.shape[1] == 1:
        ax = [ax]
    for i in range(arr_m4e.shape[1]):
        ax[i].plot(t_vec, arr_chrono[:, i,0], label='x-Chrono')
        ax[i].plot(t_vec, arr_m4e[:, i,0], label='x-M4E', linestyle='--')
        ax[i].plot(t_vec, arr_chrono[:, i,1], label='z-Chrono')
        ax[i].plot(t_vec, arr_m4e[:, i,1], label='z-M4E', linestyle='--')
        ax[i].set_title(f'Body {i + 4} Center of mass position comparison')
        ax[i].set_xlabel('Time [s]')
        ax[i].grid(True, linestyle='--', alpha=0.4)
        ax[i].legend()

        plot.plt_template(ax=ax[i])

    fig.tight_layout()
    plt.show()

    # quit()
    # Plot L2- error per body in a separate subplot
    L2_per_DOF = result['L2_per_DOF']

    fig,ax = plt.subplots(arr_m4e.shape[1], 1, figsize=(11, 9))
    if arr_m4e.shape[1] == 1:
        ax = [ax]
    for i in range(arr_m4e.shape[1]):
        x_dif = L2_per_DOF[:, i,0]
        z_dif = L2_per_DOF[:, i,1]
        ax[i].plot(t_vec, x_dif, label='x-difference')
        ax[i].plot(t_vec, z_dif, label='z-difference')
        ax[i].set_title(f'Body {i + 1} Center of mass position comparison for {folder} example')
        ax[i].set_xlabel('Time [s]')
        ax[i].grid(True, linestyle='--', alpha=0.4)
        ax[i].legend()

        plot.plt_template(ax=ax[i])
    plt.show()


    # if saveImage == 1:
    #     out1 = os.path.join(base_dir, 'percent_error_per_body_vs_time.png')
    #     plt.savefig(out1, dpi=150)
    #     print(f"Saved per-body percent error plot: {out1}")

