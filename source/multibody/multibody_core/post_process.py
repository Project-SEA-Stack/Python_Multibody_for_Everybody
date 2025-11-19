import numpy as np


def evaluate_trajectories(MBDsys, sol_integration, main_num_vars_template):
    """
    Evaluate center-of-mass positions, velocities, and joint locations over time.

    Parameters
    ----------
    time : array_like, shape (T,)
        Time vector for the simulation.
    y_hist : array_like, shape (T, 2*n_states)
        History of generalized coordinates (positions) Q and velocities QD at each time.
    main_num_vars_template : ndarray
        Template of numeric inputs for lambdified functions, length = len(mainSymVars)+extra.
        Dynamic slots 0:n_states filled by [Q, QD], slots in t_update_indices filled by time.
    MBDsys : MBDSystem
        The multibody dynamics system instance.

    Returns
    -------
    com_positions : ndarray, shape (T, NBodies, 2)
        X, Z positions of each body CG over time.
    com_velocities : ndarray, shape (T, NBodies, 2)
        X, Z velocities of each body CG over time.
    joint_positions : ndarray, shape (T, n_joints, 2)
        X, Z locations of each joint over time.
    """
    # Unpack the solution
    time    = np.asarray(sol_integration.t)
    y_hist  = np.asarray(sol_integration.y)

    #Extract state variables
    Q_length    = y_hist.shape[0] // 2
    Q_hist      = y_hist[:Q_length]
    QD_hist     = y_hist[Q_length:]

    # Pre-allocate mainNumVars
    mainNumVars = main_num_vars_template.copy()

    # Pre-allocate arrays dimensions
    n_states,T = Q_hist.shape
    NBodies     = len(MBDsys.NDOF)
    n_joints    = len(MBDsys.joints)

    com_positions   = np.zeros((T, NBodies, 2)) # 2 stands for 2 translational coordinates
    com_velocities  = np.zeros((T, NBodies, 2))
    joint_positions = np.zeros((T, n_joints, 2))
    angle_positions = np.zeros((T, NBodies))

    for i in range(T):
        # fill Q and QD
        mainNumVars[:2*n_states] = np.hstack((Q_hist[:,i], QD_hist[:,i]))
        # fill time
        if MBDsys.t_update is not None:
            for idx in MBDsys.t_update:
                mainNumVars[idx] = time[i]

        pos     = MBDsys.Pos_func(*mainNumVars)
        vel     = MBDsys.Vel_func(*mainNumVars)
        jloc    = MBDsys.JointLoc_func(*mainNumVars)

        # reshape
        pos_mat = pos.reshape((NBodies, 3))
        vel_mat = vel.reshape((NBodies, 3))

        com_positions[i]   = pos_mat[:, :2] # X, Z positions for each time step
        angle_positions[i] = pos_mat[:, 2]  # Angular positions for each time step
        com_velocities[i]  = vel_mat[:, :2] # X, Z velocities for each time step
        joint_positions[i] = jloc

    return com_positions, com_velocities, angle_positions, joint_positions
