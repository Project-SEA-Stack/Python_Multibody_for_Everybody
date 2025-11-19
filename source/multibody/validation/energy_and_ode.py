# -*- coding: utf-8 -*-
"""
Energy vs. ODE‐solver Comparison
================================

This script loads the “true” energy traces computed in MATLAB, runs
the same examples through our Python `MBDSystem.integrate(...)`, and
then:

  1. Resamples both time series onto a common grid  
  2. Computes max‐abs and RMSE in total energy  
  3. Plots MATLAB vs. Python energy traces side by side  

Usage
-----
Just run:

    python energy_and_ode.py
    
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import importlib
from multibody import MbdSystem
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

if __name__=="__main__":
    # Summary of errors
    summaryE = []
    summaryY = []
    
    # 1) pull in both traces
    j = 9
    for i in range(1,j):
        m = loadmat(f'./TrueEnergy/Em_example{i}.mat')
        t_mat, y_mat, E_mat = m['time'].ravel(), m['y'].astype('float32'), m['Em_num'].astype('float32')
        
        # NOTE: Removing the last time step from MATLAB data to avoid time alignment issues
        t_mat   = t_mat[:-1]
        y_mat   = y_mat[-2,:].ravel()
        E_mat   = E_mat[:-1].ravel()
        ############# Run python example #######################################
        # Example to import and init
        module_name = f"Pytests.Example{i}"
        ex          = importlib.import_module(module_name)
        MBDsys      = MbdSystem.from_example(ex)
    
        # Define initial numerical values
        mainNumVars = np.hstack((ex.ic, ex.ForcesPointsNum, ex.BodyDataNum))
        
        # Integrate
        sol         = MBDsys.integrate(mainNumVars, ex.m0, ex.J0, 
                                   tspan=ex.tspan, dt=ex.TimeStep)  # ❷ run
        ########################################################################
        
        E_py        = []
        t_py        = sol.t 
        y_last_py   = sol.y[:,-1]
    
        for k, ti in enumerate(t_py):
            mainNumVars_copy                   = mainNumVars.copy()
            mainNumVars_copy[:len(sol.y)]      = sol.y[:,k]
            mainNumVars_copy[MBDsys.t_update]  = ti
            
            E_py.append(MBDsys.Energy_func(*mainNumVars_copy,
                                            *ex.m0, *ex.J0))
        E_py = np.array(E_py)
        
        # 2) align on a common time‐base (if they differ slightly)
        tol = 0.001 * ex.TimeStep  # your tolerance on the time grid
        if t_mat.shape == t_py.shape and np.allclose(t_mat, t_py, atol=tol, rtol=0.0):
            # simple nearest‐neighbor re‐sampling
            E_py_interp = np.interp(t_mat, t_py, E_py).astype('float32')
        else:
            E_py_interp = E_py
        
        # 3) compute difference metrics
        diffEm  = E_py_interp - E_mat
        diffy   = y_last_py - y_mat
        abs_err = np.abs(diffEm)
        max_err = abs_err.max()
        rmse    = np.sqrt((diffEm**2).mean())
        rmseY   = np.sqrt((diffy**2).mean())
        print(f"Max abs error:  {max_err:.3e}")
        print(f"RMSE:           {rmse:.3e}")
        print(f"RMSE of last y: {rmseY:.3e}\n")
        
        summaryE.append(rmse)
        summaryY.append(rmseY)
    
        # 5) optional plot
        plt.figure(figsize=(10, 5))
        plt.plot(t_mat, E_mat, label='MATLAB')
        plt.plot(t_mat, E_py_interp, '--', label='Python')
        plt.title(f"Energy Comparison for example {i}")
        plt.xlabel("Time (s)")
        plt.ylabel("Total Energy")
        plt.legend()
        plt.show()
        
    print(f"The Energy rmse for each example is {np.float64(summaryE)}\n")
    print(f"The last times step joint coordinates rmse for each example is {np.float64(summaryY)}")
