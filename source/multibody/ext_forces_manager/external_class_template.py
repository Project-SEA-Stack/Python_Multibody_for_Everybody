# multibody/MBD2MoorDyn.py
from __future__ import annotations
from pathlib import Path
from collections import namedtuple
import numpy as np

# Import External libraries
import moordyn as ExtLibrary

from multibody.ext_forces_manager import CartForce

class TemplateInterface:
    """
    Thin adapter that hides all MoorDyn details behind the 3-method contract
    expected by `ExternalForcesManager`.
    """
    name = 'WhateverNameYouWant'    # This is used to include plotting functionality
    # ------------------------------------------------------------------
    def __init__(self,
                 mbd_sys,
                 numericalValuesTuple,
                 *,
                 Custom_inputs,
                 Custom_inputs2,
                 **writer_kwargs):
        """
        Parameters
        ----------
        mbd_sys     : multibody.MBDSystem     (to retrieve points/bodies)
        force_dict  : the **Force** section from the user example
        MDline_types  : → MoorDyn LINE TYPES section
        MDoptions     : → MoorDyn OPTIONS
        MDoutfile     : path of the generated *.txt*
        writer_kwargs : extra keywords forwarded to `write_moordyn_file`
                        e.g. bodies=...  if you want to override defaults
        """
        self._mbd   = mbd_sys

        # mainNumVars contains the numerical values for the joint coordinates positions,
        # velocities and other symbolic variables
        # m0 and J0 are the mass and inertia of the system
        mainNumVars, m0, J0 = numericalValuesTuple

        # Create the MoorDyn input file (Only if library initialized through txt file)
        # The method does not need to be 'Create', it can be any name
        self.ExtInstance = ExtLibrary.Create(Custom_inputs)

        # If you are just passing CG points you just need to keep track of which body CG you are using
        # The mapping is created inside of the functions (tweak as needed)
        # If you are operating with points other than CG, you need to create a mapping (contact Alvaro)
        self._external2mbd_mapping(self, self._mbd.Forces['ForceEntryName'])

        # Actual external library initilization (Moordyn)
        ExtLibrary.Init(self.ExtInstance, Custom_inputs2)   # zero initial state

        # cache for latest forces
        self._cached_forces: list[CartForce] = []

    # ------------------------------------------------------------------
    # required by the adapter contract
    # ------------------------------------------------------------------
    def update(self, t, q, qd, mainNumVars):
        ''' Advance MoorDyn one step assume external driver time-steps it'''

        # Get updated position and velocity of the points 
        MBDx0,MBDv0 = self._evaluateCGpoints(mainNumVars)

        dt = t-self.t_old

        # Run External library integrator with the inputs you need
        f           = ExtLibrary.Step(self.ExtInstance, MBDx0, MBDv0, t, dt)

        # Reshape the forces to points x 3 array
        f_shaped    = np.array(f).reshape((-1,3))

        self._cached_forces.clear()

        for number,pointDict in enumerate(self.CGpointMap):
            # This assumes that forces are only displayed for BD points and that 
            # the order matches the order in which these BD points are originally written            
            MBDbody    = pointDict['body']
            MBDlocalID = pointDict['localID']
            MBDpType   = pointDict['pType']
            MDvector  = f_shaped[number]

            self._cached_forces.append(
                CartForce(body=MBDbody, pType=MBDpType, localID=MBDlocalID,
                          vector=np.array([MDvector[0],MDvector[2]]))
            )

            self.t_old = t

    # keep this method for compatibility with the interface
    def forces(self):
        return list(self._cached_forces)

    def end(self):
        # close the external library system (method may not be needed)
        ExtLibrary.Close(self.ExtInstance)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _external2mbd_mapping(self, extForceEntry,numericalValues):
        """
        Translate Force['Moorings'] into MoorDyn POINTS / LINES.
        Inputs belong to MBD
        Outputs belongs to MD
        """
        mbd = self._mbd
        mainNumVars, m0, J0 = numericalValues

        # Point ID counter in case you have a global numbering of points, 
        # otherwise no need to call this function
        point_id        = 1
        self.CGpointMap = []

        for pointInfo, forceInfo in extForceEntry:                
                
            if pointInfo.startswith("CG"): # Coupled
                # Unpack values for BD points
                b  = int(pointInfo[2])
                
                # Map between global id and local ID for External Forces Manager
                # Used to translate from MoorDyn forces to MBD forces
                # Local ID is always -1 for CG points
                self.CGpointMap.append(dict(ID=point_id, pType='CG',body=b, localID=-1))
                
                point_id += 1
            
            raise ValueError(f"Unknown code {pointInfo}")
            
        # return the mapping of points
    
    def _evaluateCGpoints(self, mainNumVars):
        # These are all CG variables
        x0 = []
        v0 = []
        CGpos = self._mbd.CGpoints_func
        CGvel = self._mbd.CGvel_func
        
        for i in self.CGpointMap:
            body    = i['body']
            
            # This is relative to CG
            x0point = CGpos(*mainNumVars)[body-1] # The -1 is because the bodies are 1-based (0 is ground)
            v0point = CGvel(*mainNumVars)[body-1]

            x0.append(np.insert(x0point, 1, 0))  # Make the vector 3D with y-component 0
            v0.append(np.insert(v0point, 1, 0))  # Make the vector 3D with y-component 0

        return np.hstack(x0), np.hstack(v0)
    
