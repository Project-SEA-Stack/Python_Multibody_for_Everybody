# multibody/MBD2MoorDyn.py
from __future__ import annotations
from pathlib import Path
from collections import namedtuple
import numpy as np
import moordyn
import moorpy

from .moordyn_writter import write_moordyn_file
from multibody.ext_forces_manager import CartForce

class MoorDynInterface:
    """
    Thin adapter that hides all MoorDyn details behind the 3-method contract
    expected by `ExternalForcesManager`.
    """
    name = 'moordyn'    # This is necessary for plotting the mooring lines
    # ------------------------------------------------------------------
    def __init__(self,
                 mbd_sys,
                 numericalValuesTuple,
                 *,
                 dt: float,
                 MDline_types: list|dict|None = None,
                 MDoptions: dict|None = None,
                 MDoutputs: list[str]|None = None,
                 MDoutfile: str|None = "berrinche.txt",
                 MD_infile: str|None = None,
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
        self.MD_outfile = Path(MDoutfile)
        self.dt = dt
        self.t_old = 0.0 # for MoorDyn stability
        self.no_warning1 = True # for MoorDyn timestepping. Warn once if dt from integrator is << than self.dt
        self.no_warning2 = True # for MoorDyn out of plane forces. Warn once if FY for any coupled point is not zero

        # --------------------------------------------------------------
        # Build POINTS / LINES / BODIES once, at construction time
        # --------------------------------------------------------------
        moorings = mbd_sys.Force.get("Moorings", [])
        if not moorings:
            raise ValueError("'Force[\"Moorings\"]' is empty – nothing for MoorDyn!")

        self.MD_point_rows, self.MD_line_rows = \
            self._build_input_tables(moorings,numericalValuesTuple)

        if MD_infile is None:
            if MDoptions is None or MDoutputs is None or MDline_types is None:
                raise ValueError("If 'MD_infile' is not provided, 'MDoptions', 'MDoutputs', and 'MDline_types' must be provided.")
            
            write_moordyn_file(
                self.MD_outfile,
                line_types=MDline_types,
                points=self.MD_point_rows,
                lines=self.MD_line_rows,
                # bodies=self._body_rows,
                options=MDoptions,
                outputs=MDoutputs
            )
        
        else:
            if MDoptions is not None or MDoutputs is not None or MDline_types is not None or MDoutfile is not None:
                print("WARNING: If 'MD_infile' is provided, 'MDoptions', 'MDoutputs', 'MDline_types', and 'outfile' will be ignored.")
            
            # consider this a beta feature, that is use at your own risk. Have implemented basic error checks with the teehee.txt file. 
            # THERE IS NO CHECKING THAT SYSTEM VALUES MATCH THE MOORDYN INPUT FILE (I.E. GRAVITY)

            self.MD_outfile = Path(MD_infile)

        # --------------------------------------------------------------
        # Init MoorDyn run-time object
        # --------------------------------------------------------------
        self.MoorDynsys = moordyn.Create(str(self.MD_outfile))

        # Get position and velocity of the points 
        x0,v0 = self._evaluateBDpoints(numericalValuesTuple[0])

        # Check for valid MD input file if using custom file
        if MD_infile is not None:
            num_coupled_p = 0
            num_coupled_b = 0
            num_coupled_r = 0
            
            num_points = moordyn.GetNumberPoints(self.MoorDynsys)
            for i in range(num_points):
                point = moordyn.GetPoint(self.MoorDynsys,i+1)
                ptype = moordyn.GetPointType(point) # coupled = -1, free = 0, fixed = 1
                if ptype == -1:
                    
                    pos = np.array(moordyn.GetPointPos(point))
                    if np.any(pos != x0[num_coupled_p*3: num_coupled_p*3+3]): # leverage the num_coupled_p index here, which is why its updated below this check.
                        raise ValueError(f"Position of coupled point {i+1} in MoorDyn input file does not match the position in the MBD system. "
                                         f"Expected {x0[num_coupled_p*3: num_coupled_p*3+3]} but got {pos}.")
                    
                    num_coupled_p += 1

            # Checks, TODO: move to helper file
            num_bodies = moordyn.GetNumberBodies(self.MoorDynsys)
            for i in range(num_bodies):
                body = moordyn.GetBody(self.MoorDynsys,i+1)
                btype = moordyn.GetBodyType(body) # coupled = -1, free = 0, fixed = 1, coupledpinned = 2
                if btype == -1 or btype == 2:
                    num_coupled_b += 1

            num_rods = moordyn.GetNumberRods(self.MoorDynsys)
            for i in range(num_rods):
                rod = moordyn.GetRod(self.MoorDynsys,i+1)
                rtype = moordyn.GetRodType(rod) # coupled = -2, coupledpinned = -1 free = 0, pinned = 1, fixed = 2
                if rtype == -1 or rtype == -2:
                    num_coupled_r += 1

            if num_coupled_b + num_coupled_r != 0:
                raise ValueError("This code is not set up for coupled bodies/rods in MoorDyn input file. "
                                 f"Number of coupled bodies and rods detected is ({num_coupled_b}) and ({num_coupled_r}) respectively. "
                                 "Please check your MoorDyn input file or the MBD system.")

            if num_coupled_p != len(self.BDpointMap):
                raise ValueError(f"Number of coupled points in MoorDyn input file ({num_coupled_p})"
                                 f" does not match the number of BD points in the MBD system ({len(self.BDpointMap)})."
                                 " Please check your MoorDyn input file or the MBD system.")

        # Check x0 and v0 are vectors of length 3*len(self.BDpointMap). i.e. 3 DOF for each coupled point
        if len(x0) != 3*len(self.BDpointMap) or len(v0) != 3*len(self.BDpointMap):
            raise ValueError(f"Expected x0 and v0 to be vectors of length {3*len(self.MD_point_rows)}"
                             f" but got {len(x0)} and {len(v0)} respectively. ")

        moordyn.Init(self.MoorDynsys, x0,v0)   # zero initial state

        # cache for latest forces
        self._cached_forces: list[CartForce] = []

    # ------------------------------------------------------------------
    # required by the adapter contract
    # ------------------------------------------------------------------
    def update(self, t, q, qd, mainNumVars):
        # Advance MoorDyn one step – assume external driver time-steps it
        # Get updated position and velocity of the points 
        MBDx0,MBDv0 = self._evaluateBDpoints(mainNumVars)

        # dt is adaptive time step, self.dt is the reporting time step
        dt = t-self.t_old

        # Warning if time step is much smaller than dt (only check if dt > 0, becasue dt = 0 in the first call to this loop)
        if ((dt - self.dt) < 1e-8) and (dt != 0.0) and self.no_warning1:
            print(f"[WARNING]: Adaptive time step {dt} is smaller than" 
                  f" the dt set in the input file ({self.dt}). This can lead"
                   " to long runtimes in MoorDyn. This warning will not be"
                   " repeated, though the issue may persist.")
            self.no_warning1 = False

        f           = moordyn.Step(self.MoorDynsys, MBDx0, MBDv0, t, dt)
        f_shaped    = np.array(f).reshape((-1,3))

        self._cached_forces.clear()

        for number,pointDict in enumerate(self.BDpointMap):
            # This assumes that forces are only displayed for BD points and that 
            # the order matches the order in which these BD points are originally written            
            MBDbody    = pointDict['body']
            MBDlocalID = pointDict['localID']
            MBDpType   = pointDict['pType']
            MDvector  = f_shaped[number]

            # Check for out of plane forces
            if not np.isclose(MDvector[1], 0.0) and self.no_warning2:
                print( "WARNING: Out of plane forces detected for BD point"
                      f" {MBDlocalID} on body {MBDbody}. FY = {MDvector[1]} N" 
                       " This warning will not be repeated, though the issue" 
                       " may persist.")
                self.no_warning2 = False

            self._cached_forces.append(
                CartForce(body=MBDbody, pType=MBDpType, localID=MBDlocalID,
                          vector=np.array([MDvector[0],MDvector[2],0.]))
            )

            self.t_old = t

    def forces(self):
        return list(self._cached_forces)

    def initialize(self, t0, q0, qd0):
        # no special initialisation needed beyond MoorDyn.Init
        pass

    def end(self):
        # close the moordyn system
        moordyn.Close(self.MoorDynsys)
        
    def init_plot(self):
        # Create a MoorPy system (this loads the moordyn outputs found in <dirname>/<rootname>_<object>#.out, 
        # where <object> is either Line or Rod and # is the corresponding number). 
        # Requires p flag in MoorDyn input file line list outputs. 
        
        # This is library specific and would need to be added every time we want
        # to plot new libraries, but is quite accessible.
        filenamePath    = self.MD_outfile
        filename        = str(filenamePath)
        rootName        = filenamePath.stem
        relative_path   = str(filenamePath.parent) + '/'
        
        # qs tells MoorPy it is loading a MoorDyn system, Fortran = False tells it to use the MoorDyn-C output format
        self.plot       = moorpy.System(file=filename, dirname=relative_path, rootname=rootName, qs = 0, Fortran = False)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_input_tables(self, moorings,numericalValues):
        """
        Translate Force['Moorings'] into MoorDyn POINTS / LINES.
        Inputs belong to MBD
        Outputs belongs to MD
        """
        mbd = self._mbd
        mainNumVars, m0, J0 = numericalValues

        # numeric helpers
        CG   = mbd.CGpoints_func(*mainNumVars)
        BD   = {k: f(*mainNumVars) for k, f in mbd.BDpoints_func.items()}
        GR   = np.vstack(mbd.GRpoints_func(*mainNumVars)) if mbd.GRpoints_func(*mainNumVars) else np.empty((0,2))

        point_rows      = []
        # body_rows       = {}
        point_id        = 1
        line_rows       = []
        self.BDpointMap = []

        def _resolve(code):
            nonlocal point_id
            
            # Create a list with all the points that are connected to mooring line
            if code.startswith("GR"): # Ground
                rest    = code[2:]
                if "_" not in rest:
                    raise ValueError(f"'{code}' uses the legacy point-string format. Use the delimited form instead, e.g. 'GR0_3'.")
                idx     = int(rest.split("_")[1])
                x, z    = GR[idx]
                
                # The list
                point_rows.append(dict(ID=point_id, Type="Fixed", X=x, Y=0, Z=z))
                
                # global id for MoorDyn
                pid     = point_id; point_id += 1
                
                return pid
            
            if code.startswith("BD"): # Coupled
                # Unpack values for BD points
                rest = code[2:]
                if "_" not in rest:
                    raise ValueError(f"'{code}' uses the legacy point-string format. Use the delimited form instead, e.g. 'BD12_3'.")
                b, loc = (int(v) for v in rest.split("_"))
                
                # Relative position of BD w.r.t CG in global frame of reference
                x, z    = BD[b][loc]
                xc, zc  = CG[b-1]
                rel     = (x-xc, z-zc)
                
                # Map between global id and local ID for External Forces Manager
                # Used to translate from MoorDyn forces to MBD forces
                self.BDpointMap.append(dict(ID=point_id, pType='BD',body=b, localID=loc, r2CG=rel))
                
                # The list
                point_rows.append(dict(ID=point_id, Type="Coupled", X=x, Y=0, Z=z))
                # body_rows.setdefault(b, (xc, zc, m0[b-1], J0[b-1])) 
                pid = point_id; point_id += 1
                
                return pid
                
                
            # if code.startswith("CG"): # Coupled
            #     # Unpack values for BD points
            #     b, loc  = int(code[2]), int(code[3:])
                
            #     # Relative position of BD w.r.t CG in global frame of reference
            #     xc, zc  = CG[b-1]   
                
            #     # Map between global id and local ID for External Forces Manager
            #     # Used to translate from MoorDyn forces to MBD forces
            #     # self.BDpointMap.append(dict(ID=point_id, pType='CG',body=b, localID=loc, r2CG=(xc,zc)))
                
            #     # The list
            #     point_rows.append(dict(ID=point_id, Type="Coupled", X=x, Y=0, Z=z))
            #     # body_rows.setdefault(b, (xc, zc, m0[b-1], J0[b-1])) 
            #     pid = point_id; point_id += 1
                
            #     return pid
            
            raise ValueError(f"Unknown code {code}")

        # Create a list with all the lines connecting the mooring line points
        for idx, ((p1, p2), prm) in enumerate(moorings, start=1):
            idA, idB = _resolve(p2), _resolve(p1) # End A is the bottom of the line, End B is the top of the line
            
            # The line list
            line_rows.append(dict(
                ID=idx, LineType=prm[0], AttachA=idA, AttachB=idB,
                UnstrLen=prm[1], NumSegs=prm[2], LineOutputs=prm[3]
                ))

        return point_rows, line_rows
    
    def _evaluateBDpoints(self,mainNumVars):
        # These are all MBD variables
        x0 = []
        v0 = []
        BDvel = self._mbd.BDvel_func
        BDpos = self._mbd.BDpoints_func
        # CGvel = self._mbd.CGpoints_func(*mainNumVars)
        
        for i in self.BDpointMap:
            body    = i['body']
            localID = i['localID']
            
            # This is relative to CG
            # x0point = np.array(i['r2CG'])
            # v0point =  BDvel[body](*numericalValuesTuple[0])[localID] - CGvel[body-1]
            
            # These are absolute coordinates
            x0point = BDpos[body](*mainNumVars)[localID]
            v0point = BDvel[body](*mainNumVars)[localID]
            
            # Make the vector 3D why y-component 0
            x0point = np.insert(x0point,1,0)
            v0point = np.insert(v0point, 1, 0)
            
            x0.append(x0point)
            v0.append(v0point)
        
        return np.hstack(x0), np.hstack(v0)
