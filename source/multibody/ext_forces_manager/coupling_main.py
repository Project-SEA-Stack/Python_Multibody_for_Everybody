# multibody/ExtForcesManager.py
from __future__ import annotations
from collections import namedtuple
import numpy as np
import numpy.typing as npt

CartForce = namedtuple("CartForce", ["pType","body","localID", "vector"])   # 3-component vec

class ExternalForcesManager:
    """
    Orchestrates several *adapters* that talk to third-party libraries
    (MoorDyn, FAST, CFD, …).  Each adapter must implement:

        .initialize(t0, q0, qd0)           # optional
        .update(t, q, qd)                  # advance internal state
        .forces() -> list[CartForce]       # Cartesian forces in body frames
    """
    # ------------------------------------------------------------------
    def __init__(self, mbd_sys):
        self.mbd     = mbd_sys
        self.adapters: dict[str, object] = {}           # registered ExternalAdapters
        self.R_func  = mbd_sys.R_func      # numeric λ(Q,…) for velocity matrix
        self.xfmFlag = 0

    # ------------------------------------------------------------------
    def register(self, adapter) -> None:
        """Attach an adapter instance that follows the contract above."""
        
        try:
            key = adapter.name
        except AttributeError:
            raise AttributeError("Every adapter must define a unique 'name' attribute.")
            
        if key in self.adapters:
            raise KeyError(f"An adapter named '{key}' is already registered.")
        self.adapters[key] = adapter
        
    def get_adapter(self, name: str):
        """
        Return the adapter object previously registered under `name`.
        Raises KeyError if no such adapter exists.
        """
        return self.adapters[name]


    # ------------------------------------------------------------------
    def initialize(self, t0: float, q0: npt.ArrayLike, qd0: npt.ArrayLike):
        """ Call initialize method if needed for some reason (Optional)"""
        for ad in self.adapters.values():
            if hasattr(ad, "initialize"):
                ad.initialize(t0, q0, qd0)
                
    # ------------------------------------------------------------------
    def end(self):
        """Calls the `.end()` method on each adapter if it exists. (Optional)"""
        for ad in self.adapters.values():
            if hasattr(ad, "end"):
                ad.end()         
    
    # ------------------------------------------------------------------
    def init_plot(self):
        """Calls `.init_plot()` on each adapter if it exists. (Optional)"""
        for adapter in self.adapters.values():
            if hasattr(adapter, "init_plot"):
                adapter.init_plot()

    # ------------------------------------------------------------------
    def generalized_forces(self,
                           t: float,
                           main_num_vars: npt.ArrayLike) -> np.ndarray:
        """
        Generate the generalized‐coordinate force vector ΔRHS from external Cartesian forces.

        This method:
        
          1. Calls each registered adapter’s
             ``update(t, q, qd, main_num_vars)`` to advance its internal state.
          2. Retrieves Cartesian forces via ``adapter.forces()``, which must
             return a list of ``CartForce`` namedtuples with fields:

             - **pType** (str):  

               - ``'CG'`` for forces applied at a body’s center of gravity  
                 (no moment generated).  
               - ``'BD'`` for forces applied at a body‐defined point  
                 (moment computed about the y-axis).  
                 
             - **body** (int): 1-based child body index in the joint system.  
             - **localID** (int): index into  
               ``MBD.BDpoints_func[body]`` for ``'BD'``; ignored for ``'CG'``.  
             - **vector** (array_like of float, shape (2,)):  
               planar force components ``[Fx, Fz]`` in the local body frame.

          3. For each ``CartForce``, computes a 3-element Cartesian force/moment
             vector ``[Fx, Fz, moment]``, where
             ``moment = 0`` for ``'CG'``, and
             ``moment = - (rel_x*Fz - rel_z*Fx)`` for ``'BD'``.
          4. Assembles these into a global Cartesian vector ``Fcart`` of size
             ``(3 * n_bodies, 1)``.
          5. Applies the numeric velocity‐transformation matrix ``R_func``:
             ``Qgen = R_func(*main_num_vars).T @ Fcart``.

        Parameters
        ----------
        t : float
            Current time for which external forces should be evaluated.
        main_num_vars : array_like of float
            Concatenated numeric values of generalized coordinates (Q),
            velocities (QD), and any additional parameters required by
            the lambdified kinematic functions.

        Returns
        -------
        numpy.ndarray, shape (n_q,)
            The generalized‐force vector to add to the multibody RHS. If no
            external forces are provided, returns a zero vector.

        Notes
        -----
        - Adapters must be registered via ``register(adapter)`` and must implement:
          
            * ``update(t, q, qd, main_num_vars)``  
            * ``forces() -> list[CartForce]``  

        - ``CartForce`` is defined as:
          
            >>> CartForce = namedtuple("CartForce",
            ...                        ["pType", "body", "localID", "vector"])

        - The ``body`` index is 1-based and corresponds to the child index in
          ``JointSystem.joints``.
        - For ``pType=='BD'``, ``localID`` must reference a valid point from
          ``MBD.BDpoints_func[body]``.
        """
        # -- 1) gather all Cartesian forces ---------------------------------
        cart    = []                                             # type: list[CartForce]
        addedM  = []
        NDOF    = sum(self.mbd.NDOF)
        q       = main_num_vars[:NDOF]
        qd      = main_num_vars[NDOF:2*NDOF]
        
        for ad in self.adapters.values():
            ad.update(t, q, qd, main_num_vars)
            cart.extend(ad.forces())

            # Include inertia forces as added inertia if available
            if hasattr(ad, "added_inertia"):
                addedM.append(ad.added_inertia)

        if not cart:                      # nothing to add
            return np.zeros_like(q)

        nb    = len(self.mbd.joints)
        Fcart = np.zeros((3*nb, 1))
        
        for cf in cart:
            # TODO: logic for JO points not supported
            # Evaluate relative position of the point w.r.t CG
            if cf.pType == 'CG':
                moment = 0.
            elif cf.pType == 'BD':
                BD  = self.mbd.BDpoints_func[cf.body](*main_num_vars)
                CG  = self.mbd.CGpoints_func(*main_num_vars)
                rel = BD[cf.localID] - CG[cf.body-1]
                
                # moment around y-axis!! to be consistent with MBD
                moment = - (rel[0] * cf.vector[1] - rel[1] * cf.vector[0])

            forceFinal = cf.vector + np.array([0., 0., moment])
            idx = 3*(cf.body-1)
            Fcart[idx:idx+3, 0] += forceFinal

        # -- 2) transform to Q-space ---------------------------------------
        R_num = self.R_func(*main_num_vars)            # (3N × n_q)
        Qgen  = R_num.T @ Fcart                        # (n_q × 1)

        # -- 3) add added mass if any --------------------------------------
        Mred_add = np.zeros((NDOF, NDOF))
        if addedM:
            Madd        = np.sum(addedM, axis=0)              # (3N × 3N)
            Mred_add    += R_num.T @ Madd @ R_num

        return Qgen, Mred_add

    def _force2cart(F_vec, MBDsys):
        # Number of bodies
        N   = len(MBDsys.NDOF)
        if F_vec.shape[0] != 3*N:
            raise ValueError(f"_force2cart expected {3*N} numbers (3 per body), got {F_vec.shape[0]}.")
        
        arr = F_vec.reshape((N, 3))
        out = []
        
        for i in range(N):
            out.append(CartForce(pType='CG', body=i+1, localID=-1,
                                 vector=arr[i]))
        return out


