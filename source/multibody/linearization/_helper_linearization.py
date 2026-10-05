import numpy as np

# ---------------- space detection ----------------
def detect_space(A: np.ndarray, nb: int, nq: int) -> str:
    ncart   = 3 * int(nb)
    sh      = A.shape[-2:]

    if sh == (ncart, ncart):
        return "global"
    if sh == (nq, nq):
        return "joint"
    raise ValueError(f"Cannot infer space from trailing shape {sh} (expected {(nq,nq)} or {(ncart,ncart)}).")

def as_col(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    if v.ndim == 1:
        return v.reshape(-1, 1)
    if v.ndim == 2 and v.shape[1] == 1:
        return v
    if v.ndim == 3 and v.shape[-1] == 1:
        return v
    raise ValueError(f"Cannot coerce shape {v.shape} to column form.")

def broadcast_omega_mat(A: np.ndarray, n_omega: int) -> np.ndarray:
    A = np.asarray(A)
    if A.ndim == 2:
        return np.repeat(A[None, :, :], n_omega, axis=0)
    if A.ndim == 3:
        if A.shape[0] != n_omega:
            raise ValueError(f"omega dimension mismatch: got {A.shape[0]} expected {n_omega}")
        return A
    raise ValueError(f"Bad matrix shape {A.shape}")

def broadcast_omega_vec(v: np.ndarray, n_omega: int) -> np.ndarray:
    v = np.asarray(v)
    if v.ndim == 2:
        return np.repeat(v[None, :, :], n_omega, axis=0)
    if v.ndim == 3:
        if v.shape[0] != n_omega:
            raise ValueError(f"omega dimension mismatch: got {v.shape[0]} expected {n_omega}")
        return v
    raise ValueError(f"Bad vector shape {v.shape}")
