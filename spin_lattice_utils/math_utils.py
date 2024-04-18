import numpy as np
from scipy.linalg import kron

def sigma_z():
    return np.diagflat([1+0j, -1+0j]).astype(np.complex128)

def sigma_x():
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)

def sigma_y():
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

def I():
    return np.eye(2, dtype=np.complex128)

def SzSz(gamma: float):
    sz = sigma_z()
    id = I()
    
    sz1 = kron(sz, id)
    sz2 = kron(id, sz)
    return -gamma * np.dot(sz1, sz2)

def SdotS(gamma: float):
    sx = sigma_x()
    sy = sigma_y()
    sz = sigma_z()
    sigma_list = [sx, sy, sz]
    id = I()
    
    result = np.zeros((4, 4), dtype=np.complex128)
    for s in sigma_list:
        s1 = kron(s, id)
        s2 = kron(id, s)
        result += np.dot(s1, s2)
    return -gamma * result