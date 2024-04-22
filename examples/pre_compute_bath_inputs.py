# %%
import numpy as np
from numpy.typing import ArrayLike

from spin_lattice_utils.third_party.prony import bose_function
from spin_lattice_utils.spectral_functions import BoseSpectralFunction, decompose_SpectralFunction, estimate_error, BrownianSpectralFunction
from spin_lattice_utils.units import kelvin_to_energy

import os
from typing import Tuple, Optional
import warnings

DECOMPOSITION_RTOL = 1e-5
DECOMPOSITION_ATOL = 3e-5

def get_nind(n_total_ind: int) -> Tuple[int]:
    possible_nind_list = []
    for ii in range(n_total_ind):
        possible_nind_list.append([ii, n_total_ind - ii])
    return possible_nind_list
    

def prony_fitting_single_shot(
    spectral_function: BoseSpectralFunction ,
) -> Tuple[np.ndarray, float]:
    # TOTAL_NIND_START = 4
    # MAX_NIND = 7
    
    
    TOTAL_NIND_START = 6
    MAX_NIND = 10
    
    total_nind = TOTAL_NIND_START
    
    def stat(w):
        return bose_function(w, beta=spectral_function.para_dict["beta"], mu=0)
    
    not_converged = True
    
    nind_config_list = []
    data_list = [] 
    error_list = []
    while not_converged:
        possible_nind_list = get_nind(total_nind)
        for nind in possible_nind_list:
            etal, etar, etaa, expn = decompose_SpectralFunction(
                spectral_function,
                nind=nind,
            )
            is_converged, diff = estimate_error(
                spectral_function.get_func(),
                stat,
                etal,
                expn,
                rtol=DECOMPOSITION_RTOL,
                atol=DECOMPOSITION_ATOL,
            )
            nind_config_list.append(nind)
            data_list.append([etal, etar, etaa, expn])
            error_list.append(diff)
            if is_converged:
                not_converged = False
                break
        if total_nind >= MAX_NIND:
            warnings.warn(f"Max nind reached: {MAX_NIND}")
            break
        total_nind += 1
        
    # finally, if we have not converged, we return the best fit
    if not_converged:
        best_fit = np.argmin(error_list)
        etal, etar, etaa, expn = data_list[best_fit]
        return etal, etar, etaa, expn, error_list[best_fit], nind_config_list[best_fit]
    else:
        return etal, etar, etaa, expn, diff, nind

def prony_fitting_BO_spectral_function(
    T_in_kelvin: float,
    λ: float,
    OmegaB: float=3.0,
    zeta: float=1.0,
    pron_tf: int=50,
) -> Tuple[np.ndarray, float]:
    bo_spectral = BrownianSpectralFunction.initialize(
        λ=λ,
        beta=1 / kelvin_to_energy(T_in_kelvin),
        zeta=zeta,
        OmegaB=OmegaB,
        tf=pron_tf,
    )
    
    res = prony_fitting_single_shot(bo_spectral)
    inp_arrays = np.array(res[:-2])
    err = res[-2]
    nind = res[-1]
    return inp_arrays, err, nind

def main(
    λ: float, 
    OmegaB: float=3.0,
    zeta: float=1.0,
    T_in_kelvin_array: Optional[ArrayLike]=None,
) -> None:
    project_prefix = f"BO-λ_{λ}-OmegaB_{OmegaB}-zeta_{zeta}"
    
    if T_in_kelvin_array is not None:
        T_in_kelvin_array = np.array(T_in_kelvin_array)   
    else:
        T_in_kelvin_array = np.arange(1, 11)
       
    data = {} 
    for T_in_kelvin in np.flip(T_in_kelvin_array):
        inp_arrays, err, nind = prony_fitting_BO_spectral_function(
            T_in_kelvin=T_in_kelvin,
            λ=λ,
            OmegaB=OmegaB,
            zeta=zeta,
        )
        etal, etar, etaa, expn = inp_arrays
        data[str(T_in_kelvin)] = {
            "etal": etal,
            "etar": etar,
            "etaa": etaa,
            "expn": expn,
            "nind_re": nind[0],
            "nind_im": nind[1],
            "error": err,
        }
        print(f"Exiting T_in_kelvin={T_in_kelvin} with error={err}")    
        np.savez(f"{project_prefix}-T_{T_in_kelvin}.npz", **data[str(T_in_kelvin)])
    if os.path.exists(f"{project_prefix}.npz"):
        _data_prev = np.load(f"{project_prefix}.npz", allow_pickle=True) 
        data_prev = {key: val.item() for key, val in _data_prev.items()}
        for key, val in data.items():
            if key in data_prev:
                import warnings
                warnings.warn(f"Key {key} already exists in the data.")
                if data_prev[key]["error"] > val["error"]:
                    data_prev[key] = val
            else:
                data_prev[key] = val
        np.savez(f"{project_prefix}.npz", **data_prev)
    else:
        np.savez(f"{project_prefix}.npz", **data)
    
# %%
if __name__ == "__main__":
    def get_T_list_seg1():
        return np.arange(1, 11)

    def get_T_list_seg2():
        return np.arange(12, 22, 2)
    
    #main(λ=0.003, T_in_kelvin_array=get_T_list_seg1())
    main(λ=0.003, T_in_kelvin_array=get_T_list_seg2())
                
# %%
import os
import re
import glob

data = np.load("BO-λ_0.003-OmegaB_3.0-zeta_1.0.npz", allow_pickle=True)



# T_list_str = [f"{T}" for T in get_T_list_seg1()] 
T_list_str = [f"{T}" for T in get_T_list_seg2()]

for T_str in T_list_str:
    # print(f"{data[T_str].item()['error']=}")
    print(f"{T_str=}, {data[T_str].item()['nind_re']=}, {data[T_str].item()['nind_im']=}")


# dat_pattern = "BO-λ_0.003-OmegaB_3.0-zeta_1.0-T_*.npz"
# dat_files = glob.glob(dat_pattern)  
# T = np.array([re.search(r"T_(\d+)", dat_file).group(1) for dat_file in dat_files]).astype(np.float64)
# argsort = np.argsort(T)
# data = [np.load(dat_file) for dat_file in dat_files]
# 
# for ii, dat in enumerate(data):
#     print(f"{T[ii]=}")
#     print(f"{dat['error']=}")
#     print(f"{dat['expn'].shape[0]=}")




# %%
