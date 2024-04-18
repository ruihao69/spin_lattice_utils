import numpy as np

from spin_lattice_utils.deom_ctrls import DEOM_CTRLS
from spin_lattice_utils.params_base import ParamsBase
from spin_lattice_utils.one_spin import SpinLatticeParamsOneSpin
from spin_lattice_utils.two_spins import SpinLatticeParamsTwoSpin
from spin_lattice_utils.spectral_functions import BrownianSpectralFunction, decompose_SpectralFunction
from spin_lattice_utils.third_party.deom import complex_2_json, init_qmd, init_qmd_quad, convert
from spin_lattice_utils.third_party.prony import bose_function
from spin_lattice_utils.spectral_functions import estimate_error

import os
import json
from typing import Tuple, Union

def get_init_state_one(slp: SpinLatticeParamsOneSpin, initial_state: str) -> str:
    if initial_state == 'spin_up':
        return 0
    elif initial_state == 'spin_down':
        return 1
    else:
        raise ValueError("The initial state must be either 'spin_up' or 'spin_down'.")
    
def get_init_state_two(slp: SpinLatticeParamsTwoSpin, initial_state: str) -> str:
    raise NotImplementedError("The method get_init_state_two is not implemented yet.")

def get_initial_state(slp: ParamsBase, initial_state: str) -> str:
    if isinstance(slp, SpinLatticeParamsOneSpin):
        return get_init_state_one(slp, initial_state)
    elif isinstance(slp, SpinLatticeParamsTwoSpin):
        return get_init_state_two(slp, initial_state)
    else:
        raise ValueError("The input parameter must be an instance of SpinLatticeParamsOneSpin or SpinLatticeParamsTwoSpin.")
    
def get_NSYS(slp: ParamsBase) -> int:
    if isinstance(slp, SpinLatticeParamsOneSpin):
        return 2
    elif isinstance(slp, SpinLatticeParamsTwoSpin):
        return 4
    else:
        raise ValueError("The input parameter must be an instance of SpinLatticeParamsOneSpin or SpinLatticeParamsTwoSpin.")

def get_inp_json(
    slp: ParamsBase,
    ctrl: DEOM_CTRLS,
    etal: np.ndarray,
    etar: np.ndarray,
    etaa: np.ndarray,
    expn: np.ndarray,
    is_kernel: bool,
):
    json_init = {
        "nmax": ctrl.nmax,
        "lmax": ctrl.lmax,
        "ferr": ctrl.ferr,
        "filter": True,
        "nind": len(expn),
        "nmod": 1,
        "inistate": get_initial_state(slp, ctrl.init_state),
        "equilibrium": {
            "sc2": False,
            "dt-method": True,
            "ti": 0,
            "tf": ctrl.tf,
            "dt": ctrl.dt,
            "backup": True,
        },
        "expn": complex_2_json(expn),
        "ham1": complex_2_json(slp.get_Hs()),
        "coef_abs": complex_2_json(etaa)
    }

    if is_kernel:
        kernel = {
            "kernel": True,
            "population": True,
            "system": False,
        }
        # append kernel into json_init
        json_init["kernel"] = kernel

    mode = np.zeros_like(expn, dtype=int)
    qmds = slp.get_Q1()
    
    NSYS = get_NSYS(slp)
    
    if slp.get_interaction_scheme() == 'quadratic':
        alpha0, alpha1, alpha2 = slp.get_alphas()
        qmd2 = slp.get_Q2()
        renorm = complex_2_json(
            (alpha0 + alpha2 * np.sum(etal)) * slp.get_Q1())

        quadratic = {
            "alp0": alpha0,
            "alp1": alpha1,
            "alp2": alpha2,
            "renormalize": renorm,
        }
        json_init.update(quadratic)
        init_qmd(json_init, qmds, qmds, mode, NSYS, etaa, etal, etar)
        init_qmd_quad(json_init, qmd2, qmd2, qmd2, mode, NSYS, len(expn),
                      len(mode), etaa, etal, etar)
    else:
        init_qmd(json_init, qmds, qmds, mode, NSYS, etaa, etal, etar)

    return json_init


def linspace_log(lb, ub, n):
    out = np.linspace(np.log(lb), np.log(ub), n)
    return np.exp(out)

def write_json_one_spin(dir: str, json_init: dict) -> None:
    # it is a kernel job if there's a kernel key in json_init
    is_kernel = "kernel" in json_init
    if is_kernel:
        inistate = json_init["inistate"]
        if inistate == 0:
            fn = os.path.join(dir, "00.input.json")
        else:
            fn = os.path.join(dir, "01.input.json")
    else:
        fn = os.path.join(dir, "input.json")

    with open(fn, 'w') as f:
        json.dump(json_init, f, indent=2, default=convert)
        
def write_json_two_spin(dir: str, json_init: dict) -> None:
    raise NotImplementedError("The method write_json_two_spin is not implemented yet.")


def write_json(dir: str, json_init: dict, slp: ParamsBase) -> None:
    if isinstance(slp, SpinLatticeParamsOneSpin):
        write_json_one_spin(dir, json_init)
    elif isinstance(slp, SpinLatticeParamsTwoSpin):
        write_json_two_spin(dir, json_init)
    

def get_deom_inp_bo(
    params: ParamsBase,
    ctrls: DEOM_CTRLS,
    λ: float,
    beta: float,
    nind: Union[int, Tuple[int, int]],
    prony_tf: int = 50,
    zeta: float = 1.0,
    OmegaB: float = 3.0,
    is_kernel: bool = False,
) -> Tuple[dict, float]:
    bo_spectral = BrownianSpectralFunction.initialize(λ, beta, zeta, OmegaB, prony_tf)
    etal, etar, etaa, expn = decompose_SpectralFunction(bo_spectral, nind)
    inp = get_inp_json(params, ctrls, etal, etar, etaa, expn, is_kernel)
    
    def stat(w):
        return bose_function(w, beta=beta, mu=0)
        
    err = estimate_error(bo_spectral.get_func(), stat, etal, expn)
    return inp, err
    