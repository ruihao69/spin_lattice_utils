import numpy as np
import sympy as sp

from spin_lattice_utils.third_party.deom.deom import decompose_spe
from spin_lattice_utils.third_party.prony import TimeDomainData 
from spin_lattice_utils.third_party.prony import bose_function, fermi_function, get_spectral_function_from_exponentials
from spin_lattice_utils.third_party.prony import prony
from spin_lattice_utils.spectral_functions.bose_spectral_function import BoseSpectralFunction

import warnings
from typing import Union

# util functions

def fmt_cnumber(cnumber):
    return np.format_float_scientific(cnumber.real, precision=8) + " + 1.j * " + np.format_float_scientific(cnumber.imag, precision=8)

def numpy_float_to_set(a: np.ndarray):
    """Transform a numpy float array to a set
    Force 8 precision to avoid rounding error
    """
    return set(map(fmt_cnumber, a))

def remove_from_set(s, element):
    s.remove(fmt_cnumber(element))

def extract_conjugate_pairs(expn):
    expn_set = numpy_float_to_set(expn)
    if len(expn) > len(expn_set):
        raise ValueError("There is duplicated items in your exponent list. This does not fit in the context of spectral function decomposition. Double check your spectral decomposition data!!!")

    expn_conjugate_pairs = []
    expn_non_conjugate_pairs = []

    for ii, e in enumerate(expn):
        conjugate = np.conj(e)
        if np.isreal(e):
            expn_non_conjugate_pairs.append((ii, e))
            remove_from_set(expn_set, e)
        elif fmt_cnumber(conjugate) in expn_set:
            expn_conjugate_pairs.append((ii, e))
            jj = np.where(np.abs((expn-conjugate)/conjugate) < 1e-8)[0][0]
            expn_conjugate_pairs.append((jj, conjugate))
            remove_from_set(expn_set, e)
            remove_from_set(expn_set, conjugate)
        elif fmt_cnumber(e) in expn_set:
            expn_non_conjugate_pairs.append((ii, e))
            remove_from_set(expn_set, e)
        else:
            pass
    return expn_conjugate_pairs, expn_non_conjugate_pairs

def get_symmetrized_deom_inputs(etal: np.ndarray, expn: np.ndarray):
    expn_conjugate_pairs, expn_non_conjugate_pairs = extract_conjugate_pairs(expn)

    if expn_conjugate_pairs:
        arg_pair, expn_pair = list(map(np.array, zip(*expn_conjugate_pairs)))
        etal_pair = etal[arg_pair]
        etar_pair = np.flip(np.conj(etal_pair.reshape(2, -1))).flatten()
    else:
        expn_pair = np.array([])
        etal_pair = np.array([])
        etar_pair = np.array([])

    if expn_non_conjugate_pairs:
        arg_non_pair, expn_non_pair = list(map(np.array, zip(*expn_non_conjugate_pairs)))
        etal_non_pair = etal[arg_non_pair]
        etar_non_pair = np.conj(etal_non_pair)
    else:
        expn_non_pair = np.array([])
        etal_non_pair = np.array([])
        etar_non_pair = np.array([])

    expn = np.append(expn_pair, expn_non_pair)
    etal = np.append(etal_pair, etal_non_pair)
    etar = np.append(etar_pair, etar_non_pair)
    etaa = np.abs(etal)

    return etal, etar, etaa, expn

def decompose_spe_prony_moonshine(spe: sp.Mul,
                                  w_sp: sp.Symbol,
                                  sp_para_dict: dict,
                                  para_dict: dict,
                                  condition_dict: dict,
                                  nind: Union[int, list],
                                  tf: int = 50,
                                  bose_fermi=1):
    # generate time domain data
    spe_imag = spe.as_real_imag()[1]
    gen_jw = sp.lambdify(w_sp, spe_imag.subs(sp_para_dict))
    if bose_fermi == 1:
        def stat(E):
            return bose_function(E, beta=para_dict["beta"])
        data = TimeDomainData(gen_jw, bose_function, para_dict["beta"], tf=tf)
    elif bose_fermi == 2:
        def stat(E):
            return fermi_function(E, beta=para_dict["beta"])
        data = TimeDomainData(gen_jw, fermi_function, para_dict["beta"], tf=tf)
    else:
        raise ValueError("You request neither bose or fermi.")

    expn, etal = prony(data, nind[0], nind[1])

    error = estimate_error(gen_jw, stat, etal, expn, )
    print(f"The estimated error for prony fitting is {error}")

    return get_symmetrized_deom_inputs(etal, expn)

def estimate_error(gen_jw, stat, etal, expn):
    len_ = 30000
    spe_wid = 100

    w = np.append(np.linspace(-spe_wid, 0, len_), np.linspace(0, spe_wid, len_))
    jw_prony = get_spectral_function_from_exponentials(w, expn, etal).real
    jw_exact = gen_jw(w) * stat(w)

    diff = np.nanmax(np.abs(jw_exact - jw_prony))

    return diff

def is_valid_nind(l: Union[list, int]):
    if isinstance(l, list):
        if len(l) != 2:
            raise ValueError(
                "the input nind need to have 2 elements. precisely")
        if all(isinstance(elem, int) for elem in l):
            return True
        else:
            raise ValueError(
                "the elements in the input nind list must be integers.")
    elif isinstance(l, int):
        return True
    else:
        raise ValueError("the input nind is neither a list of int or int.")

def decompose_SpectralFunction(sp: BoseSpectralFunction, nind: list):
    try:
        if is_valid_nind(nind):
            pass
        else:
            print("Invalid NIND")
    except ValueError as e:
        raise RuntimeError(e)
    
    spe, w, sp_para_dict, para_dict, condition_dict = sp.spectral_function, sp.omega, sp.sp_para_dict, sp.para_dict, sp.condition_dict

    if sp.decompose_method == 'prony':
        return decompose_spe_prony_moonshine(spe, w, sp_para_dict, para_dict, condition_dict, nind, tf=sp.get_prony_tf(), bose_fermi=1)
    elif sp.decompose_method == 'pade':
        return decompose_spe(spe, w, sp_para_dict, para_dict, condition_dict, nind, pade=1, bose_fermi=1)
    else:
        warnings.warn(
            f"Your decompose method is {sp.decompose_method}, which is neither prony or pade. Use pade by default."
        )
        return decompose_spe(spe, w, sp_para_dict, para_dict, condition_dict, nind, pade=1, bose_fermi=1)
