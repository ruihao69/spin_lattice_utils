import sympy as sp
import numpy as np
from numpy.typing import NDArray, ArrayLike

from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class BoseSpectralFunction:
    omega: sp.Symbol
    spectral_function: sp.Expr
    sp_para_dict: dict
    condition_dict: dict
    para_dict: dict
    decompose_method: str = "Matsubara"
    _func: Optional[sp.Expr] = None
    
    def as_real_imag(self) -> Tuple[sp.Expr, sp.Expr]:
        return self.spectral_function.as_real_imag()
    
    def J(self, omega: ArrayLike) -> NDArray[np.float64]:
        if self._func is None:
            self._func = self.get_func()
        
        return self._func(np.array(omega, dtype=np.float64))
    
    def get_func(self) -> sp.Expr:
        _, imag_part = self.as_real_imag()
        imag_part_subs = imag_part.subs(self.sp_para_dict)
        return sp.lambdify(self.omega, imag_part_subs, modules="numpy")
    
    def get_prony_tf(self) -> int:
        raise NotImplementedError("The method get_prony_tf is not implemented yet.")