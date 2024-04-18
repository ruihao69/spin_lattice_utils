import numpy as np
from numpy.typing import NDArray

from spin_lattice_utils.params_base import ParamsBase
from spin_lattice_utils.interaction_scheme import InteractionScheme
from spin_lattice_utils.math_utils import sigma_z

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class SpinLatticeParamsOneSpin(ParamsBase):
    Delta: float = 1.0
    coherence: float = 0.0
    V_sph_diag: float = 0.0
    V_sph_offd: float = 1.0
    interaction_scheme: InteractionScheme = InteractionScheme.LINEAR
    alpha0: Optional[float] = None  
    alpha1: Optional[float] = None
    alpha2: Optional[float] = None
    
    def __post_init__(self):
        if self.interaction_scheme == InteractionScheme.QUADRATIC:
            if self.alpha0 is None or self.alpha1 is None or self.alpha2 is None:
                raise ValueError("Please input alpha0, alpha1, and alpha2 for quadratic interaction scheme.")
        elif self.interaction_scheme == InteractionScheme.LINEAR:
            if self.alpha0 is not None or self.alpha1 is not None or self.alpha2 is not None:
                raise ValueError("Please do not input alpha0, alpha1, and alpha2 for linear interaction scheme.")
        else:
            raise ValueError("Please input interaction_scheme `linear` or `quadratic`.")
        
    def get_Hs(self) -> NDArray[np.complex128]:
        return self.Delta / 2 * sigma_z() + np.array([[0, self.coherence], [np.conjugate(self.coherence), 0]], dtype=np.complex128) 
    
    def get_Q(self) -> NDArray[np.complex128]:
        return np.array([[0.0, self.V_sph_offd], [np.conjugate(self.V_sph_offd), self.V_sph_diag]], dtype=np.complex128)
    
    def get_Q1(self) -> NDArray[np.complex128]:
        return self.get_Q()[np.newaxis]
    
    def get_Q2(self) -> NDArray[np.complex128]:
        return self.get_Q()[np.newaxis, np.newaxis]
    
    def get_alphas(self):
        return self.alpha0, self.alpha1, self.alpha2
    
    def get_interaction_scheme(self) -> InteractionScheme:
        return self.interaction_scheme