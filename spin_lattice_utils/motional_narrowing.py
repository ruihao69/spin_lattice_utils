# %%
import numpy as np

from spin_lattice_utils.params_base import ParamsBase
from spin_lattice_utils.interaction_scheme import InteractionScheme
from spin_lattice_utils.math_utils import sigma_z, sigma_x
from spin_lattice_utils.third_party.deom import sort_symmetry

from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class MotionNarrowing(ParamsBase):
    """
    Class to simulate the effect of motion narrowing.
    """
    Delta_corr: float
    Lambda_corr: float
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
        
    def get_Hs(self) -> np.ndarray:
        sz = sigma_z()
        return 0.5 * self.Delta * sz
    
    def get_Q(self) -> np.ndarray:
        sx = sigma_x()
        sz = sigma_z()
        return 0.5 * self.V_sph_diag * sz + self.V_sph_offd * sx 
    
    def get_alphas(self):
        return self.alpha0, self.alpha1, self.alpha2
    
    def get_interaction_scheme(self) -> InteractionScheme:
        return self.interaction_scheme
    
    def get_rho0(self) -> np.ndarray:
        return np.array([[1, 0], [0, 0]])
    
    def get_exponentials(self ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        etal, expn = np.array([self.Delta_corr**2]), np.array([self.Lambda_corr])
        return sort_symmetry(etal, expn)
    
    def get_initial_state(self) -> str:
        return 0