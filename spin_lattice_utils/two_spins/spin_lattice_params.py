# %%
import numpy as np
from numpy.typing import NDArray

from spin_lattice_utils.params_base import ParamsBase
from spin_lattice_utils.interaction_scheme import InteractionScheme
from spin_lattice_utils.math_utils import sigma_z, sigma_x, I, kron, SzSz, SdotS
from spin_lattice_utils.two_spins.alignment import Alignment
from spin_lattice_utils.two_spins.spin_spin_type import SpinSpinType

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class SpinLatticeParamsTwoSpin(ParamsBase):
    Delta: float = 1.0
    coherence: float = 0.0
    V_sph_diag: float = 0.0
    V_sph_offd: float = 1.0
    gamma: float = 0.1
    alignment: Alignment = Alignment.Parallel 
    spin_spin_type: SpinSpinType = SpinSpinType.SzSz
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
        # get constant operators
        sz = sigma_z()
        id = I()
        
        # get system Hamiltonian
        sz1 = kron(0.5 * self.Delta * sz, id)
        sz2 = kron(id, 0.5 * self.Delta * sz) if self.alignment == Alignment.Parallel else kron(id, -sz)
        Hs_0 = sz1 + sz2
        
        # get spin-spin interaction Hamiltonian
        if self.spin_spin_type == SpinSpinType.SzSz:
            Hs_int = SzSz(gamma=self.gamma)
        elif self.spin_spin_type == SpinSpinType.SdotS:
            Hs_int = SdotS(gamma=self.gamma)
        else:
            raise ValueError(f"The spin-spin interaction is neither SzSz or SdotS. Got {self.spin_spin_type}.")
        
        return Hs_0 + Hs_int
    
    def get_Q(self) -> NDArray[np.complex128]:
        sx = sigma_x()
        id = I()
        
        Q = kron(sx, id) + kron(id, sx)
        return Q
    
    def get_alphas(self):
        return self.alpha0, self.alpha1, self.alpha2
    
    def get_interaction_scheme(self) -> InteractionScheme:
        return self.interaction_scheme

# %%
if __name__ == "__main__":
    params = SpinLatticeParamsTwoSpin()
    print(f"{params.get_Hs()}")
    print(f"{params=}")
    print(f"{params.get_Q()}")
    
# %%
