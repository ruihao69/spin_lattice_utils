# %%
import numpy as np
from numpy.typing import NDArray

from spin_lattice_utils.interaction_scheme import InteractionScheme

from abc import ABC, abstractclassmethod
from typing import Tuple

class ParamsBase(ABC):
    @abstractclassmethod
    def get_Hs(self) -> NDArray[np.complex128]:
        pass
    
    @abstractclassmethod
    def get_Q(self) -> NDArray[np.complex128]:
        pass
    
    @abstractclassmethod
    def get_alphas(self) -> Tuple[float, float, float]:
        pass
    
    def get_Q1(self) -> NDArray[np.complex128]:
        return self.get_Q()[np.newaxis]
    
    def get_Q2(self) -> NDArray[np.complex128]:
        return self.get_Q()[np.newaxis, np.newaxis]
    
    @abstractclassmethod 
    def get_interaction_scheme(self) -> InteractionScheme:
        pass
# %%
