# %%

import sympy as sp

from spin_lattice_utils.spectral_functions.bose_spectral_function import BoseSpectralFunction

from dataclasses import dataclass

@dataclass
class BrownianSpectralFunction(BoseSpectralFunction):
    prony_tf: int = 50
    
    @classmethod
    def initialize(cls, λ: float, beta: float, zeta: float = 1.0, OmegaB: float = 3.0, tf: int = 50):
        omega, lamd, zeta_sp, omgs = sp.symbols(r"\omega , \lambda, \zeta, \Omega_{B}", real=True)
        nominator = 2 * omgs * omgs * lamd
        denominator = (omgs * omgs - omega * omega - zeta_sp * sp.I * omega)
        
        sp_para_dict = {lamd: λ, omgs: OmegaB, zeta_sp: zeta}
        condition_dict = {}
        para_dict = {'beta': beta}
        decompose_method = 'prony'
        
        return cls(
            omega=omega, 
            spectral_function=nominator / denominator, 
            sp_para_dict=sp_para_dict, 
            condition_dict=condition_dict, 
            para_dict=para_dict, 
            decompose_method=decompose_method, 
            prony_tf=tf
        )
    
    def get_prony_tf(self) -> int:
        return self.prony_tf
    
# %%
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    brownian = BrownianSpectralFunction.initialize(0.5, 1.0)
    print(brownian)
    w = np.linspace(-10, 10, 1000)
    jw = brownian.J(w)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(w, jw)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$J(\omega)$")
    plt.show()
    
# %%
