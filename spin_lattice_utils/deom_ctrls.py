from dataclasses import dataclass

@dataclass(frozen=True)
class DEOM_CTRLS:
    tf: float
    dt: float
    init_state: str # either spin_up or spin_down
    nmax: int = 1000000
    lmax: int = 50  
    ferr: float = 1e-12