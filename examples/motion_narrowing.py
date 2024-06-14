# %%
from spin_lattice_utils.deom_ctrls import DEOM_CTRLS
from spin_lattice_utils.motional_narrowing import MotionNarrowing
from spin_lattice_utils.gen_deom_inp_bo import get_deom_inp_bo, write_json

from dataclasses import dataclass

@dataclass(frozen=True)
class GlobalParameters:
    init_state: str = "spin_up"
    dt: float = 0.01
    tf: float = 100.0

    Delta_in_GHz: float = 10.0
    
def main(Lambda_corr: float = 0.01, Delta_corr: float = 0.1):
    gp = GlobalParameters() 
    
    mn = MotionNarrowing(
        Delta_corr=Delta_corr,
        Lambda_corr=Lambda_corr
    )
    
    ctrl = DEOM_CTRLS(
        tf=gp.tf,
        dt=gp.dt,
        init_state=gp.init_state
    )
    
    inp, err = get_deom_inp_bo(
        params=mn,
        ctrls=ctrl,
        λ=None,
        beta=1.0,
        nind=[2, 1]
    )
    
    # write input json to file
    write_json("./motional_narrowing", inp)
    
    print(f"{mn.get_Hs()}")
    print(f"{mn=}")
    print(f"{mn.get_Q()}")
    
# %%
if __name__ == "__main__":
    main()
        
    
     
    

# %%
