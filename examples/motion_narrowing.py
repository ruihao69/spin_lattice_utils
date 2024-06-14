# %%
from spin_lattice_utils.deom_ctrls import DEOM_CTRLS
from spin_lattice_utils.motional_narrowing import MotionNarrowing
from spin_lattice_utils.gen_deom_inp_bo import get_deom_inp_bo, write_json
from spin_lattice_utils.units import GHz_to_energy, kelvin_to_energy

from dataclasses import dataclass

@dataclass(frozen=True)
class GlobalParameters:
    init_state: str = "spin_up"
    dt: float = 0.03
    tf: float = 1000.0

    Delta_in_GHz: float = 100.0

def main(Lambda_corr: float = 0.01, Delta_corr: float = 0.1):
    gp = GlobalParameters()

    mn_up = MotionNarrowing(
        Delta_corr=Delta_corr,
        Lambda_corr=Lambda_corr,
        Delta=GHz_to_energy(gp.Delta_in_GHz),
        init_state="spin_up"
    )
    
    mn_dn = MotionNarrowing(
        Delta_corr=Delta_corr,
        Lambda_corr=Lambda_corr,
        Delta=GHz_to_energy(gp.Delta_in_GHz),
        init_state="spin_down"
    )

    ctrl = DEOM_CTRLS(
        tf=gp.tf,
        dt=gp.dt,
        init_state=gp.init_state
    )

    inp00, err = get_deom_inp_bo(
        params=mn_up,
        ctrls=ctrl,
        λ=None,
        beta=1.0,
        nind=[2, 1],
        is_kernel=True
    )
    
    inp01, err = get_deom_inp_bo(
        params=mn_dn,
        ctrls=ctrl,
        λ=None,
        beta=1.0,
        nind=[2, 1],
        is_kernel=True
    )

    # write input json to file
    write_json("./motional_narrowing", inp00)
    write_json("./motional_narrowing", inp01)

# %%
if __name__ == "__main__":
    main()





# %%
