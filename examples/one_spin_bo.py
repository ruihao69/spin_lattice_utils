# %%
from spin_lattice_utils.deom_ctrls import DEOM_CTRLS
from spin_lattice_utils.one_spin import SpinLatticeParamsOneSpin
from spin_lattice_utils.units import GHz_to_energy, kelvin_to_energy
from spin_lattice_utils.gen_deom_inp_bo import get_deom_inp_bo, write_json

from dataclasses import dataclass

@dataclass(frozen=True)
class GlobalParameters:
    init_state: str = "spin_up"
    dt: float = 0.01
    tf: float = 100.0

    Delta_in_GHz: float = 10.0


def main(temperature_in_kelvin: float = 10.0, lambd: float = 0.003):
    # get global parameters
    gp = GlobalParameters()

    # get variables
    beta = 1 / kelvin_to_energy(temperature_in_kelvin)

    # prepare objects from global parameters and variables
    ctrl = DEOM_CTRLS(
        tf=gp.tf,
        dt=gp.dt,
        init_state=gp.init_state
    )

    slp = SpinLatticeParamsOneSpin(
        Delta=GHz_to_energy(gp.Delta_in_GHz),
    )

    # generate input json
    inp, err = get_deom_inp_bo(
        params=slp,
        ctrls=ctrl,
        λ=lambd,
        beta=beta,
        nind=[2, 1],
    )

    # write input json to file
    write_json("./tmp_one", inp)


    print(f"{slp.get_Hs()}")
    print(f"{slp=}")
    print(f"{slp.get_Q()}")

# %%
if __name__ == "__main__":
    main()
# %%
