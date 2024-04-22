# %%
import numpy as np
import scipy.constants as const

ENERGY_UNIT_IN_GIGAHERTZ = 100.0
ENERGY_UNIT_IN_WAVE_NUMBER = 10.0 / 3

# helper functions
def _wave_number_to_meV(wave_number: float) -> float:
    joules = wave_number * const.h * const.c * 100
    joules_to_meV = 1e3 / const.e
    return joules * joules_to_meV

def _wave_number_to_kelvin(wave_number: float) -> float:
    joules = wave_number * const.h * const.c * 100
    joules_to_kelvin = 1 / const.k
    return joules * joules_to_kelvin

    
# functions for unit conversion

def energy_in_wave_number(energy: float) -> float:
    return energy * ENERGY_UNIT_IN_WAVE_NUMBER

def wave_number_to_energy(wave_number: float) -> float:
    return wave_number / ENERGY_UNIT_IN_WAVE_NUMBER

def meV_to_energy(meV: float) -> float:
    transform = energy_in_meV(1)
    return meV / transform

def energy_in_meV(energy: float) -> float:
    energy_in_wn = energy_in_wave_number(energy)
    return _wave_number_to_meV(energy_in_wn)

def kelvin_to_energy(kelvin: float) -> float:
    transform = energy_in_kelvin(1)
    return kelvin / transform

def energy_in_kelvin(energy: float) -> float:
    energy_in_wn = energy_in_wave_number(energy)
    return _wave_number_to_kelvin(energy_in_wn)

def energy_in_GHz(energy: float) -> float:
    return energy * ENERGY_UNIT_IN_GIGAHERTZ

def GHz_to_energy(GHz: float) -> float:
    return GHz / ENERGY_UNIT_IN_GIGAHERTZ

def unit_time_to_picosecond(time: float) -> float:
    raise NotImplemented


def test():
    energy = 1.0
    assert np.allclose(energy_in_wave_number(energy), 3.3333333333333335)
    assert np.allclose(wave_number_to_energy(3.3333333333333335), 1.0)
    assert np.allclose(energy_in_meV(energy), 0.413280857716144)
    assert np.allclose(meV_to_energy(0.413280857716144), 1.0)
    assert np.allclose(energy_in_kelvin(energy), 4.795954320551766)
    assert np.allclose(kelvin_to_energy(298), 207.11986657605271 / ENERGY_UNIT_IN_WAVE_NUMBER, rtol=1e-5)
    assert np.allclose(energy_in_GHz(energy), 100.0)
    assert np.allclose(GHz_to_energy(100.0), 1.0)

# %%    
if __name__ == "__main__":
    test()

# %%
