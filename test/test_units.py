# BEGIN: Test Cases
from spin_lattice_utils.units import energy_in_meV, energy_in_kelvin, energy_in_wave_number, meV_to_energy, wave_number_to_energy, kelvin_to_energy, ENERGY_UNIT_IN_WAVE_NUMBER, ENERGY_UNIT_IN_GIGAHERTZ, energy_in_GHz, GHz_to_energy

import unittest

class TestUnits(unittest.TestCase):
    places: int = 5
    def test_energy_in_wave_number(self):
        self.assertAlmostEqual(energy_in_wave_number(1.0), 3.3333333333333335, places=self.places)

    def test_wave_number_to_energy(self):
        self.assertAlmostEqual(wave_number_to_energy(3.3333333333333335), 1.0, places=self.places)

    def test_energy_in_meV(self):
        self.assertAlmostEqual(energy_in_meV(1.0), 0.413280857716144, places=self.places)

    def test_meV_to_energy(self):
        self.assertAlmostEqual(meV_to_energy(0.413280857716144), 1.0, places=self.places)

    def test_energy_in_kelvin(self):
        self.assertAlmostEqual(energy_in_kelvin(1.0), 4.7959233333333335, places=self.places)
        
    def test_kelvin_to_energy(self):
        self.assertAlmostEqual(kelvin_to_energy(298), 207.11986657605271 / ENERGY_UNIT_IN_WAVE_NUMBER, places=self.places-2)
        
    def test_energy_in_GHz(self):
        self.assertAlmostEqual(energy_in_GHz(1.0), 100.0, places=self.places)
        
    def test_GHz_to_energy(self):
        self.assertAlmostEqual(GHz_to_energy(100.0), 1.0, places=self.places)
        
    
    
if __name__ == "__main__":
    unittest.main()