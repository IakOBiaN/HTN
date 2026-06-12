import numpy as np

import htn

# Ising model on a diamond hierarchical lattice: heat capacity vs temperature.
calc = htn.CalcConfig()
calc.model = "ising"
calc.lattice = "diamond"
calc.coord = 3
calc.constant = 1

J = 1.0
h = 1.0
print("Temperature, Heat_capacity")
for T in np.arange(1.8, 2.6, 0.01):
    m_par = [h, -J, 0, 0, 0, 0]
    result = htn.thermodynamics(calc, T, m_par, heat_capacity=True)
    print(T, result["heat_capacity"])
