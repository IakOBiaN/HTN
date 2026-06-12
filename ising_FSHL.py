import numpy as np

import htn

# Ising model on the folded square hierarchical lattice (FSHL).
calc = htn.CalcConfig()
calc.model = "ising"
calc.lattice = "FSHL"
calc.metParam = 1  # p parameter of FSHL
calc.constant = 1

J = 1.0
h = 1.0
print("Temperature, Heat_capacity")
for T in np.arange(1.8, 2.6, 0.01):
    m_par = [h, -J, 0, 0, 0, 0]
    result = htn.thermodynamics(calc, T, m_par, heat_capacity=True)
    print(T, result["heat_capacity"])
