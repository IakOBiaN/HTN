import Scripts.MainScripts as ms

#ising model
calc = ms.CalcConfig()
calc.model = "binary"
calc.lattice = "diamond"
calc.coord = 3

#model params
T = 100.0

print("Chemical_potential, Density, Entropy, Susceptibility, Heat_capacity, Grand_potential")
for mu in ms.np.arange(-10.00, 40.01, 1.0):
	m_par = [mu, 10.0, 4.0, 6.0, 0, 0]
	result = ms.thermodynamics(
		calc, T, m_par,
		coverage=True, susceptibility=True, entropy=True, heat_capacity=True,
	)
	print(mu, result["coverage"], result["entropy"], result["susceptibility"],
		  result["heat_capacity"], result["grand_potential"])
