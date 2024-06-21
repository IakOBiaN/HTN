import Scripts.MainScripts as ms

#ising model
calc = ms.CalcConfig()
calc.model = "ising"
calc.lattice = "diamond"
calc.coord = 3

#model params
calc.constant = 1
T = 1.0
J = 1.0
h = 1.0
for T in ms.np.arange(1.8, 2.6, 0.01):
	m_par = [h, -J, 0, 0, 0, 0]
	result = ms.heat_capasity(calc, T, m_par)
	print(T, result)
