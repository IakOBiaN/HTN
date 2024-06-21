import numpy as np

inf = -1e8

def build_matrix (calc, temp, m_par):

	model = calc.model
	neigbours = calc.coord

	if len(m_par) < 10:
		m_par = m_par + [0.0] * (10 - len(m_par))

	models_dict = {
		"mono" : True,
		"binary" : True,
		"ising" : True
	}

	exist = models_dict.get(calc.model)
	assert (exist is not None), "Error! There is no such model in the database"

	#[right, bottom]
	matrixes = []
	if model == "mono":
		matrixes = [np.array([[0.0, m_par[0] / neigbours], [m_par[0] / neigbours, -m_par[1] + m_par[0] / (neigbours / 2.0)]]) ,] * 3
	elif model == "binary":
		#m_par: 0 - muA, 1 - muB, 2 - epsAA, 3 - epsBB, 4 - epsAB
		matrixes = [np.array([[0.0, m_par[0] / neigbours, m_par[1] / neigbours], [m_par[0] / neigbours, -m_par[2] + 2.0 * m_par[0] / neigbours, (m_par[0] + m_par[1]) / neigbours], [m_par[1] / neigbours, (m_par[0] + m_par[1]) / neigbours, -m_par[3] + 2.0 * m_par[1] / neigbours]]) ,] * 3
	elif model == "ising":
		matrixes = [np.array([[(m_par[1] - m_par[0] / (neigbours / 2.0)), (-m_par[1])],[(-m_par[1]), (m_par[1] + m_par[0] / (neigbours / 2.0))]]), ] * 3

	for i in range(len(matrixes)):
		matrixes[i] = matrixes[i] / (calc.constant * temp)
		matrixes[i] = np.array([np.exp(line) for line in matrixes[i]])
	return matrixes
