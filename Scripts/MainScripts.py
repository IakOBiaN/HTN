import numpy as np
from scipy.misc import derivative
import Scripts.TensorNetworks as tn
import Scripts.BuildTensors as bt

class CalcConfig:

	def __init__(self, methodTolerance = 1e-8, constant = 0.008314, method = "trg", metModification = "default", scale = 4, iterations = 300, model = "ising", lattice = "square", gen_tensor = "default", nodes = 1 , coord = 4, metParam = 10, join_tensors = [1, 1]):
		#tolerance of method
		self.methodTolerance = methodTolerance
		#constant. Default is R = 0.008314
		self.constant = constant
		#method for partition function calculation
		self.method = method
		#internal method modification
		self.metModification = metModification
		#scale of the method iteration. For trg and square lattice it is 2 * 2
		self.scale = scale
		#number of method iterations
		self.iterations = iterations
		#model for tensor network constructions
		self.model = model
		#lattice geometry
		self.lattice = lattice
		#method of tensor network generation
		self.gen_tensor = gen_tensor
		#number of initial nodes for first tensor
		self.nodes = nodes
		#coordination number of a lattice
		self.coord = coord
		#method parameter. For trg it is chi
		self.metParam = metParam
		#joining nodes
		self.join_tensors = join_tensors

	def __str__(self):
		return method + "_p_" + str(metParam) + "_" + model + "_" + lattice

def simulate(calc, T = 1.0, m_par = [0.0] * 10):

	matrixes = bt.build_matrix(calc, T, m_par)

	scale = 0.0
	old_scale = -1.0
	norm = 0

	covergence = [-1e8, ]
	for i in range(calc.iterations):
		calc.scale = i
		(tensors, scale, norm) = tn.htn_step(matrixes, scale, norm, calc)
		covergence.append((scale + np.log(norm)) / (calc.nodes / (calc.constant * T)))
		if abs(covergence[-2] - covergence[-1]) < (calc.methodTolerance / 100):
			break

	if i > 250:
		print("Warning! More than 250 iterations")
	nodes = calc.nodes
	return (scale + np.log(norm)) / (nodes / (calc.constant * T))

def heat_capasity(calc, T = 1., m_par = [0.0]*10):
	result = T * derivative(lambda x: simulate(calc, x, m_par), T, n=2, dx=1e-4)
	return result

def full(calc, T = 1., m_par = [0.0] * 10, dmu = 1e-3, dT = 1e-3, derivatives = [1, ] + [0] * 2, T_derivative = True, mu_derivative = True):
	grandPotential_dmu = []
	grandPotential_dT = []
	if mu_derivative:
		der_par = m_par[:]
		for i, par in enumerate(derivatives):
			if par == 1:
				der_par[i] -= dmu
		for _ in range(2):
			lnZ = simulate(calc, T, der_par)
			grandPotential_dmu.append(lnZ)
			for i, par in enumerate(derivatives):
				if par == 1:
					der_par[i] += 2.0 * dmu
		del der_par
	else:
		grandPotential_dmu = [0, 0]
	if T_derivative:
		for diff_T in [T - dT, T, T + dT]:
			lnZ = simulate(calc, diff_T, m_par)
			grandPotential_dT.append(lnZ)
	else:
		grandPotential_dT = [0, 0, 0]

	coverage = - (grandPotential_dmu[0] - grandPotential_dmu[1]) / (dmu * 2.0)
	entropy = - (grandPotential_dT[0] - grandPotential_dT[2]) / (dT * 2.0)
	susceptibility = calc.constant * T * (grandPotential_dmu[0] - 2.0 * grandPotential_dT[1] + grandPotential_dmu[1]) / (dT ** 2.0)
	heat_capacity = T * (grandPotential_dT[0] - 2.0 * grandPotential_dT[1] + grandPotential_dT[2]) / (dT ** 2.0)
	return coverage, entropy, susceptibility, heat_capacity, grandPotential_dT[1]
