import numpy as np
import Scripts.TensorNetworks as tn
import Scripts.BuildTensors as bt


class CalcConfig:
	"""Configuration container for an HTN simulation.

	Attributes
	----------
	model : str
		Physical model: ``"ising"``, ``"binary"`` or ``"mono"``.
	lattice : str
		Lattice geometry: ``"diamond"`` or ``"FSHL"``.
	metParam : int
		Lattice / method parameter.  For FSHL this is the *p* value that
		controls the cluster size; larger *p* increases the effective
		coordination number.
	coord : int
		Coordination number of the lattice.
	constant : float
		Boltzmann / gas constant used to convert energy units.  Default is
		*R* = 0.008314 kJ mol-1 K-1; set to ``1`` for dimensionless units.
	iterations : int
		Maximum number of HTN renormalization steps.
	methodTolerance : float
		Convergence threshold: iteration stops when the change in free
		energy per node drops below ``methodTolerance / 100``.
	method, metModification, gen_tensor, nodes, scale, join_tensors
		Internal bookkeeping fields used by the simulation routines.
	"""

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
		return self.method + "_p_" + str(self.metParam) + "_" + self.model + "_" + self.lattice

def simulate(calc, T = 1.0, m_par = [0.0] * 10):
	"""Compute the grand potential per node via HTN iteration.

	Parameters
	----------
	calc : CalcConfig
		Simulation configuration.
	T : float
		Temperature (in units consistent with ``calc.constant``).
	m_par : list of float
		Model parameters; see :func:`Scripts.BuildTensors.build_matrix`
		for the per-model layout.

	Returns
	-------
	float
		``ln Z / N`` -- grand potential per node in units of
		``calc.constant * T``.
	"""
	matrixes = bt.build_matrix(calc, T, m_par)

	scale = 0.0
	old_scale = -1.0
	norm = 0

	convergence = [-1e8, ]
	for i in range(calc.iterations):
		calc.scale = i
		(tensors, scale, norm) = tn.htn_step(matrixes, scale, norm, calc)
		convergence.append((scale + np.log(norm)) / (calc.nodes / (calc.constant * T)))
		if abs(convergence[-2] - convergence[-1]) < (calc.methodTolerance / 100):
			break

	if i > 250:
		print("Warning! More than 250 iterations")
	nodes = calc.nodes
	return (scale + np.log(norm)) / (nodes / (calc.constant * T))

def thermodynamics(calc, T=1.0, m_par=None, *, coverage=False, susceptibility=False,
				   entropy=False, heat_capacity=False, mu_index=0, dmu=1e-3, dT=1e-3):
	"""Compute the requested thermodynamic observables.

	Each observable is a finite-difference derivative of the grand
	potential ``Omega = simulate(calc, T, m_par)``.  Only the points
	needed for the requested observables are evaluated, and the central
	point ``Omega(mu0, T0)`` is shared between the two second
	derivatives:

	==================  ======================  ==================
	Observable          Derivative              Points evaluated
	==================  ======================  ==================
	``coverage``        first,  d/dmu           ``mu-``, ``mu+``
	``susceptibility``  second, d2/dmu2         ``mu-``, C, ``mu+``
	``entropy``         first,  d/dT            ``T-``, ``T+``
	``heat_capacity``   second, d2/dT2          ``T-``, C, ``T+``
	==================  ======================  ==================

	Parameters
	----------
	calc : CalcConfig
		Simulation configuration.
	T : float
		Temperature.
	m_par : list of float, optional
		Model parameters.
	coverage, susceptibility, entropy, heat_capacity : bool
		Select which observables to compute.
	mu_index : int
		Index into ``m_par`` of the chemical potential to differentiate
		(for ``coverage`` / ``susceptibility``).
	dmu, dT : float
		Finite-difference step sizes for the chemical potential and the
		temperature.

	Returns
	-------
	dict
		Maps each requested observable name to its value.  When a second
		derivative is requested the central point is evaluated anyway, so
		``grand_potential`` (= ``Omega(mu0, T0)``) is included as well.
	"""
	if m_par is None:
		m_par = [0.0] * 10

	need_mu = coverage or susceptibility
	need_T = entropy or heat_capacity
	need_center = susceptibility or heat_capacity

	result = {}

	center = None
	if need_center:
		center = simulate(calc, T, m_par)
		result["grand_potential"] = center

	if need_mu:
		mu_minus = m_par[:]
		mu_plus = m_par[:]
		mu_minus[mu_index] -= dmu
		mu_plus[mu_index] += dmu
		omega_mu_minus = simulate(calc, T, mu_minus)
		omega_mu_plus = simulate(calc, T, mu_plus)
		if coverage:
			result["coverage"] = -(omega_mu_minus - omega_mu_plus) / (2.0 * dmu)
		if susceptibility:
			result["susceptibility"] = calc.constant * T * (omega_mu_minus - 2.0 * center + omega_mu_plus) / (dmu ** 2)

	if need_T:
		omega_T_minus = simulate(calc, T - dT, m_par)
		omega_T_plus = simulate(calc, T + dT, m_par)
		if entropy:
			result["entropy"] = -(omega_T_minus - omega_T_plus) / (2.0 * dT)
		if heat_capacity:
			result["heat_capacity"] = T * (omega_T_minus - 2.0 * center + omega_T_plus) / (dT ** 2)

	return result
