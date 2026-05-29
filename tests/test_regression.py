"""Regression tests for HTN simulations.

The expected values below were captured on the bundled examples from
the original published code (see ``tools/capture_baseline.py``).
HTN iteration is deterministic, so the tolerance is tight; an absolute
floor handles observables that vanish in the dilute / saturated limits.

Any change that perturbs these values breaks the test -- which is
usually the desired behaviour: it flags an unintended regression.
When a change is *intentional* (e.g. a physics-level bug fix),
re-run ``tools/capture_baseline.py`` and update the expected values
in this file deliberately.
"""

import pytest

import Scripts.MainScripts as ms


# ---------------------------------------------------------------------------
# Tolerance for floating-point comparison.
# ---------------------------------------------------------------------------
TOL = dict(rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# Helpers.  Each builds a fresh CalcConfig that mirrors the corresponding
# example script in the project root.
# ---------------------------------------------------------------------------
def _ising_cv(lattice, T, **extra):
	calc = ms.CalcConfig()
	calc.model = "ising"
	calc.lattice = lattice
	calc.constant = 1.0
	for key, value in extra.items():
		setattr(calc, key, value)
	out = ms.thermodynamics(calc, T, [1.0, -1.0, 0, 0, 0, 0], heat_capacity=True)
	return out["heat_capacity"]


def _binary_full(lattice, mu, **extra):
	calc = ms.CalcConfig()
	calc.model = "binary"
	calc.lattice = lattice
	for key, value in extra.items():
		setattr(calc, key, value)
	out = ms.thermodynamics(
		calc, 100.0, [mu, 10.0, 4.0, 6.0, 0.0, 0.0],
		coverage=True, susceptibility=True, entropy=True, heat_capacity=True,
	)
	return (
		out["coverage"], out["entropy"], out["susceptibility"],
		out["heat_capacity"], out["grand_potential"],
	)


# ---------------------------------------------------------------------------
# Expected values (captured baseline).
# ---------------------------------------------------------------------------
ISING_DIAMOND = {
	1.85: 0.3955557559276457,
	1.95: 0.3728638797717565,
	2.05: 0.34951000760230494,
	2.15: 0.3263800141262152,
	2.25: 0.3040831586531567,
	2.35: 0.2829977141516338,
	2.45: 0.26332432012665663,
	2.55: 0.2451379366164019,
}

ISING_FSHL_P1 = {
	1.85: 0.5557323068394737,
	1.95: 0.6190866019828434,
	2.05: 0.6963291965256245,
	2.15: 0.8147935437463616,
	2.25: 0.7159248194277268,
	2.35: 0.6377734385054623,
	2.45: 0.5711088868487479,
	2.55: 0.513660387557735,
}

BINARY_DIAMOND = {
	-8.0: (0.0008158285669246368, 0.0001511837912637759, 0.0005429989856864381, 0.0008383516103549482, 5.00204000206663),
	-4.0: (0.01941264993332581, 0.0010967055010802085, 0.012439306811806716, 0.0027834623494982225, 5.025713755792556),
	0.0: (0.24979525746005038, 0.004360003257986023, 0.08333336222356992, 0.00021049828546892968, 5.43272254786954),
	5.0: (0.4910743413963914, 0.0005597707417592801, 0.005844213028183276, 0.00177369230414115, 7.511231755611281),
	10.0: (0.5000330775049022, 2.289888900719461e-05, 2.211011711494848e-05, 0.00016999734953060397, 10.000246592036286),
	20.0: (0.5870556773288627, 0.0029302271444819894, 0.04848145089368927, 0.002432010148822883, 15.117563121684569),
	30.0: (0.9991818994823376, 7.568169024807503e-05, 0.0005449536139678912, 0.0004199307568342192, 24.00102067494924),
	38.0: (0.9999986596547217, 2.312106062163366e-07, 9.038402026817494e-07, 2.8421709430404007e-06, 32.00000167152942),
}

BINARY_FSHL_P1 = {
	-8.0: (0.0009137112413348802, 0.00012405978155882735, 0.00046061217897630513, 0.0005514699807918078, 5.002006394108093),
	-4.0: (0.011037357074528131, 0.0006543408228587566, 0.006047562305155907, 0.0013797851750041445, 5.018000730191096),
	0.0: (0.24923415117328318, 0.002923293119572179, 0.48314429346980603, 0.0002227551476607914, 5.288729158444814),
	5.0: (0.4942181546683244, 0.0004148394707748082, 0.0030420189347069027, 0.0010818901330367225, 7.509824298784659),
	10.0: (0.4997350096322961, 6.480891912019615e-05, 0.00014017055143256132, 0.00033928415632544784, 10.000916659124359),
	20.0: (0.5030904605440867, 0.00026731535562873887, 0.0015925841594821577, 0.0008483880264975596, 15.0055186834389),
	30.0: (0.9783952045232525, 0.0012161854279213458, 0.011405222792859606, 0.002240341245851596, 22.03475306541249),
	38.0: (0.9998363288215018, 2.2400929111654477e-05, 8.189796503188518e-05, 0.00014281908988778014, 30.000272053490555),
}


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T, expected", list(ISING_DIAMOND.items()))
def test_ising_diamond(T, expected):
	"""Heat capacity of the Ising model on a diamond hierarchical lattice."""
	assert _ising_cv("diamond", T, coord=3) == pytest.approx(expected, **TOL)


@pytest.mark.parametrize("T, expected", list(ISING_FSHL_P1.items()))
def test_ising_fshl_p1(T, expected):
	"""Heat capacity of the Ising model on FSHL (p = 1)."""
	assert _ising_cv("FSHL", T, metParam=1) == pytest.approx(expected, **TOL)


@pytest.mark.parametrize("mu, expected", list(BINARY_DIAMOND.items()))
def test_binary_diamond(mu, expected):
	"""Thermodynamic observables of the binary gas on a diamond lattice."""
	assert _binary_full("diamond", mu, coord=3) == pytest.approx(expected, **TOL)


@pytest.mark.parametrize("mu, expected", list(BINARY_FSHL_P1.items()))
def test_binary_fshl_p1(mu, expected):
	"""Thermodynamic observables of the binary gas on FSHL (p = 1)."""
	assert _binary_full("FSHL", mu, metParam=1) == pytest.approx(expected, **TOL)
