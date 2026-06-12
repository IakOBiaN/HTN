"""Hierarchical Tensor Networks (HTN) for lattice models.

Compute thermodynamic properties of statistical-physics models (Ising, binary
lattice gas) on hierarchical lattices (diamond-like and folded-square) via an
exact tensor-network renormalization scheme.

Public API
----------
>>> import htn
>>> calc = htn.CalcConfig(model="ising", lattice="diamond", coord=3, constant=1.0)
>>> htn.thermodynamics(calc, T=2.0, m_par=[0.0, 1.0], heat_capacity=True)
"""

from htn.MainScripts import CalcConfig, simulate, thermodynamics

__version__ = "0.1.0"

__all__ = ["CalcConfig", "simulate", "thermodynamics", "__version__"]
