"""Shared computation, plotting and I/O helpers for the reproduction scripts.

Keeping the physics in one place lets ``figure3_ising.py``,
``figure4_binary_gas.py`` and the ``notebooks/reproduce_paper.ipynb``
notebook share exactly the same code paths.
"""

import os
import sys

# Make the project root importable no matter how the script is launched
# (``python -m reproduce.figureX`` or ``python reproduce/figureX.py``).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

import Scripts.MainScripts as ms

# Output locations (created on demand).
FIGURES_DIR = os.path.join(_ROOT, "figures")
DATA_DIR = os.path.join(_ROOT, "data")

# Exact reference critical temperatures, in units of k_B T_c / J.
TC_SQUARE = 2.0 / np.log(1.0 + np.sqrt(2.0))  # 2.2691...  (Onsager, square lattice)
TC_DHL = 1.641                                # diamond hierarchical lattice (Ref. [29])


# ---------------------------------------------------------------------------
# Ising model
# ---------------------------------------------------------------------------
def _ising_calc(lattice, **overrides):
    calc = ms.CalcConfig()
    calc.model = "ising"
    calc.lattice = lattice
    calc.constant = 1.0  # dimensionless units: temperature measured in J / k_B
    for key, value in overrides.items():
        setattr(calc, key, value)
    return calc


def ising_cv(calc, T, h, J):
    """Heat capacity of the Ising model at one temperature.

    ``J`` is the coupling entry of ``m_par``: ``J > 0`` is ferromagnetic,
    ``J < 0`` antiferromagnetic.  ``h`` is the (single-node) external field.
    """
    return ms.thermodynamics(
        calc, T, [h, J, 0, 0, 0, 0], heat_capacity=True
    )["heat_capacity"]


def ising_cv_curve(lattice, temperatures, h=0.0, J=1.0, betah=None, **overrides):
    """Converged heat-capacity curve vs temperature.

    If ``betah`` is given, the field is fixed in units of temperature at every
    point (``h = betah / beta = betah * constant * T``), reproducing the
    ``beta h = const`` curves of the paper.
    """
    calc = _ising_calc(lattice, **overrides)
    out = []
    for T in temperatures:
        field = betah * calc.constant * T if betah is not None else h
        out.append(ising_cv(calc, T, field, J))
    return np.asarray(out)


def ising_cv_curve_fixed_k(lattice, temperatures, k, h=0.0, J=1.0, **overrides):
    """Heat-capacity curve at a *fixed* renormalization depth ``k``.

    Setting ``methodTolerance = 0`` disables the early-convergence break, so
    the HTN iteration runs exactly ``k`` steps -- this is what produces the
    finite-``k`` family of curves in Fig. 3(a).
    """
    calc = _ising_calc(lattice, iterations=k, methodTolerance=0.0, **overrides)
    return np.asarray([ising_cv(calc, T, h, J) for T in temperatures])


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def peak(x, y):
    """Return ``(x*, y*)`` at the maximum of ``y``."""
    i = int(np.argmax(y))
    return float(x[i]), float(y[i])


def save_csv(filename, header, columns):
    """Save column data to ``data/<filename>`` as CSV with a header line."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    np.savetxt(path, np.column_stack(columns), delimiter=",",
               header=header, comments="")
    return path


def savefig(fig, filename):
    """Save a Matplotlib figure into ``figures/<filename>``."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    return path


def setup_style():
    """Apply a clean, consistent Matplotlib style. Imported lazily so the rest
    of the module stays usable without the optional plotting dependency."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    })
    return plt
