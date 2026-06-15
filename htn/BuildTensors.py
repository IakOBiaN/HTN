from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from htn.MainScripts import CalcConfig


def build_matrix(calc: CalcConfig, temp: float, m_par: list[float]) -> list[np.ndarray]:
    """Build the per-edge transfer matrices for the chosen model.

    Returns a list of three Boltzmann-weight matrices (one per edge
    direction) with shape ``(states, states)``, where ``states`` depends on
    the model: 2 for ``ising``/``mono`` and 3 for ``binary``.

    Parameters
    ----------
    calc : htn.MainScripts.CalcConfig
            Simulation configuration; ``calc.model``, ``calc.coord`` and
            ``calc.constant`` are read.
    temp : float
            Temperature in the same units as ``calc.constant``.
    m_par : list of float
            Model parameters.  Layout per model:

            * ``mono``:   ``[mu, eps]``
            * ``binary``: ``[muA, muB, epsAA, epsBB, epsAB]``
            * ``ising``:  ``[h, J]``

            Trailing zeros are appended automatically up to length 10.

    Returns
    -------
    list of numpy.ndarray
            Three Boltzmann-weight matrices (right / bottom / extra edge).
    """
    model = calc.model
    neighbours = calc.coord

    if len(m_par) < 10:
        m_par = m_par + [0.0] * (10 - len(m_par))

    models_dict = {"mono": True, "binary": True, "ising": True}

    exist = models_dict.get(calc.model)
    assert exist is not None, "Error! There is no such model in the database"

    # [right, bottom]
    matrixes: list[np.ndarray] = []
    if model == "mono":
        matrixes = [
            np.array(
                [
                    [0.0, m_par[0] / neighbours],
                    [m_par[0] / neighbours, -m_par[1] + m_par[0] / (neighbours / 2.0)],
                ]
            ),
        ] * 3
    elif model == "binary":
        # m_par: 0 - muA, 1 - muB, 2 - epsAA, 3 - epsBB, 4 - epsAB
        matrixes = [
            np.array(
                [
                    [0.0, m_par[0] / neighbours, m_par[1] / neighbours],
                    [
                        m_par[0] / neighbours,
                        -m_par[2] + 2.0 * m_par[0] / neighbours,
                        (m_par[0] + m_par[1]) / neighbours,
                    ],
                    [
                        m_par[1] / neighbours,
                        (m_par[0] + m_par[1]) / neighbours,
                        -m_par[3] + 2.0 * m_par[1] / neighbours,
                    ],
                ]
            ),
        ] * 3
    elif model == "ising":
        matrixes = [
            np.array(
                [
                    [(m_par[1] - m_par[0] / (neighbours / 2.0)), (-m_par[1])],
                    [(-m_par[1]), (m_par[1] + m_par[0] / (neighbours / 2.0))],
                ]
            ),
        ] * 3

    for i in range(len(matrixes)):
        matrixes[i] = matrixes[i] / (calc.constant * temp)
        matrixes[i] = np.exp(matrixes[i])
    return matrixes
