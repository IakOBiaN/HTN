"""Scripts that reproduce the figures of

    S. S. Akimenko and A. V. Myshlyavtsev,
    "Tensor networks for hierarchical lattices",
    EPL 148, 61001 (2024).  https://doi.org/10.1209/0295-5075/ad994b

Each ``figureN_*.py`` module is runnable from the repository root, e.g.::

    python -m reproduce.figure3_ising
    python reproduce/figure4_binary_gas.py

and writes PNG figures into ``figures/`` and CSV data into ``data/``.
Plotting requires the optional dependencies in ``requirements-plot.txt``.
"""
