"""Capture baseline thermodynamic values for regression tests.

Run from the project root:

    python tools/capture_baseline.py > tools/baseline.txt

The printed Python literals are intended to be pasted into
``tests/test_regression.py`` as expected values.
"""

import os
import sys

# Make the project root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Scripts.MainScripts as ms


def ising_baseline(lattice, **extra):
    calc = ms.CalcConfig()
    calc.model = "ising"
    calc.lattice = lattice
    calc.constant = 1.0
    for key, value in extra.items():
        setattr(calc, key, value)

    h, J = 1.0, 1.0
    temperatures = [1.85, 1.95, 2.05, 2.15, 2.25, 2.35, 2.45, 2.55]

    results = {}
    for T in temperatures:
        out = ms.thermodynamics(calc, T, [h, -J, 0, 0, 0, 0], heat_capacity=True)
        cv = out["heat_capacity"]
        results[T] = cv
        print(f"    {T!r}: {cv!r},")
    return results


def binary_baseline(lattice, **extra):
    calc = ms.CalcConfig()
    calc.model = "binary"
    calc.lattice = lattice
    for key, value in extra.items():
        setattr(calc, key, value)

    T = 100.0
    chemical_potentials = [-8.0, -4.0, 0.0, 5.0, 10.0, 20.0, 30.0, 38.0]

    results = {}
    for mu in chemical_potentials:
        m_par = [mu, 10.0, 4.0, 6.0, 0.0, 0.0]
        out = ms.thermodynamics(
            calc, T, m_par,
            coverage=True, susceptibility=True, entropy=True, heat_capacity=True,
        )
        tup = (
            out["coverage"], out["entropy"], out["susceptibility"],
            out["heat_capacity"], out["grand_potential"],
        )
        results[mu] = tup
        print(f"    {mu!r}: {tup!r},")
    return results


if __name__ == "__main__":
    print("# === ising_diamond (coord=3) ===")
    print("ISING_DIAMOND = {")
    ising_baseline("diamond", coord=3)
    print("}")

    print("\n# === ising_FSHL (metParam=1) ===")
    print("ISING_FSHL_P1 = {")
    ising_baseline("FSHL", metParam=1)
    print("}")

    print("\n# === binary_diamond (coord=3) ===")
    print("BINARY_DIAMOND = {")
    binary_baseline("diamond", coord=3)
    print("}")

    print("\n# === binary_FSHL (metParam=1) ===")
    print("BINARY_FSHL_P1 = {")
    binary_baseline("FSHL", metParam=1)
    print("}")
