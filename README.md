# Hierarchical Tensor Networks for Lattice Models

Python implementation of the **Hierarchical Tensor Network (HTN)** approach for computing thermodynamic properties of statistical physics models on hierarchical lattices.

## Overview

The HTN method computes the partition function of a lattice model by iteratively contracting a tensor network built on a hierarchical lattice geometry. Unlike regular-lattice methods (e.g., TRG), hierarchical lattices admit an exact renormalization group scheme: each contraction step replaces a cluster of tensors with a single effective tensor, and convergence of the free energy signals that the thermodynamic limit has been reached.

This repository provides a reusable library (`Scripts/`) and four ready-to-run examples covering two models on two lattice geometries:

| Example script | Model | Lattice | Observable |
|---|---|---|---|
| `ising_diamond.py` | Ising | Diamond | Heat capacity vs. *T* |
| `ising_FSHL.py` | Ising | Folded square (FSHL) | Heat capacity vs. *T* |
| `binary_diamond.py` | Binary lattice gas | Diamond | Coverage, entropy, susceptibility, heat capacity vs. *μ* |
| `binary_FSHL.py` | Binary lattice gas | FSHL | Coverage, entropy, susceptibility, heat capacity vs. *μ* |

## Related publication

The tensor network construction approach used here is described and benchmarked in:

> S. S. Akimenko, *"Tensor network construction for lattice gas models: Hard-core and triangular lattice models"*, **Physical Review E** 107, 054116 (2023).  
> DOI: [10.1103/PhysRevE.107.054116](https://doi.org/10.1103/PhysRevE.107.054116)

## Supported models

### Ising model

Spin-½ ferromagnet in an external field. The transfer matrix elements are determined by coupling constant *J* and external field *h*:

```
m_par = [h, J]
```

The heat capacity peak locates the critical temperature.

### Binary lattice gas

Two-component (A + B) lattice gas with chemical potentials *μ*_A, *μ*_B and pairwise interaction energies *ε*_AA, *ε*_BB, *ε*_AB:

```
m_par = [muA, muB, epsAA, epsBB, epsAB]
```

The `full()` function returns coverage, entropy, susceptibility, heat capacity, and grand potential in one call.

## Supported lattice geometries

### Diamond hierarchical lattice

A fractal lattice built by replacing each bond with a diamond motif. The coordination number is set via `calc.coord` (default 3 for a diamond graph).

### Folded Square Hierarchical Lattice (FSHL)

A family of hierarchical lattices parameterised by an integer *p* (`calc.metParam`). At *p* = 1 the geometry resembles a folded square; larger *p* increases the effective coordination number and cluster size.

## Installation

**Requirements:** Python ≥ 3.8, NumPy, SciPy.

```bash
git clone https://github.com/iakobian/htn.git
cd htn
pip install -r requirements.txt
```

## Quick start

### Ising model on a diamond lattice — heat capacity scan

```python
import Scripts.MainScripts as ms
import numpy as np

calc = ms.CalcConfig()
calc.model   = "ising"
calc.lattice = "diamond"
calc.coord   = 3
calc.constant = 1.0   # dimensionless units

J, h = 1.0, 0.0
for T in np.arange(1.8, 2.6, 0.05):
    Cv = ms.heat_capacity(calc, T, m_par=[h, -J])
    print(f"T = {T:.2f}   Cv = {Cv:.6f}")
```

### Binary lattice gas on FSHL — equation of state

```python
import Scripts.MainScripts as ms
import numpy as np

calc = ms.CalcConfig()
calc.model    = "binary"
calc.lattice  = "FSHL"
calc.metParam = 1        # p parameter

T = 100.0
print("mu   coverage   entropy   susceptibility   heat_capacity   grand_potential")
for mu in np.arange(-10.0, 40.0, 2.0):
    m_par = [mu, 10.0, 4.0, 6.0, 0.0]
    theta, S, chi, Cv, Omega = ms.full(calc, T, m_par)
    print(f"{mu:6.1f}  {theta:.4f}  {S:.4f}  {chi:.4f}  {Cv:.4f}  {Omega:.4f}")
```

Run the bundled example scripts directly from the repository root:

```bash
python ising_diamond.py
python binary_FSHL.py
```

Output is printed to stdout as whitespace-separated columns, ready for piping into plotting tools.

## Running tests

A small regression suite (`tests/test_regression.py`) checks selected
thermodynamic observables against values captured from the original
published code on the four bundled examples.

```bash
python -m venv .venv
.venv\Scripts\activate              # PowerShell: .venv\Scripts\Activate.ps1
                                    # Unix:       source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -v
```

To regenerate the baseline values (for example after an intentional
physics-level change), run:

```bash
.venv\Scripts\python.exe tools\capture_baseline.py
```

and paste the printed dictionaries into `tests/test_regression.py`.

## API reference

### `CalcConfig`

Central configuration object. Key attributes:

| Attribute | Default | Description |
|---|---|---|
| `model` | `"ising"` | Physical model: `"ising"`, `"binary"`, or `"mono"` |
| `lattice` | `"square"` | Lattice geometry: `"diamond"` or `"FSHL"` |
| `metParam` | `10` | Method / lattice parameter (FSHL: *p* value) |
| `coord` | `4` | Coordination number |
| `constant` | `0.008314` | Gas constant *R* (kJ mol⁻¹ K⁻¹); set to `1` for dimensionless units |
| `iterations` | `300` | Maximum HTN iterations |
| `methodTolerance` | `1e-8` | Convergence threshold on free energy per node |

### `simulate(calc, T, m_par)`

Returns `ln Z / N` (grand potential per node in units of `constant * T`) at temperature *T* and model parameters `m_par`.

### `full(calc, T, m_par)`

Returns a tuple `(coverage, entropy, susceptibility, heat_capacity, grand_potential)` computed via finite differences of `simulate`.

### `heat_capacity(calc, T, m_par)`

Returns the heat capacity *C*_V at temperature *T* via a second numerical derivative of `simulate`.

## Project structure

```
htn/
├── Scripts/
│   ├── __init__.py          # package marker
│   ├── BuildTensors.py      # transfer matrix construction for each model
│   ├── TensorNetworks.py    # HTN contraction step (diamond and FSHL)
│   └── MainScripts.py       # CalcConfig, simulate(), full(), heat_capacity()
├── tests/
│   └── test_regression.py   # baseline values captured from the published code
├── tools/
│   └── capture_baseline.py  # regenerate baseline values
├── ising_diamond.py         # example: Ising / diamond lattice
├── ising_FSHL.py            # example: Ising / FSHL
├── binary_diamond.py        # example: binary gas / diamond lattice
├── binary_FSHL.py           # example: binary gas / FSHL
├── pyproject.toml           # pytest configuration
├── requirements.txt         # runtime dependencies
├── requirements-dev.txt     # adds pytest on top of requirements.txt
└── LICENSE
```

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in the repository.
