# Reproducing the paper

Scripts that regenerate the figures of

> S. S. Akimenko and A. V. Myshlyavtsev, *"Tensor networks for hierarchical
> lattices"*, **EPL** 148, 61001 (2024).
> [doi:10.1209/0295-5075/ad994b](https://doi.org/10.1209/0295-5075/ad994b)

## Setup

```bash
pip install -r requirements-plot.txt   # NumPy + Matplotlib
```

## Run

From the repository root:

```bash
python -m reproduce.figure3_ising        # Fig. 3 -> figures/figure3_ising.png
```

The script writes a PNG into `figures/` and the underlying curves as CSV into
`data/`. The same code paths are exercised interactively in
[`notebooks/reproduce_paper.ipynb`](../notebooks/reproduce_paper.ipynb).

## What the figure shows

| Panel | Content | What it demonstrates |
|---|---|---|
| 3(a) | Ising heat capacity on the diamond lattice for increasing depth `k` | peak converges to the exact DHL value `k_B T_c / J = 1.641` |
| 3(b) | Ising heat capacity for the FSHL family (`p = 1, 2, 5, 10`) at `h = 0` | peak at the exact square-lattice value `2.269` |
| 3(c) | Antiferromagnet in a field (`beta h = 1`) on the FSHL family | critical point shifts with `p` |
