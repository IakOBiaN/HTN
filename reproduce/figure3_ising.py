"""Reproduce Fig. 3 of Akimenko & Myshlyavtsev (2024): the Ising model.

Panel (a): diamond-like hierarchical lattice (DHL) at h = 0 for a growing
           renormalization depth k; the heat-capacity peak converges to the
           exact DHL critical temperature k_B T_c / J = 1.641.
Panel (b): the FSHL family (p = 1, 2, 5, 10) at h = 0; the peak sits at the
           exact square-lattice value k_B T_c / J = 2 / ln(1 + sqrt(2)) = 2.269.
Panel (c): antiferromagnet (J < 0) in a field beta h = 1 on the FSHL family;
           the critical point shifts as p grows.

Run from the repository root::

    python -m reproduce.figure3_ising
"""

import numpy as np

from reproduce import _common as C


def main():
    plt = C.setup_style()

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4.4))

    # --- Panel (a): DHL, finite k -----------------------------------------
    Ta = np.arange(0.80, 2.45, 0.005)
    cols_a = [Ta]
    for k in (1, 2, 4, 6, 12):
        cv = C.ising_cv_curve_fixed_k("diamond", Ta, k=k, h=0.0, J=1.0, coord=3)
        ax_a.plot(Ta, cv, label=f"k = {k}")
        cols_a.append(cv)
    ax_a.axvline(C.TC_DHL, color="0.4", ls="--", lw=1.2)
    ax_a.text(C.TC_DHL + 0.03, ax_a.get_ylim()[1] * 0.05,
              r"$k_B T_c/J = 1.641$", color="0.4", fontsize=9)
    ax_a.set_title("(a) DHL, $h = 0$, increasing $k$")
    ax_a.set_xlabel(r"$k_B T / J$")
    ax_a.set_ylabel(r"Heat capacity  $C$")
    ax_a.legend(ncol=2)

    # --- Panel (b): FSHL family, h = 0 ------------------------------------
    Tb = np.arange(1.80, 2.80, 0.01)
    cols_b = [Tb]
    for p in (1, 2, 5, 10):
        cv = C.ising_cv_curve("FSHL", Tb, h=0.0, J=1.0, metParam=p)
        ax_b.plot(Tb, cv, label=f"p = {p}")
        cols_b.append(cv)
    ax_b.axvline(C.TC_SQUARE, color="0.4", ls="--", lw=1.2)
    ax_b.text(C.TC_SQUARE + 0.02, ax_b.get_ylim()[1] * 0.05,
              r"$k_B T_c/J = 2.269$", color="0.4", fontsize=9)
    ax_b.set_title("(b) FSHL family, $h = 0$")
    ax_b.set_xlabel(r"$k_B T / J$")
    ax_b.set_ylabel(r"Heat capacity  $C$")
    ax_b.legend()

    # --- Panel (c): FSHL family, antiferromagnet, beta h = 1 --------------
    Tc = np.arange(1.50, 3.00, 0.01)
    cols_c = [Tc]
    for p in (1, 2, 5):
        cv = C.ising_cv_curve("FSHL", Tc, betah=1.0, J=-1.0, metParam=p)
        ax_c.plot(Tc, cv, label=f"p = {p}")
        cols_c.append(cv)
    ax_c.set_title(r"(c) FSHL family, $\beta h = 1$, $J < 0$")
    ax_c.set_xlabel(r"$k_B T / |J|$")
    ax_c.set_ylabel(r"Heat capacity  $C$")
    ax_c.legend()

    fig.tight_layout()
    out = C.savefig(fig, "figure3_ising.png")
    print("wrote", out)

    # --- Save the underlying data -----------------------------------------
    C.save_csv("figure3a_dhl_finite_k.csv",
               "T,k1,k2,k4,k6,k12", cols_a)
    C.save_csv("figure3b_fshl_h0.csv",
               "T,p1,p2,p5,p10", cols_b)
    C.save_csv("figure3c_fshl_antiferro.csv",
               "T,p1,p2,p5", cols_c)

    # --- Report the recovered critical temperatures -----------------------
    print("\nRecovered heat-capacity peak positions:")
    for k, col in zip((1, 2, 4, 6, 12), cols_a[1:]):
        print(f"  DHL  k = {k:>2}:  T_peak = {C.peak(Ta, col)[0]:.3f}   "
              f"(exact 1.641)")
    for p, col in zip((1, 2, 5, 10), cols_b[1:]):
        print(f"  FSHL p = {p:>2}:  T_peak = {C.peak(Tb, col)[0]:.3f}   "
              f"(exact 2.269)")


if __name__ == "__main__":
    main()
