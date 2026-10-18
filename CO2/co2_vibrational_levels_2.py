"""
co2_vibrational_levels.py

A drop-in companion to numerov_vibrations2.py for the CO2 normal-mode
PES curves -- NOT a straight reuse, and deliberately so.

Testing numerov_vibrations2.py's shooting-method solver against the
CO2 symmetric-stretch PES (mu_eff = 16.0 amu, well width ~0.08 bohr)
showed genuinely unreliable output: reported node counts were
non-monotonic (e.g. 9, 7, 6, 5 instead of 0, 1, 2, 3) and level
spacings were wildly uneven, across a whole sweep of --rmin/--rmax/--dR
combinations. This isn't a parameter-tuning problem -- it's the known
instability of single-direction "shooting" Numerov integration through
a long classically-forbidden region: with CO2's much heavier effective
mass (16-27 amu vs. HCl's ~0.98 amu) the true wavefunction decays to
below double-precision noise within a short distance of the turning
point, and the growing (unphysical) solution component dominates the
rest of the propagation, corrupting the boundary value the bisection
search relies on. HCl's shallower/lighter problem apparently doesn't
expose this as badly, which is presumably why the original script's
defaults (rmin=0.6, rmax=12.0) never ran into it.

Rather than trying to patch the shooting method's stability, this
solves the same 1D nuclear Schrodinger equation the numerically robust
way: build a finite-difference Hamiltonian on a grid and diagonalize
it directly (dense eigh, or eigh_tridiagonal for speed). No forward
propagation, no exponential blow-up, no bisection -- eigh gives you
every level in the box in one shot, ordered and orthogonal by
construction. This is exactly the method already used (informally)
inside co2_taichi_vibration.py's solve_vibrational_states(); this
script just generalizes it into a proper multi-level CLI tool with
the same input/output shape as numerov_vibrations2.py, so it's a
drop-in replacement rather than a different workflow.

Input CSV format (same as numerov_vibrations2.py):
    R_bohr, R_angstrom, E_hartree

Example:
    python co2_vibrational_levels2.py --input pes_co2_sym.csv \\
        --mu-amu 15.999 --nlevels 6 --plot-png levels.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh_tridiagonal
from scipy.constants import physical_constants

HARTREE_TO_CM = physical_constants["hartree-inverse meter relationship"][0] / 100.0
HARTREE_TO_EV = physical_constants["Hartree energy in eV"][0]
AMU_TO_ME = physical_constants["atomic mass constant"][0] / physical_constants["electron mass"][0]


@dataclass
class PESData:
    R_bohr: np.ndarray
    E_hartree: np.ndarray


def read_pes_csv(path: Path) -> PESData:
    R, E = [], []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "R_bohr" not in (reader.fieldnames or []) or "E_hartree" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain columns: R_bohr, E_hartree")
        for row in reader:
            R.append(float(row["R_bohr"]))
            E.append(float(row["E_hartree"]))
    R = np.array(R, dtype=float)
    E = np.array(E, dtype=float)
    order = np.argsort(R)
    return PESData(R_bohr=R[order], E_hartree=E[order])


def solve_levels(R_bohr: np.ndarray, E_hartree: np.ndarray, mu_amu: float,
                  rmin: Optional[float] = None, rmax: Optional[float] = None,
                  ngrid: int = 2000, nlevels: int = 6, pad_frac: float = 0.15) -> List[dict]:
    """
    Diagonalize the finite-difference nuclear Hamiltonian on a dense
    grid spanning [rmin, rmax] (defaults to the CSV's own domain,
    padded by pad_frac on each side so the wavefunctions have room to
    decay to ~0 at the box edges -- diagonalization doesn't need the
    generous forbidden-region buffer a shooting method would, since
    there's no exponential blow-up to worry about; padding here is
    just to avoid an artificial hard-wall boundary condition biasing
    the upper levels).
    """
    Emin_pes = float(np.min(E_hartree))
    V_shifted = E_hartree - Emin_pes
    spline = CubicSpline(R_bohr, V_shifted, bc_type="natural")

    span = R_bohr[-1] - R_bohr[0]
    lo = rmin if rmin is not None else R_bohr[0] - pad_frac * span
    hi = rmax if rmax is not None else R_bohr[-1] + pad_frac * span
    # Never evaluate the spline outside its fitted domain.
    lo = max(lo, R_bohr[0])
    hi = min(hi, R_bohr[-1])

    R_grid = np.linspace(lo, hi, ngrid)
    dR = R_grid[1] - R_grid[0]
    mu_me = mu_amu * AMU_TO_ME
    V = spline(R_grid)

    diag = 1.0 / (mu_me * dR ** 2) + V
    offdiag = np.full(ngrid - 1, -0.5 / (mu_me * dR ** 2))

    evals, evecs = eigh_tridiagonal(diag, offdiag, select="i", select_range=(0, nlevels - 1))

    levels = []
    for v in range(nlevels):
        u = evecs[:, v]
        try: 
            u = u / np.sqrt(np.trapezoid(u ** 2, R_grid))
        except:
            u = u / np.sqrt(np.trapz(u ** 2, R_grid))
        
        nodes = int(np.sum(np.diff(np.sign(u[np.abs(u) > 1e-8 * np.max(np.abs(u))])) != 0))
        levels.append({
            "v": v,
            "E_hartree": float(evals[v]),
            "E_cm1": float(evals[v] * HARTREE_TO_CM),
            "E_eV": float(evals[v] * HARTREE_TO_EV),
            "nodes": nodes,
            "R": R_grid,
            "u": u,
        })
    return levels, Emin_pes, (lo, hi)


def plot_results(levels, R_bohr, E_hartree, mu_amu, bounds, output_png=None):
    import matplotlib
    if output_png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Emin_pes = float(np.min(E_hartree))
    spline = CubicSpline(R_bohr, E_hartree - Emin_pes, bc_type="natural")
    R_plot = np.linspace(bounds[0], bounds[1], 2000)
    V_plot = spline(R_plot) * HARTREE_TO_EV

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(R_plot, V_plot, "k-", lw=2, label="Shifted PES")

    colors = plt.cm.tab10(np.linspace(0, 0.7, len(levels)))
    scale = 0.12 * max(V_plot) if max(V_plot) > 0 else 1.0
    for lev, col in zip(levels, colors):
        E_eV = lev["E_eV"]
        mask = spline(lev["R"]) * HARTREE_TO_EV <= E_eV + 1e-6
        if mask.any():
            ax.hlines(E_eV, lev["R"][mask][0], lev["R"][mask][-1],
                       colors=col, linewidths=0.8, linestyles="--")
        u_plot = E_eV + scale * lev["u"] / (np.max(np.abs(lev["u"])) + 1e-30)
        ax.plot(lev["R"], u_plot, color=col, lw=1.2,
                label=f"v={lev['v']}  {lev['E_cm1']:.1f} cm\u207b\u00b9")

    ax.set_xlabel("q  (bohr, generalized normal-mode displacement)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"CO2 mode: matrix-diagonalization vibrational states (\u03bc={mu_amu:.4f} amu)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    if output_png:
        fig.savefig(output_png, dpi=150)
        print(f"Plot saved to: {output_png.resolve()}")
    else:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Solve CO2 normal-mode vibrational states by dense Hamiltonian diagonalization "
                    "(numerically robust alternative to shooting-method Numerov for stiff/heavy modes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True, help="CSV from co2_dft_pes.py")
    p.add_argument("--mu-amu", type=float, required=True, help="Reduced mass (amu)")
    p.add_argument("--rmin", type=float, default=None, help="Grid lower bound (bohr); default = CSV domain, padded")
    p.add_argument("--rmax", type=float, default=None, help="Grid upper bound (bohr); default = CSV domain, padded")
    p.add_argument("--ngrid", type=int, default=50000, help="Grid points for diagonalization")
    p.add_argument("--nlevels", type=int, default=6, help="Number of vibrational levels")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-png", type=Path, default=None)
    args = p.parse_args()

    data = read_pes_csv(args.input)
    levels, Emin_pes, bounds = solve_levels(
        data.R_bohr, data.E_hartree, args.mu_amu,
        rmin=args.rmin, rmax=args.rmax, ngrid=args.ngrid, nlevels=args.nlevels,
    )

    print("=== CO2 vibrational levels (dense diagonalization) ===")
    print(f"Input PES file : {args.input.resolve()}")
    print(f"Reduced mass   : {args.mu_amu:.8f} amu")
    print(f"PES minimum    : {Emin_pes:.12f} Ha")
    print(f"Grid           : [{bounds[0]:.4f}, {bounds[1]:.4f}] bohr, {args.ngrid} points")
    print()
    print(f"{'v':>3}  {'E (Ha)':>16}  {'E (cm/u207b/u00b9)':>12}  {'E (eV)':>10}  {'/u0394(cm/u207b/u00b9)':>10}  nodes")
    print("-" * 70)
    for i, lev in enumerate(levels):
        delta = lev["E_cm1"] - levels[i - 1]["E_cm1"] if i > 0 else float("nan")
        delta_str = f"{delta:10.3f}" if i > 0 else "       ---"
        zpe = "  \u2190 ZPE" if i == 0 else ""
        print(f"{lev['v']:>3}  {lev['E_hartree']:>16.10f}  {lev['E_cm1']:>12.4f}  "
              f"{lev['E_eV']:>10.6f}  {delta_str}  {lev['nodes']:>5}{zpe}")

    if args.plot or args.plot_png:
        plot_results(levels, data.R_bohr, data.E_hartree, args.mu_amu, bounds, args.plot_png)


if __name__ == "__main__":
    main()
