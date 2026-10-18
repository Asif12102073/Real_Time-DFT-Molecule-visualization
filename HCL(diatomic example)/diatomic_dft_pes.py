"""
diatomic_dft_pes.py

Generate a 1D diatomic potential energy curve (PES) with DFT using PySCF,
with optional GPU4PySCF backend if available.

Output:
    CSV with columns:
        R_bohr, R_angstrom, E_hartree

Example – HCl with a decent basis set:
    python diatomic_dft_pes.py \\
        --atom1 H --atom2 Cl --basis def2-svp --xc b3lyp \\
        --rmin 0.9 --rmax 3.5 --points 35 \\
        --output pes_hcl.csv --plot

Then solve vibrations:
    python numerov_vibrations.py \\
        --input pes_hcl.csv --mu-amu 0.97959 --nlevels 6 --plot

Notes:
    - Reduced mass of HCl: mu = 1.00794*35.453/(1.00794+35.453) ≈ 0.97959 amu
    - This script generates the electronic PES only.
    - The vibrational problem is solved by the companion script:
      numerov_vibrations.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy.constants import physical_constants
    from scipy.optimize import curve_fit
    from scipy.interpolate import CubicSpline
    BOHR_RADIUS_M = physical_constants["Bohr radius"][0]
    ANGSTROM_TO_BOHR = 1e-10 / BOHR_RADIUS_M
    HARTREE_TO_EV = physical_constants["Hartree energy in eV"][0]
    HARTREE_TO_CM = physical_constants["hartree-inverse meter relationship"][0] / 100.0
    _HAS_SCIPY = True
except Exception:
    ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
    HARTREE_TO_EV = 27.211386245988
    HARTREE_TO_CM = 219474.6313632
    _HAS_SCIPY = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Backend import
# ---------------------------------------------------------------------------

def _import_backend(use_gpu: bool):
    """
    Try gpu4pyscf first if requested; otherwise fall back to PySCF.
    The API is intentionally kept close enough that the rest of the code
    does not need to care which backend is active.
    """
    if use_gpu:
        try:
            from gpu4pyscf import gto, dft
            return "gpu4pyscf", gto, dft
        except Exception as exc:
            print(
                f"[warn] gpu4pyscf requested but unavailable ({exc}). "
                "Falling back to PySCF.",
                file=sys.stderr,
            )

    from pyscf import gto, dft
    return "pyscf", gto, dft


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PESPoint:
    R_bohr: float
    R_angstrom: float
    E_hartree: float


# ---------------------------------------------------------------------------
# Single-point DFT
# ---------------------------------------------------------------------------

def run_single_point(
    atom1: str,
    atom2: str,
    R_angstrom: float,
    basis: str = "def2-svp",
    xc: str = "b3lyp",
    charge: int = 0,
    spin: int = 0,
    use_gpu: bool = False,
    verbose: int = 0,
    symmetry: bool = False,
) -> float:
    """
    Run a DFT single-point calculation for a diatomic aligned on the z-axis.
    Spin convention: spin = Nalpha - Nbeta.
    """
    backend_name, gto, dft = _import_backend(use_gpu)

    atom_spec = f"{atom1} 0 0 0; {atom2} 0 0 {R_angstrom}"
    mol = gto.Mole()
    mol.atom = atom_spec
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.symmetry = symmetry
    mol.verbose = verbose
    mol.build()

    if spin == 0:
        mf = dft.RKS(mol)
    else:
        mf = dft.UKS(mol)

    mf.xc = xc
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.verbose = verbose

    if hasattr(mf, "grids"):
        mf.grids.level = 3

    energy = mf.kernel()
    if not mf.converged:
        print(
            f"[warn] SCF not fully converged at R = {R_angstrom:.6f} Å "
            f"({backend_name}, energy={energy:.12f})",
            file=sys.stderr,
        )
    return float(energy)


# ---------------------------------------------------------------------------
# PES generation
# ---------------------------------------------------------------------------

def generate_pes(
    atom1: str,
    atom2: str,
    basis: str,
    xc: str,
    rmin: float,
    rmax: float,
    points: int,
    charge: int = 0,
    spin: int = 0,
    use_gpu: bool = False,
    verbose: int = 0,
    symmetry: bool = False,
) -> List[PESPoint]:
    """Generate a PES on a uniform bond-length grid in Angstrom."""
    R_list = np.linspace(rmin, rmax, points)
    energies: List[float] = []

    for i, R in enumerate(R_list, start=1):
        print(f"[{i:03d}/{points:03d}] R = {R:.6f} Å ... ", end="", flush=True)
        E = run_single_point(
            atom1=atom1,
            atom2=atom2,
            R_angstrom=float(R),
            basis=basis,
            xc=xc,
            charge=charge,
            spin=spin,
            use_gpu=use_gpu,
            verbose=verbose,
            symmetry=symmetry,
        )
        energies.append(E)
        print(f"E = {E:.12f} Ha")

    return [
        PESPoint(
            R_bohr=float(R * ANGSTROM_TO_BOHR),
            R_angstrom=float(R),
            E_hartree=float(E),
        )
        for R, E in zip(R_list, energies)
    ]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_csv(points: List[PESPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["R_bohr", "R_angstrom", "E_hartree"])
        for p in points:
            writer.writerow([
                f"{p.R_bohr:.16e}",
                f"{p.R_angstrom:.16e}",
                f"{p.E_hartree:.16e}",
            ])


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _morse_potential(R: np.ndarray, De: float, Re: float, a: float, E0: float) -> np.ndarray:
    """
    Morse potential:  V(R) = De*(1 - exp(-a*(R-Re)))^2 + E0
    Fit in Angstrom, Hartree.
    """
    return De * (1.0 - np.exp(-a * (R - Re))) ** 2 + E0


def fit_morse(R_ang: np.ndarray, E_hartree: np.ndarray) -> Optional[dict]:
    """
    Fit a Morse potential to the PES.  Returns None if fit fails.
    """
    if not _HAS_SCIPY:
        return None

    imin = int(np.argmin(E_hartree))
    Re0 = float(R_ang[imin])
    E0_0 = float(E_hartree[imin])
    De0 = float(E_hartree[-1] - E0_0)
    if De0 <= 0:
        De0 = 0.1

    try:
        p0 = [De0, Re0, 1.5, E0_0]
        bounds = ([0, 0.5, 0.1, -1e6], [10, 5.0, 10.0, 1e6])
        popt, pcov = curve_fit(
            _morse_potential, R_ang, E_hartree,
            p0=p0, bounds=bounds, maxfev=8000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "De": popt[0], "Re": popt[1], "a": popt[2], "E0": popt[3],
            "De_err": perr[0], "Re_err": perr[1], "a_err": perr[2],
            # Morse harmonic frequency: omega_e = a*sqrt(2*De/mu) — needs mu
            # De in eV for display
            "De_eV": popt[0] * HARTREE_TO_EV,
            "popt": popt,
        }
    except Exception as exc:
        print(f"[warn] Morse fit failed: {exc}", file=sys.stderr)
        return None


def harmonic_force_constant(R_ang: np.ndarray, E_hartree: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate the harmonic force constant k = d²E/dR² at the minimum using
    a cubic spline second derivative.

    Returns:
        Re_ang  – equilibrium bond length in Angstrom
        k_ha    – force constant in Hartree/Angstrom²
        k_si    – force constant in N/m  (for spectroscopists)
    """
    if not _HAS_SCIPY:
        # fall back to finite difference
        imin = int(np.argmin(E_hartree))
        if 1 <= imin < len(R_ang) - 1:
            dR = R_ang[imin + 1] - R_ang[imin - 1]
            k_ha = (E_hartree[imin + 1] - 2 * E_hartree[imin] + E_hartree[imin - 1]) / (0.5 * dR) ** 2
        else:
            k_ha = float("nan")
        Re_ang = float(R_ang[int(np.argmin(E_hartree))])
    else:
        cs = CubicSpline(R_ang, E_hartree)
        # Refine minimum location via spline
        Re_ang = float(R_ang[int(np.argmin(E_hartree))])
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(cs, bounds=(R_ang[0], R_ang[-1]), method="bounded")
        if res.success:
            Re_ang = float(res.x)
        k_ha = float(cs(Re_ang, 2))   # second derivative

    # Convert Ha/Å² → N/m
    # 1 Ha = 4.3597447222071e-18 J; 1 Å = 1e-10 m
    # k [N/m] = k [Ha/Å²] * (4.3597447222071e-18 / (1e-10)^2)
    HARTREE_J = 4.3597447222071e-18
    ANGSTROM_M = 1e-10
    k_si = k_ha * HARTREE_J / ANGSTROM_M**2
    return Re_ang, float(k_ha), float(k_si)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pes(
    pts: List[PESPoint],
    atom1: str,
    atom2: str,
    xc: str,
    basis: str,
    output_png: Optional[Path] = None,
) -> None:
    """
    Four-panel PES plot:
      (A) Raw PES with Morse fit overlay
      (B) Shifted PES (relative to minimum)
      (C) Force = -dE/dR (numerical)
      (D) Second derivative d²E/dR² (harmonic surface curvature)
    """
    if not _HAS_MPL:
        print("[warn] matplotlib not available; skipping plots.", file=sys.stderr)
        return

    R_ang  = np.array([p.R_angstrom for p in pts])
    R_bohr = np.array([p.R_bohr     for p in pts])
    E_ha   = np.array([p.E_hartree  for p in pts])
    E_ev   = E_ha * HARTREE_TO_EV
    E_sh   = E_ha - np.min(E_ha)          # shifted (Ha)
    E_sh_ev = E_sh * HARTREE_TO_EV        # shifted (eV)

    imin   = int(np.argmin(E_ha))
    Re_ang = R_ang[imin]
    Emin   = E_ha[imin]

    # Smooth grid for overlays
    if _HAS_SCIPY:
        cs_raw    = CubicSpline(R_ang, E_ha)
        cs_sh     = CubicSpline(R_ang, E_sh)
        R_fine    = np.linspace(R_ang[0], R_ang[-1], 800)
        E_fine_ha = cs_raw(R_fine)
        E_fine_sh = cs_sh(R_fine)
        force_fine = -cs_raw(R_fine, 1)              # -dE/dR [Ha/Å]
        curv_fine  =  cs_raw(R_fine, 2)              # d²E/dR² [Ha/Å²]
    else:
        R_fine    = R_ang
        E_fine_ha = E_ha
        E_fine_sh = E_sh
        force_fine = -np.gradient(E_ha, R_ang)
        curv_fine  =  np.gradient(force_fine, R_ang) * (-1)  # approx

    # Morse fit
    morse = fit_morse(R_ang, E_ha)
    Re_refined, k_ha, k_si = harmonic_force_constant(R_ang, E_ha)

    # ---- figure layout ----
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"DFT PES: {atom1}–{atom2}   [{xc.upper()} / {basis}]",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    # ---- (A) Raw PES + Morse overlay ----
    ax_A.scatter(R_ang, E_ev, s=50, zorder=5, color="steelblue", label="DFT points")
    ax_A.plot(R_fine, E_fine_ha * HARTREE_TO_EV, color="steelblue", lw=1.5, label="Spline")
    if morse is not None:
        E_morse = _morse_potential(R_fine, *morse["popt"])
        ax_A.plot(R_fine, E_morse * HARTREE_TO_EV, "r--", lw=1.5,
                  label=f"Morse fit  (Dₑ={morse['De_eV']:.3f} eV, Rₑ={morse['Re']:.4f} Å)")
    ax_A.axvline(Re_ang, color="gray", ls=":", lw=1)
    ax_A.set_xlabel("R (Å)")
    ax_A.set_ylabel("Energy (eV)")
    ax_A.set_title("(A) Raw PES")
    ax_A.legend(fontsize=8)
    ax_A.grid(True, alpha=0.3)

    # ---- (B) Shifted PES ----
    ax_B.scatter(R_ang, E_sh_ev, s=50, zorder=5, color="darkorange")
    ax_B.plot(R_fine, E_fine_sh * HARTREE_TO_EV, color="darkorange", lw=1.5)
    # Harmonic parabola overlay at minimum
    if _HAS_SCIPY and not np.isnan(k_ha):
        parabola = 0.5 * k_ha * (R_fine - Re_refined) ** 2
        ax_B.plot(R_fine, parabola * HARTREE_TO_EV, "g:", lw=1.5,
                  label=f"Harmonic (k={k_si:.1f} N/m)")
    ax_B.axvline(Re_ang, color="gray", ls=":", lw=1)
    ax_B.set_xlabel("R (Å)")
    ax_B.set_ylabel("ΔE (eV)")
    ax_B.set_title("(B) Shifted PES + Harmonic approx.")
    ax_B.legend(fontsize=8)
    ax_B.set_ylim(bottom=-0.01 * np.max(E_sh_ev))
    ax_B.grid(True, alpha=0.3)

    # ---- (C) Force curve ----
    force_pts = -np.gradient(E_ha, R_ang)    # raw data force [Ha/Å]
    ax_C.scatter(R_ang, force_pts * HARTREE_TO_EV, s=40, color="purple", zorder=5, label="Finite diff.")
    ax_C.plot(R_fine, force_fine * HARTREE_TO_EV, color="purple", lw=1.5, label="Spline deriv.")
    ax_C.axhline(0, color="k", lw=0.8, ls="--")
    ax_C.axvline(Re_ang, color="gray", ls=":", lw=1, label=f"Rₑ = {Re_ang:.4f} Å")
    ax_C.set_xlabel("R (Å)")
    ax_C.set_ylabel("F = −dE/dR  (eV/Å)")
    ax_C.set_title("(C) Force curve")
    ax_C.legend(fontsize=8)
    ax_C.grid(True, alpha=0.3)

    # ---- (D) Curvature / harmonic surface ----
    curv_pts = np.gradient(np.gradient(E_ha, R_ang), R_ang)
    ax_D.scatter(R_ang, curv_pts, s=40, color="teal", zorder=5, label="Finite diff.")
    ax_D.plot(R_fine, curv_fine, color="teal", lw=1.5, label="Spline 2nd deriv.")
    ax_D.axhline(0, color="k", lw=0.8, ls="--")
    ax_D.axvline(Re_ang, color="gray", ls=":", lw=1)
    if not np.isnan(k_ha):
        ax_D.axhline(k_ha, color="green", ls="--", lw=1.2,
                     label=f"k at Rₑ = {k_ha:.4f} Ha/Å²\n= {k_si:.1f} N/m")
    ax_D.set_xlabel("R (Å)")
    ax_D.set_ylabel("d²E/dR²  (Ha/Å²)")
    ax_D.set_title("(D) Surface curvature")
    ax_D.legend(fontsize=8)
    ax_D.grid(True, alpha=0.3)

    # ---- summary text box ----
    summary_lines = [
        f"Rₑ (DFT grid)   = {Re_ang:.4f} Å",
        f"Rₑ (refined)    = {Re_refined:.4f} Å",
        f"E_min           = {Emin:.10f} Ha",
        f"k at Rₑ         = {k_si:.2f} N/m",
    ]
    if morse:
        summary_lines += [
            f"Morse Dₑ        = {morse['De_eV']:.4f} eV",
            f"Morse Rₑ        = {morse['Re']:.4f} Å",
            f"Morse α         = {morse['a']:.4f} Å⁻¹",
        ]
    fig.text(
        0.5, 0.01,
        "   |   ".join(summary_lines),
        ha="center", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    if output_png is not None:
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_png.resolve()}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a diatomic DFT PES and optionally plot it",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--atom1",     type=str,   default="H",          help="First atom symbol")
    p.add_argument("--atom2",     type=str,   default="Cl",         help="Second atom symbol")
    p.add_argument("--basis",     type=str,   default="def2-svp",   help="Basis set")
    p.add_argument("--xc",        type=str,   default="b3lyp",      help="DFT functional")
    p.add_argument("--charge",    type=int,   default=0,            help="Molecular charge")
    p.add_argument("--spin",      type=int,   default=0,            help="2S = Nalpha − Nbeta")
    p.add_argument("--rmin",      type=float, default=0.9,          help="Min bond length (Å)")
    p.add_argument("--rmax",      type=float, default=3.5,          help="Max bond length (Å)")
    p.add_argument("--points",    type=int,   default=35,           help="Number of grid points")
    p.add_argument("--output",    type=Path,  default=Path("pes_hcl.csv"), help="Output CSV path")
    p.add_argument("--plot",      action="store_true",  help="Show PES plots")
    p.add_argument("--plot-png",  type=Path,  default=None,         help="Save plot to PNG instead of showing")
    p.add_argument("--use-gpu",   action="store_true",  help="Try gpu4pyscf if available")
    p.add_argument("--verbose",   type=int,   default=0,            help="PySCF verbosity")
    p.add_argument("--symmetry",  action="store_true",  help="Enable molecular symmetry")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pts = generate_pes(
        atom1=args.atom1,
        atom2=args.atom2,
        basis=args.basis,
        xc=args.xc,
        rmin=args.rmin,
        rmax=args.rmax,
        points=args.points,
        charge=args.charge,
        spin=args.spin,
        use_gpu=args.use_gpu,
        verbose=args.verbose,
        symmetry=args.symmetry,
    )

    write_csv(pts, args.output)

    E_vals = np.array([p.E_hartree for p in pts])
    imin   = int(np.argmin(E_vals))
    print("\nDone.")
    print(f"Minimum energy at R = {pts[imin].R_angstrom:.6f} Å  ({E_vals[imin]:.12f} Ha)")
    print(f"CSV written to: {args.output.resolve()}")

    if args.plot or args.plot_png:
        plot_pes(
            pts,
            atom1=args.atom1,
            atom2=args.atom2,
            xc=args.xc,
            basis=args.basis,
            output_png=args.plot_png,
        )


if __name__ == "__main__":
    main()
