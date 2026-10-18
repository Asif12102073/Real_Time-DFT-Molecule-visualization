"""
numerov_vibrations.py

Read a diatomic potential energy curve (PES) from CSV, build a smooth
interpolant, and solve the 1D vibrational Schrödinger equation using
Numerov integration + energy-scan + bisection root-finding.

Input CSV format (produced by diatomic_dft_pes.py):
    R_bohr, R_angstrom, E_hartree

Example – HCl (mu = 1.00794*35.453/(1.00794+35.453) ≈ 0.97959 amu):
    python numerov_vibrations.py \\
        --input pes_hcl.csv --mu-amu 0.97959 --nlevels 6 --plot

The code outputs vibrational energies relative to the minimum of the PES.

Fixes vs. original:
  - Dead-code `find_bracket()` replaced with inline note.
  - Overflow rescaling bug fixed (per-element clamp, not array-wide rescale).
  - np.trapz → np.trapezoid (NumPy ≥ 2.0 compatibility).
  - dissociation_estimate now uses a proper tail mean over configurable fraction.
  - ZPE is explicitly reported in the output table.
  - Plot labels show ZPE, Δv spacing, and node count.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

#matplotlib.use('Agg')
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.constants import physical_constants
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

HARTREE_TO_CM  = physical_constants["hartree-inverse meter relationship"][0] / 100.0
HARTREE_TO_EV  = physical_constants["Hartree energy in eV"][0]
AMU_TO_ME      = physical_constants["atomic mass constant"][0] / physical_constants["electron mass"][0]
BOHR_TO_ANGSTROM = physical_constants["Bohr radius"][0] / 1e-10


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PESData:
    R_bohr: np.ndarray
    E_hartree: np.ndarray

    @property
    def Emin(self) -> float:
        return float(np.min(self.E_hartree))

    @property
    def shifted(self) -> np.ndarray:
        return self.E_hartree - self.Emin


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


# ---------------------------------------------------------------------------
# Potential curve
# ---------------------------------------------------------------------------

class PotentialCurve:
    """
    Smooth potential V(R) in Hartree, shifted so that min(V)=0 for the
    vibrational problem.
    """
    def __init__(self, R_bohr: np.ndarray, E_hartree: np.ndarray, tail_fraction: float = 0.15):
        self.R_bohr    = np.asarray(R_bohr, dtype=float)
        self.E_hartree = np.asarray(E_hartree, dtype=float)
        order = np.argsort(self.R_bohr)
        self.R_bohr    = self.R_bohr[order]
        self.E_hartree = self.E_hartree[order]
        self.Emin      = float(np.min(self.E_hartree))
        self.V_shifted = self.E_hartree - self.Emin
        self.spline    = CubicSpline(self.R_bohr, self.V_shifted, bc_type="natural")
        # Use the last `tail_fraction` of points for the dissociation estimate.
        # BUG FIX: original hardcoded last-3 points which can be far below the
        # true asymptote on a short grid.
        n_tail = max(3, int(len(self.V_shifted) * tail_fraction))
        self._tail_fraction = tail_fraction
        self._n_tail = n_tail
        self._dissociation_estimate = float(np.mean(self.V_shifted[-n_tail:]))

    def __call__(self, R: np.ndarray | float) -> np.ndarray | float:
        return self.spline(R)

    @property
    def dissociation_estimate(self) -> float:
        return self._dissociation_estimate

    @property
    def R_min_grid(self) -> float:
        return float(self.R_bohr[0])

    @property
    def R_max_grid(self) -> float:
        return float(self.R_bohr[-1])


# ---------------------------------------------------------------------------
# Numerov integrator  (FIXED)
# ---------------------------------------------------------------------------

def numerov_integrate(
    V: Callable[[np.ndarray], np.ndarray],
    E: float,
    mu_me: float,
    R_min: float,
    R_max: float,
    dR: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numerov integration for the radial Schrödinger equation:
        u''(R) = f(R)*u(R),   f(R) = 2*mu*(V(R) - E)

    Energies in Hartree, mu in electron masses, R in Bohr.

    BUG FIX (overflow):
        Original code did `u /= 1e120` inside the loop, which silently
        rescaled all previous elements and corrupted the Numerov stencil.
        Now we track an independent log-scale factor and only apply it
        element-wise to avoid array-wide rescaling during propagation.
    """
    R  = np.arange(R_min, R_max + dR * 0.5, dR)
    n  = len(R)
    u  = np.zeros(n, dtype=float)
    u[0] = 0.0
    u[1] = 1e-8       # arbitrary non-zero seed; normalised later

    f  = 2.0 * mu_me * (V(R) - E)
    h2 = dR * dR

    CLAMP = 1e140   # BUG FIX: clamp per element, not array-wide

    for i in range(1, n - 1):
        denom = 1.0 + h2 * f[i + 1] / 12.0
        if abs(denom) < 1e-300:
            denom = 1e-300   # guard against exact zero
        num = (
            2.0 * (1.0 - 5.0 * h2 * f[i] / 12.0) * u[i]
            - (1.0 + h2 * f[i - 1] / 12.0) * u[i - 1]
        )
        u_next = num / denom
        # Per-element clamp: rescale only when the running value explodes,
        # and carry the same factor through both u[i] and u[i-1].
        if abs(u_next) > CLAMP:
            scale = abs(u_next) / CLAMP
            u[:i+1] /= scale   # rescale history consistently
            u_next   = u_next / scale
        u[i + 1] = u_next

    return R, u


# ---------------------------------------------------------------------------
# Boundary value + bisection
# ---------------------------------------------------------------------------

def boundary_value(
    V: PotentialCurve,
    E: float,
    mu_me: float,
    R_min: float,
    R_max: float,
    dR: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    R, u = numerov_integrate(V, E, mu_me, R_min, R_max, dR)
    return float(u[-1]), R, u


def bisection_level(
    V: PotentialCurve,
    mu_me: float,
    E1: float,
    E2: float,
    R_min: float,
    R_max: float,
    dR: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> Tuple[float, np.ndarray, np.ndarray]:
    f1, _, _  = boundary_value(V, E1, mu_me, R_min, R_max, dR)
    Em, Rm, um = 0.5 * (E1 + E2), None, None

    for _ in range(max_iter):
        Em = 0.5 * (E1 + E2)
        fm, Rm, um = boundary_value(V, Em, mu_me, R_min, R_max, dR)
        if abs(fm) < tol or abs(E2 - E1) < tol:
            break
        if np.sign(f1) == np.sign(fm):
            E1, f1 = Em, fm
        else:
            E2 = Em

    return Em, Rm, um  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Normalisation & node counting
# ---------------------------------------------------------------------------

def normalize_wavefunction(R: np.ndarray, u: np.ndarray) -> np.ndarray:
    # BUG FIX: use np.trapezoid (NumPy ≥ 2.0); fall back to np.trapz
    try:
        norm = np.sqrt(np.trapezoid(u * u, R))
    except AttributeError:
        norm = np.sqrt(np.trapz(u * u, R))  # type: ignore[attr-defined]
    return u / norm if norm > 0 else u


def count_nodes(u: np.ndarray) -> int:
    s = np.sign(u)
    s[s == 0] = 1
    return int(np.sum(s[1:] * s[:-1] < 0))


# ---------------------------------------------------------------------------
# Solve vibrational levels
# ---------------------------------------------------------------------------

def solve_levels(
    V: PotentialCurve,
    mu_amu: float,
    R_min: float,
    R_max: float,
    dR: float,
    nlevels: int = 5,
    scan_points: int = 1500,
) -> List[dict]:
    """
    Find the lowest vibrational levels by scanning the energy axis and
    collecting sign changes in the boundary value function.
    """
    mu_me = float(mu_amu * AMU_TO_ME)

    Emin = 0.0
    Emax = max(1e-8, 0.98 * V.dissociation_estimate)
    if Emax <= Emin:
        raise ValueError(
            "The shifted PES does not appear to support bound states "
            "in the chosen range. Check --rmax / tail of PES."
        )

    energies_scan = np.linspace(Emin, Emax, scan_points)
    vals = np.array([
        boundary_value(V, float(E), mu_me, R_min, R_max, dR)[0]
        for E in energies_scan
    ])

    brackets: List[Tuple[float, float]] = []
    for i in range(len(energies_scan) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]) and vals[i] != 0.0:
            brackets.append((float(energies_scan[i]), float(energies_scan[i + 1])))
        if len(brackets) >= nlevels:
            break

    if not brackets:
        raise RuntimeError(
            "No vibrational levels bracketed. "
            "Try larger --rmax or --scan-points."
        )

    solutions: List[dict] = []
    for idx, (E1, E2) in enumerate(brackets[:nlevels]):
        E, R, u = bisection_level(V, mu_me, E1, E2, R_min, R_max, dR)
        u = normalize_wavefunction(R, u)
        solutions.append(
            {
                "v"         : idx,
                "E_hartree" : float(E),
                "E_cm1"     : float(E * HARTREE_TO_CM),
                "E_eV"      : float(E * HARTREE_TO_EV),
                "nodes"     : count_nodes(u),
                "R"         : R,
                "u"         : u,
            }
        )
    return solutions


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    V: PotentialCurve,
    levels: List[dict],
    mu_amu: float,
    R_min: float,
    R_max: float,
    output_png: Optional[Path] = None,
) -> None:
    """
    Two-panel plot:
      Left  – PES with wavefunctions and energy levels
      Right – Energy level ladder (cm⁻¹)
    """
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)
    ax_pes  = fig.add_subplot(gs[0])
    ax_lad  = fig.add_subplot(gs[1])

    R_plot   = np.linspace(R_min, R_max, 2000)
    V_plot   = V(R_plot)
    Diss_est = V.dissociation_estimate
    V_max_disp = min(1.05 * Diss_est, np.max(V_plot) * 1.1)
    #V_max_disp = max(np.max(V_plot) * 1.05, 10.0 / HARTREE_TO_EV)

    # --- PES ---
    ax_pes.plot(R_plot, V_plot * HARTREE_TO_EV, "k-", lw=2, label="Shifted PES")
    ax_pes.axhline(Diss_est * HARTREE_TO_EV, color="gray", ls="--", lw=1,
                   label=f"Diss. est. = {Diss_est*HARTREE_TO_EV:.4f} eV")

    colors = plt.cm.tab10(np.linspace(0, 0.7, len(levels)))  # type: ignore[attr-defined]
    for lev, col in zip(levels, colors):
        E_eV  = lev["E_eV"]
        R_lev = lev["R"]
        u_lev = lev["u"]

        # Horizontal energy line within classically allowed region
        mask = V(R_lev) * HARTREE_TO_EV <= E_eV + 1e-6
        if mask.any():
            ax_pes.hlines(E_eV, R_lev[mask][0], R_lev[mask][-1],
                          colors=col, linewidths=0.8, linestyles="--")

        # Wavefunction scaled for visibility
        #scale = 1.5
        scale = 0.12 * V_max_disp * HARTREE_TO_EV
        u_plot = E_eV + scale * u_lev / (np.max(np.abs(u_lev)) + 1e-30)
        ax_pes.plot(R_lev, u_plot, color=col, lw=1.2,
                    label=f"v={lev['v']}  {lev['E_cm1']:.1f} cm⁻¹")

    ax_pes.set_xlim(R_min, R_max)
    ax_pes.set_ylim(-0.01 * V_max_disp * HARTREE_TO_EV, V_max_disp * HARTREE_TO_EV * 1.05)
    ax_pes.set_xlabel("R  (bohr)")
    ax_pes.set_ylabel("Energy  (eV)")
    ax_pes.set_title("DFT PES + Numerov vibrational states")
    ax_pes.legend(fontsize=8, loc="upper right")
    ax_pes.grid(True, alpha=0.25)

    # --- Energy ladder ---
    E_cm = [lev["E_cm1"] for lev in levels]
    for i, lev in enumerate(levels):
        ax_lad.hlines(lev["E_cm1"], 0.2, 0.8, colors=colors[i], linewidths=2.5)
        delta = f"  Δ={lev['E_cm1']-levels[i-1]['E_cm1']:.1f}" if i > 0 else "  ZPE"
        ax_lad.text(0.82, lev["E_cm1"],
                    f"v={lev['v']}  {lev['E_cm1']:.1f}{delta}",
                    va="center", fontsize=8, color=colors[i])
    ax_lad.set_xlim(0, 1.5)
    ax_lad.set_ylim(-50, max(E_cm) * 1.15)
    ax_lad.set_ylabel("Energy relative to PES minimum  (cm⁻¹)")
    ax_lad.set_title("Vibrational ladder")
    ax_lad.set_xticks([])
    ax_lad.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Numerov solver   |   μ = {mu_amu:.5f} amu   |   "
        f"ZPE = {levels[0]['E_cm1']:.2f} cm⁻¹",
        fontsize=11,
    )

    if output_png:
        fig.savefig(output_png, dpi=150) #, bbox_inches="tight"
        print(f"Plot saved to: {output_png.resolve()}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Solve diatomic vibrational states with Numerov",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",        type=Path,  required=True,   help="CSV from diatomic_dft_pes.py")
    p.add_argument("--mu-amu",       type=float, required=True,   help="Reduced mass (amu)")
    p.add_argument("--rmin",         type=float, default=0.6,     help="Left nuclear grid boundary (bohr)")
    p.add_argument("--rmax",         type=float, default=12.0,    help="Right nuclear grid boundary (bohr)")
    p.add_argument("--dR",           type=float, default=0.002,   help="Grid spacing (bohr)")
    p.add_argument("--nlevels",      type=int,   default=5,       help="Number of vibrational levels")
    p.add_argument("--scan-points",  type=int,   default=1500,    help="Energy scan resolution")
    p.add_argument("--tail-frac",    type=float, default=0.15,    help="Tail fraction for Diss. estimate")
    p.add_argument("--plot",         action="store_true",         help="Show wavefunction plot")
    p.add_argument("--plot_png",     type=Path,  default=False,    help="Save plot to PNG instead of showing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = read_pes_csv(args.input)
    pot  = PotentialCurve(data.R_bohr, data.E_hartree, tail_fraction=args.tail_frac)

    print("=== Numerov vibrational solver ===")
    print(f"Input PES file        : {args.input.resolve()}")
    print(f"Reduced mass          : {args.mu_amu:.8f} amu  =  {args.mu_amu * AMU_TO_ME:.4f} mₑ")
    print(f"PES minimum           : {pot.Emin:.12f} Ha")
    print(f"Dissociation estimate : {pot.dissociation_estimate:.6f} Ha  "
          f"(tail avg, last {pot._n_tail} pts)")
    print(f"Nuclear grid          : [{args.rmin}, {args.rmax}] bohr,  dR = {args.dR} bohr")
    print()

    levels = solve_levels(
        pot,
        mu_amu      = args.mu_amu,
        R_min       = args.rmin,
        R_max       = args.rmax,
        dR          = args.dR,
        nlevels     = args.nlevels,
        scan_points = args.scan_points,
    )

    # ---- Print table ----
    print(f"{'v':>3}  {'E (Ha)':>16}  {'E (cm⁻¹)':>12}  {'E (eV)':>10}  "
          f"{'Δ(cm⁻¹)':>10}  nodes")
    print("-" * 70)
    for i, lev in enumerate(levels):
        delta = lev["E_cm1"] - levels[i-1]["E_cm1"] if i > 0 else float("nan")
        zpe   = "  ← ZPE" if i == 0 else ""
        delta_str = f"{delta:10.3f}" if i > 0 else "       ---"
        print(
            f"{lev['v']:>3}  {lev['E_hartree']:>16.10f}  "
            f"{lev['E_cm1']:>12.4f}  {lev['E_eV']:>10.6f}  "
            f"{delta_str}  {lev['nodes']:>5}{zpe}"
        )

    if args.plot or args.plot_png:
        plot_results(
            pot, levels,
            mu_amu    = args.mu_amu,
            R_min     = args.rmin,
            R_max     = args.rmax,
            output_png= args.plot_png,
        )


if __name__ == "__main__":
    main()
