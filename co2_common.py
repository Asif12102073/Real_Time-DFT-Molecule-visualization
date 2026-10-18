"""
co2_common.py

Shared utilities for the CO2 vibration pipeline.

This is the piece that replaces the diatomic pipeline's implicit
"R is the bond length, mu is the two-body reduced mass" assumption.
For CO2 (or any linear XY2), the correct generalized coordinate and
its effective mass come out of a proper harmonic normal-mode analysis
instead of being guessed by hand.

Unit convention (validated by finite-difference cross-check against
PySCF's own harmonic_analysis -- see the notes below):

    PySCF's thermo.harmonic_analysis returns, for each mode i:
      - norm_mode[i]       shape (natm, 3), units amu^-1/2
      - reduced_mass[i]    scalar, amu

    Define the unit-norm Cartesian displacement direction

        L[i] = norm_mode[i] * sqrt(reduced_mass[i])          (dimensionless)

    Then, for a scalar generalized coordinate q in bohr, displacing
    every atom as

        r_atom(q) = r_atom_eq + L[i][atom] * q

    reproduces the atoms' actual harmonic motion, and the effective
    mass appearing in the 1D nuclear Schrodinger equation for q is
    *exactly* reduced_mass[i] (in amu) -- the same quantity you'd pass
    as --mu-amu to a diatomic-style Numerov solver. This was checked
    numerically: displacing CO2's symmetric-stretch mode this way and
    fitting a parabola to E(q) reproduces PySCF's reported harmonic
    frequency to ~0.01%.

    q therefore plays exactly the role that R played for HCl: a
    literal, bohr-valued displacement coordinate with a well-defined
    effective mass. That's what lets the rest of the pipeline
    (numerov_vibrations.py in particular) carry over unmodified.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np

try:
    from scipy.constants import physical_constants
    from scipy.interpolate import CubicSpline
    from scipy.optimize import minimize_scalar
    BOHR_RADIUS_M = physical_constants["Bohr radius"][0]
    ANGSTROM_TO_BOHR = 1e-10 / BOHR_RADIUS_M
except Exception:
    ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR

MODE_NAMES = ("bend", "sym_stretch", "antisym_stretch")


@dataclass
class NormalMode:
    name: str                     # "bend" | "sym_stretch" | "antisym_stretch"
    freq_cm1: float                # harmonic frequency at this level of theory
    reduced_mass_amu: float        # effective mass for the q coordinate (see module docstring)
    L: np.ndarray                   # (natm, 3), unit-norm Cartesian direction
    symbols: List[str]
    eq_coords_bohr: np.ndarray      # (natm, 3), equilibrium geometry


def import_backend(use_gpu: bool):
    """Try gpu4pyscf first if requested; otherwise fall back to PySCF."""
    if use_gpu:
        try:
            from gpu4pyscf import gto, dft
            return "gpu4pyscf", gto, dft
        except Exception as exc:
            print(f"[warn] gpu4pyscf requested but unavailable ({exc}). "
                  "Falling back to PySCF.", file=sys.stderr)
    from pyscf import gto, dft
    return "pyscf", gto, dft


def _build_linear_co2(gto_mod, atom_c: str, atom_o: str, R_ang: float,
                       basis: str, charge: int, spin: int, verbose: int):
    mol = gto_mod.Mole()
    mol.atom = f"{atom_o} 0 0 -{R_ang:.10f}; {atom_c} 0 0 0; {atom_o} 0 0 {R_ang:.10f}"
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = verbose
    mol.build()
    return mol


def run_dft(mol, xc: str, dft_mod, force_uks: bool = False, dm_guess=None):
    if mol.spin == 0 and not force_uks:
        mf = dft_mod.RKS(mol)
    else:
        mf = dft_mod.UKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    if hasattr(mf, "grids"):
        mf.grids.level = 3
    if dm_guess is not None:
        mf.kernel(dm=dm_guess)
    else:
        mf.kernel()
    if not mf.converged:
        print("[warn] SCF not converged", file=sys.stderr)
    return mf


def find_equilibrium_R(atom_c="C", atom_o="O", basis="def2-svp", xc="b3lyp",
                        charge=0, spin=0, rmin=1.05, rmax=1.30, npts=9,
                        use_gpu=False, verbose=0) -> float:
    """
    Scan the symmetric O-C-O bond length and locate the minimum.

    CO2's closed-shell ground state is linear and D_inf_h symmetric, so
    a 1D scan along the totally symmetric coordinate is enough to find
    the equilibrium geometry -- unlike a general polyatomic, we don't
    need a full multi-dimensional geometry optimizer here. (Skipping
    this step and guessing a literature bond length is a real trap:
    at a mismatched geometry the Hessian can return small *imaginary*
    bending frequencies purely as a geometry artifact -- this was
    reproduced during development at the sto-3g level.)
    """
    backend, gto_mod, dft_mod = import_backend(use_gpu)
    Rs = np.linspace(rmin, rmax, npts)
    Es = []
    for R in Rs:
        mol = _build_linear_co2(gto_mod, atom_c, atom_o, float(R), basis, charge, spin, verbose)
        mf = run_dft(mol, xc, dft_mod)
        Es.append(mf.e_tot)
    Es = np.asarray(Es)
    cs = CubicSpline(Rs, Es)
    res = minimize_scalar(cs, bounds=(float(Rs[0]), float(Rs[-1])), method="bounded")
    if not res.success:
        raise RuntimeError("Could not refine the CO2 equilibrium bond length")
    return float(res.x)


def compute_normal_modes(R_eq_ang, atom_c="C", atom_o="O", basis="def2-svp",
                          xc="b3lyp", charge=0, spin=0, use_gpu=False, verbose=0):
    """
    Run a harmonic frequency (Hessian) calculation at the CO2
    equilibrium geometry and return the three physically distinct
    vibrational modes: one representative of the doubly-degenerate
    bend (rotated to lie in the xz-plane -- see below), plus the
    symmetric and antisymmetric stretches.

    Mode identification is by frequency ordering (bend < sym stretch <
    antisym stretch, which is CO2's actual physical ordering and was
    confirmed during development), with a sanity check that the
    "symmetric stretch" mode really does leave the central atom
    stationary.

    Returns (modes_dict, E_eq_hartree).
    """
    backend, gto_mod, dft_mod = import_backend(use_gpu)
    from pyscf.hessian import thermo

    mol = _build_linear_co2(gto_mod, atom_c, atom_o, R_eq_ang, basis, charge, spin, verbose)
    mf = run_dft(mol, xc, dft_mod)
    hess = mf.Hessian().kernel()
    info = thermo.harmonic_analysis(mol, hess)

    freqs = np.real(np.asarray(info["freq_wavenumber"]))
    if len(freqs) != 4:
        raise RuntimeError(
            f"Expected 4 vibrational modes for a linear triatomic (3N-5), got "
            f"{len(freqs)}. Check that the geometry is linear and closed-shell."
        )
    order = np.argsort(freqs)               # -> bend, bend, sym stretch, antisym stretch
    i_bend0, i_bend1, i_sym, i_anti = order

    norm_mode = info["norm_mode"]            # (4, natm, 3), amu^-1/2
    reduced_mass = info["reduced_mass"]      # (4,), amu
    eq_coords = mol.atom_coords()            # bohr
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    def make_L(idx):
        return norm_mode[idx] * np.sqrt(reduced_mass[idx])

    # Rotate the degenerate bend pair so one partner is purely
    # polarized along x, i.e. lies in the xz-plane the density script
    # slices. Any orthonormal combination of two degenerate
    # eigenvectors is itself a valid normal mode with the same
    # frequency and reduced mass, so this doesn't change the physics
    # -- only which of the two equivalent bending directions we call
    # "the" bend.
    L0, L1 = make_L(i_bend0), make_L(i_bend1)
    y0, y1 = L0[:, 1], L1[:, 1]
    ref = int(np.argmax(np.abs(y0) + np.abs(y1)))
    theta = np.arctan2(-y0[ref], y1[ref])
    L_bend = np.cos(theta) * L0 + np.sin(theta) * L1
    residual_y = float(np.max(np.abs(L_bend[:, 1])))
    if residual_y > 1e-6:
        print(f"[warn] bend mode not cleanly in-plane after rotation "
              f"(residual y-component {residual_y:.2e}); the density "
              "slice will miss part of the bending motion.", file=sys.stderr)

    modes = {
        "bend": NormalMode("bend", float(freqs[i_bend0]), float(reduced_mass[i_bend0]),
                            L_bend, symbols, eq_coords),
        "sym_stretch": NormalMode("sym_stretch", float(freqs[i_sym]), float(reduced_mass[i_sym]),
                                   make_L(i_sym), symbols, eq_coords),
        "antisym_stretch": NormalMode("antisym_stretch", float(freqs[i_anti]), float(reduced_mass[i_anti]),
                                       make_L(i_anti), symbols, eq_coords),
    }

    c_disp = float(np.linalg.norm(modes["sym_stretch"].L[1]))
    if c_disp > 0.05:
        print(f"[warn] central atom moves by {c_disp:.3f} in the mode assigned "
              "as sym_stretch -- mode ordering may be off for this basis/functional; "
              "inspect the printed frequencies before trusting the scan.", file=sys.stderr)

    return modes, float(mf.e_tot)
