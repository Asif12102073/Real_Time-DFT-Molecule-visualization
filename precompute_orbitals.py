"""
precompute_orbitals.py

Compute 2D electron density slices in the xz plane (y=0) at multiple
bond lengths using PySCF DFT.  Saves a .npz archive consumed by
taichi_ggui_vibration.py.
"""

# MUST USE DFT env with pyscf !!!!
"""
precompute_orbitals.py
 
Compute 2D electron density slices in the xz plane (y=0) at multiple
bond lengths using PySCF DFT.  Saves a .npz archive consumed by
taichi_ggui_vibration.py.
 
Molecule convention (same as diatomic_dft_pes.py):
  atom1 at z = 0,  atom2 at z = R_ang  (Angstrom, along z axis).
The 2D grid is centred at the bond midpoint each R.
 
Output keys in the .npz:
  R_ang     (nR,)          bond lengths [Angstrom]
  R_bohr    (nR,)          bond lengths [Bohr]
  rho       (nR, nx, nz)   electron density [e / bohr³]
  x_bohr    (nx,)          x grid  [-box, +box]  bohr
  z_rel     (nz,)          z grid relative to midpoint  [-box, +box]  bohr
  box_bohr  scalar         half-box size
  atom1, atom2             element strings
 
Example – HCl with 25 geometry snapshots:
  python precompute_orbitals.py \\
      --atom1 H --atom2 Cl --basis def2-svp --xc b3lyp \\
      --rmin 0.90 --rmax 2.20 --nR 25 \\
      --box 5.0 --grid 128 --output orbitals_hcl.npz
"""
 



from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── unit conversion ──────────────────────────────────────────────────────────
try:
    from scipy.constants import physical_constants
    ANG_TO_BOHR = 1e-10 / physical_constants["Bohr radius"]
except Exception:
    ANG_TO_BOHR = 1.0 / 0.529177210903


# ── DFT helpers ──────────────────────────────────────────────────────────────

def _build_mol(atom1: str, atom2: str, R_ang: float,
               basis: str, charge: int, spin: int, verbose: int):
    from pyscf import gto
    mol = gto.Mole()
    mol.atom   = f"{atom1} 0 0 0; {atom2} 0 0 {R_ang:.10f}"
    mol.unit   = "Angstrom"
    mol.basis  = basis
    mol.charge = charge
    mol.spin   = spin
    mol.verbose = verbose
    mol.build()
    return mol


def _run_dft(mol, xc: str, force_uks: bool = False, dm_guess: np.ndarray | None = None):
    from pyscf import dft as pdft
    
    # FIX: Allow UKS even for spin=0 to handle bond dissociation properly
    if mol.spin == 0 and not force_uks:
        mf = pdft.RKS(mol)
    else:
        mf = pdft.UKS(mol)
        
    mf.xc = xc
    mf.conv_tol  = 1e-9
    mf.max_cycle = 100
    if hasattr(mf, "grids"):
        mf.grids.level = 3
        
    # FIX: Use density matrix from previous step as initial guess if available
    if dm_guess is not None:
        mf.kernel(dm=dm_guess)
    else:
        mf.kernel()
        
    if not mf.converged:
        print(f"[warn] SCF not converged at R={mol.atom.split()[-1]} Å", file=sys.stderr)
    return mf


def _eval_density_2d(mol, dm_raw, box: float, ng: int,
                     R_bohr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate electron density on an ng × ng grid in the xz plane (y=0).
    Grid is centred at the bond midpoint z = R_bohr / 2.
    """
    # FIX: Guarantee dm is a numpy array even if PySCF returns a list/tuple
    dm = np.asarray(dm_raw)
    
    x_arr = np.linspace(-box, box, ng, dtype=np.float64)
    z_rel = np.linspace(-box, box, ng, dtype=np.float64)
    z_abs = z_rel + R_bohr / 2.0                             # absolute z in bohr

    X, Z = np.meshgrid(x_arr, z_abs, indexing="ij")          # (ng, ng)
    Y    = np.zeros_like(X)
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # (ng², 3)

    ao = mol.eval_gto("GTOval_sph", coords)                  # (ng², nao)

    if dm.ndim == 3:   # UKS: spin-summed density (shape: 2, nao, nao)
        rho = (np.einsum("pi,ij,pj->p", ao, dm, ao) +
               np.einsum("pi,ij,pj->p", ao, dm, ao))
    else:              # RKS: shape (nao, nao)
        rho = np.einsum("pi,ij,pj->p", ao, dm, ao)

    rho = np.maximum(rho, 0.0).reshape(ng, ng)               # (nx, nz)
    return (x_arr.astype(np.float32),
            z_rel.astype(np.float32),
            rho.astype(np.float32))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Precompute 2D DFT electron density slices for GGUI animation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--atom1",   default="H")
    p.add_argument("--atom2",   default="Cl")
    p.add_argument("--basis",   default="def2-svp")
    p.add_argument("--xc",      default="b3lyp")
    p.add_argument("--charge",  type=int,   default=0)
    p.add_argument("--spin",    type=int,   default=0)
    p.add_argument("--rmin",    type=float, default=0.90, help="Min R (Å)")
    p.add_argument("--rmax",    type=float, default=2.20, help="Max R (Å)")
    p.add_argument("--nR",      type=int,   default=25,   help="Number of snapshots")
    p.add_argument("--box",     type=float, default=5.0,  help="Half-box (bohr)")
    p.add_argument("--grid",    type=int,   default=128,  help="Grid points per axis")
    p.add_argument("--output",  type=Path,  default=Path("orbitals_hcl.npz"))
    p.add_argument("--force-uks", action="store_true",    help="Force UKS for open-shell dissociation")
    p.add_argument("--verbose", type=int,   default=0)
    args = p.parse_args()

    R_ang_arr  = np.linspace(args.rmin, args.rmax, args.nR, dtype=np.float32)
    R_bohr_arr = (R_ang_arr * ANG_TO_BOHR).astype(np.float32)
    rho_stack  = np.zeros((args.nR, args.grid, args.grid), dtype=np.float32)
    x_out = z_out = None
    
    # Track density matrix to use as initial guess for the next geometry
    last_dm = None

    for i, (R_ang, R_bohr) in enumerate(zip(R_ang_arr, R_bohr_arr)):
        print(f"[{i+1:03d}/{args.nR}]  R = {R_ang:.4f} Å  ({R_bohr:.4f} a₀)  ...",
              end="  ", flush=True)
        
        mol = _build_mol(args.atom1, args.atom2, float(R_ang),
                         args.basis, args.charge, args.spin, args.verbose)
        
        # Run DFT utilizing guess from previous step
        mf  = _run_dft(mol, args.xc, force_uks=args.force_uks, dm_guess=last_dm)
        
        # Save current density matrix for next loop iteration
        last_dm = mf.make_rdm1()
        
        x_arr, z_rel, rho = _eval_density_2d(mol, last_dm, args.box, args.grid, float(R_bohr))
        rho_stack[i] = rho
        x_out, z_out = x_arr, z_rel
        print(f"ρ_max = {rho.max():.4f} e/a₀³")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        R_ang    = R_ang_arr,
        R_bohr   = R_bohr_arr,
        rho      = rho_stack,
        x_bohr   = x_out,
        z_rel    = z_out,
        box_bohr = np.float32(args.box),
        atom1    = args.atom1,
        atom2    = args.atom2,
    )
    print(f"\n✓  Saved → {args.output.resolve()}")
    print(f"   rho stack: {rho_stack.shape}   global max = {rho_stack.max():.4f} e/a₀³")


if __name__ == "__main__":
    main()