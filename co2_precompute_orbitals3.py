"""
co2_precompute_orbitals.py

Generalized version of precompute_orbitals.py for CO2 (or any linear
XY2). Reads the PES CSV + .meta.json written by co2_dft_pes.py,
reconstructs the exact 3-atom Cartesian geometry at every scan point
(eq_coords_bohr + L * q), and evaluates the 2D electron density in
the xz-plane (y=0) at each one -- same density-matrix warm-start
approach as precompute_orbitals.py, just with N atoms instead of 2.

Grid convention: unlike the diatomic script, the grid here is centred
at a FIXED point (x=0, z=0 -- the equilibrium carbon position), not
re-centred on a moving bond midpoint each frame. With three atoms
that can all move (antisymmetric stretch, bend), "the bond midpoint"
isn't a single well-defined point any more, so a fixed frame large
enough to contain the whole scan range is simpler and unambiguous.

The output .npz stores every atom's (x, z) position at each snapshot
(atom_x_bohr, atom_z_bohr, shape (nR, natm)) so the renderer can draw
all three nuclei, not just track one against a formula.

Example:
    python co2_precompute_orbitals3.py --pes-csv pes_co2.csv \\
        --box 6.0 --grid 128 --output orbitals_co2.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from co2_common import import_backend, run_dft, BOHR_TO_ANGSTROM


def read_pes_csv(path: Path):
    qs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            qs.append(float(row["R_bohr"]))
    return np.asarray(qs, dtype=np.float64)


def eval_density_2d(mol, dm_raw, box: float, ng: int):
    """
    Evaluate electron density on an ng x ng grid in the xz plane (y=0),
    centred at the fixed origin (0, 0).
    """
    dm = np.asarray(dm_raw)

    x_arr = np.linspace(-box, box, ng, dtype=np.float64)
    z_arr = np.linspace(-box, box, ng, dtype=np.float64)
    X, Z = np.meshgrid(x_arr, z_arr, indexing="ij")
    Y = np.zeros_like(X)
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    ao = mol.eval_gto("GTOval_sph", coords)

    if dm.ndim == 3:  # UKS: sum both spin channels
        rho = (np.einsum("pi,ij,pj->p", ao, dm[0], ao) +
               np.einsum("pi,ij,pj->p", ao, dm[1], ao))
    else:              # RKS
        rho = np.einsum("pi,ij,pj->p", ao, dm, ao)

    rho = np.maximum(rho, 0.0).reshape(ng, ng)
    return (x_arr.astype(np.float32), z_arr.astype(np.float32), rho.astype(np.float32))


def main():
    p = argparse.ArgumentParser(
        description="Precompute 2D DFT electron density slices for the CO2 GGUI animation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pes-csv", type=Path, required=True,
                    help="CSV written by co2_dft_pes.py (uses its R_bohr column as q)")
    p.add_argument("--meta", type=Path, default=None,
                    help="meta.json written alongside the CSV (default: <pes-csv>.meta.json)")
    p.add_argument("--box", type=float, default=6.0, help="Half-box (bohr), fixed frame")
    p.add_argument("--grid", type=int, default=128, help="Grid points per axis")
    p.add_argument("--output", type=Path, default=Path("orbitals_co2.npz"))
    p.add_argument("--force-uks", action="store_true")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    meta_path = args.meta or args.pes_csv.with_suffix(args.pes_csv.suffix + ".meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    symbols = meta["symbols"]
    natm = len(symbols)
    eq_coords_bohr = np.asarray(meta["eq_coords_bohr"], dtype=np.float64)
    L = np.asarray(meta["L"], dtype=np.float64)
    basis, xc = meta["basis"], meta["xc"]
    charge, spin = meta["charge"], meta["spin"]

    qs = read_pes_csv(args.pes_csv)
    nR = len(qs)
    print(f"Loaded {nR} snapshots for mode '{meta['mode']}' from {args.pes_csv}")

    max_extent = float(np.max(np.abs(eq_coords_bohr)) +
                        np.max(np.abs(L)) * np.max(np.abs(qs)))
    if max_extent > args.box * 0.9:
        print(f"[warn] atoms may extend to ~{max_extent:.2f} bohr, close to or beyond "
              f"--box {args.box}; consider increasing --box.", file=sys.stderr)

    backend, gto_mod, dft_mod = import_backend(args.use_gpu)

    rho_stack = np.zeros((nR, args.grid, args.grid), dtype=np.float32)
    atom_x_bohr = np.zeros((nR, natm), dtype=np.float32)
    atom_z_bohr = np.zeros((nR, natm), dtype=np.float32)
    x_out = z_out = None
    last_dm = None

    for i, q in enumerate(qs):
        coords_bohr = eq_coords_bohr + L * q
        atom_x_bohr[i] = coords_bohr[:, 0]
        atom_z_bohr[i] = coords_bohr[:, 2]

        atom_spec = "; ".join(
            f"{symbols[a]} {coords_bohr[a, 0]:.10f} {coords_bohr[a, 1]:.10f} {coords_bohr[a, 2]:.10f}"
            for a in range(natm)
        )
        print(f"[{i+1:03d}/{nR}]  q = {q:+.4f} bohr  ...", end="  ", flush=True)

        mol = gto_mod.Mole()
        mol.atom = atom_spec
        mol.unit = "Bohr"
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.verbose = args.verbose
        mol.build()

        mf = run_dft(mol, xc, dft_mod, force_uks=args.force_uks, dm_guess=last_dm)
        last_dm = mf.make_rdm1()

        x_arr, z_arr, rho = eval_density_2d(mol, last_dm, args.box, args.grid)
        rho_stack[i] = rho
        x_out, z_out = x_arr, z_arr
        print(f"\u03c1_max = {rho.max():.4f} e/a\u2080\u00b3")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        R_bohr=qs.astype(np.float32),          # generalized displacement coordinate q, kept as "R_bohr" for naming symmetry with the diatomic pipeline
        R_ang=(qs * BOHR_TO_ANGSTROM).astype(np.float32),
        rho=rho_stack,
        x_bohr=x_out,
        z_bohr=z_out,
        box_bohr=np.float32(args.box),
        natm=natm,
        symbols=np.array(symbols),
        atom_x_bohr=atom_x_bohr,
        atom_z_bohr=atom_z_bohr,
        mode=meta["mode"],
        reduced_mass_amu=np.float32(meta["reduced_mass_amu"]),
    )
    print(f"\n\u2713  Saved \u2192 {args.output.resolve()}")
    print(f"   rho stack: {rho_stack.shape}   global max = {rho_stack.max():.4f} e/a\u2080\u00b3")


if __name__ == "__main__":
    main()
