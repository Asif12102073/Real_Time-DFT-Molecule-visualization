"""
co2_dft_pes.py

Generalized version of diatomic_dft_pes.py for CO2 (or any linear XY2
you point it at). Instead of scanning a literal bond length R, this
scans a normal-mode coordinate q (bohr) along one of CO2's three
physically distinct vibrational modes:

    --mode sym_stretch      nu1, ~1333 cm-1, IR-inactive, C stationary
    --mode antisym_stretch  nu3, ~2349 cm-1, IR-active,   C moves too
    --mode bend             nu2, ~667 cm-1,  IR-active,   doubly degenerate

q is a Cartesian displacement in bohr along that mode's normal-mode
eigenvector (see co2_common.py's module docstring for the exact
convention and why it's the right generalization of "R").

The output CSV uses the *same* column names as diatomic_dft_pes.py
(R_bohr, R_angstrom, E_hartree) so numerov_vibrations2.py works on it
completely unmodified -- q just plays the role R used to play; you
only need to pass the correct --mu-amu (printed at the end of this
script, and saved in the .meta.json sidecar).

The .meta.json also records the equilibrium geometry and the mode's
displacement vector L, so co2_precompute_orbitals.py can reconstruct
the exact 3-atom geometry at every scan point without recomputing the
Hessian.

Example:
    python co2_dft_pes.py --mode sym_stretch --basis def2-svp --xc b3lyp \\
        --qmin -0.6 --qmax 0.6 --points 31 --output pes_co2_sym.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from co2_common import (
    MODE_NAMES, BOHR_TO_ANGSTROM,
    import_backend, run_dft, find_equilibrium_R, compute_normal_modes,
)


def write_csv(path: Path, qs_bohr: np.ndarray, Es_hartree: np.ndarray):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["R_bohr", "R_angstrom", "E_hartree"])
        for q, E in zip(qs_bohr, Es_hartree):
            w.writerow([f"{q:.10f}", f"{q * BOHR_TO_ANGSTROM:.10f}", f"{E:.12f}"])


def write_meta(path: Path, args, mode, R_eq_ang: float, E_eq: float):
    meta = {
        "mode": mode.name,
        "reduced_mass_amu": mode.reduced_mass_amu,
        "freq_cm1_harmonic": mode.freq_cm1,
        "R_eq_angstrom": R_eq_ang,
        "E_eq_hartree": E_eq,
        "basis": args.basis,
        "xc": args.xc,
        "charge": args.charge,
        "spin": args.spin,
        "atom_c": args.atom_c,
        "atom_o": args.atom_o,
        "symbols": mode.symbols,
        "eq_coords_bohr": mode.eq_coords_bohr.tolist(),
        "L": mode.L.tolist(),
        "note": ("q (aka 'R_bohr' in the CSV, for numerov_vibrations2.py "
                 "compatibility) is a Cartesian displacement in bohr along "
                 "L: atom_position(q) = eq_coords_bohr + L * q"),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    p = argparse.ArgumentParser(
        description="Scan CO2 energy along one normal-mode coordinate using PySCF DFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=list(MODE_NAMES), default="sym_stretch")
    p.add_argument("--atom-c", default="C")
    p.add_argument("--atom-o", default="O")
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--xc", default="b3lyp")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--req-scan-min", type=float, default=1.05, help="Å, for locating R_eq")
    p.add_argument("--req-scan-max", type=float, default=1.30, help="Å, for locating R_eq")
    p.add_argument("--req-scan-points", type=int, default=9)
    p.add_argument("--qmin", type=float, default=-0.6, help="min normal-mode displacement (bohr)")
    p.add_argument("--qmax", type=float, default=0.6, help="max normal-mode displacement (bohr)")
    p.add_argument("--points", type=int, default=31)
    p.add_argument("--output", type=Path, default=Path("pes_co2.csv"))
    p.add_argument("--force-uks", action="store_true")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    print(f"[1/3] Locating CO2 equilibrium geometry ({args.basis}/{args.xc}) ...")
    R_eq = find_equilibrium_R(args.atom_c, args.atom_o, args.basis, args.xc,
                               args.charge, args.spin, args.req_scan_min,
                               args.req_scan_max, args.req_scan_points,
                               args.use_gpu, args.verbose)
    print(f"      R_eq = {R_eq:.5f} \u00c5")

    print("[2/3] Harmonic frequency analysis (Hessian) at equilibrium ...")
    modes, E_eq = compute_normal_modes(R_eq, args.atom_c, args.atom_o, args.basis,
                                        args.xc, args.charge, args.spin,
                                        args.use_gpu, args.verbose)
    for name in MODE_NAMES:
        m = modes[name]
        flag = "   <-- scanning this one" if name == args.mode else ""
        print(f"      {name:16s} {m.freq_cm1:9.2f} cm-1   mu_eff = {m.reduced_mass_amu:8.4f} amu{flag}")

    mode = modes[args.mode]
    natm = len(mode.symbols)

    print(f"[3/3] Scanning E(q) along '{args.mode}', q in "
          f"[{args.qmin}, {args.qmax}] bohr, {args.points} points ...")

    backend, gto_mod, dft_mod = import_backend(args.use_gpu)
    qs = np.linspace(args.qmin, args.qmax, args.points)
    Es = np.zeros(args.points)
    last_dm = None
    for i, q in enumerate(qs, start=1):
        coords_bohr = mode.eq_coords_bohr + mode.L * q
        atom_spec = "; ".join(
            f"{mode.symbols[a]} {coords_bohr[a, 0]:.10f} {coords_bohr[a, 1]:.10f} {coords_bohr[a, 2]:.10f}"
            for a in range(natm)
        )
        molq = gto_mod.Mole()
        molq.atom = atom_spec
        molq.unit = "Bohr"
        molq.basis = args.basis
        molq.charge = args.charge
        molq.spin = args.spin
        molq.verbose = args.verbose
        molq.build()
        mf = run_dft(molq, args.xc, dft_mod, force_uks=args.force_uks, dm_guess=last_dm)
        last_dm = mf.make_rdm1()
        Es[i - 1] = mf.e_tot
        print(f"[{i:03d}/{args.points}]  q = {q:+.4f} bohr   E = {mf.e_tot:.10f} Ha")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, qs, Es)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    write_meta(meta_path, args, mode, R_eq, E_eq)

    print(f"\n\u2713  PES  \u2192 {args.output.resolve()}")
    print(f"\u2713  meta \u2192 {meta_path.resolve()}")
    print(f"\nFor numerov_vibrations2.py, use:  --mu-amu {mode.reduced_mass_amu:.6f}")


if __name__ == "__main__":
    main()
