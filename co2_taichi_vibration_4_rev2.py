"""
co2_taichi_vibration.py

Generalized version of taichi_ggui_vibration3.py for CO2 (or any
linear XY2). Loads the electron-density stack + per-atom nucleus
positions from co2_precompute_orbitals.py and the PES from
co2_dft_pes.py, solves the 1D nuclear vibrational problem along the
scanned normal-mode coordinate q (same finite-difference approach as
the diatomic script -- only the reduced mass, loaded from the
orbitals file, changes), and renders the moving electron density with
all N nuclei tracked and drawn (not just one).

Grid convention: the density grid is a FIXED frame (co2_precompute_orbitals.py
centres it at the equilibrium carbon position), so nucleus pixel
positions are a plain linear map from bohr -> pixel index -- no
per-frame recentring formula like the diatomic script's j_H needed.

Run interactively:
    python co2_taichi_vibration.py --orbitals orbitals_co2_sym.npz --pes pes_co2_sym.csv

Smoke-test without a GPU/display (renders one frame to a PNG and exits):
    python co2_taichi_vibration.py --orbitals orbitals_co2_sym.npz --pes pes_co2_sym.csv \\
        --cpu --headless-test frame.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import taichi as ti
from scipy.interpolate import CubicSpline

HARTREE_TO_EV = 27.211386
AMU_TO_ME = 1822.888486192
ANG_TO_BOHR = 1.8897261246257702

# Per-element color scheme (core, glow), chosen to stay distinct from
# the cyan electron cloud and the orange wavepacket-plot trace.
ELEMENT_COLORS = {
    "O": ((1.00, 0.15, 0.10), (0.80, 0.00, 0.00)),
    "C": ((1.00, 0.85, 0.20), (0.75, 0.55, 0.05)),
    "N": ((0.10, 0.20, 1.00), (0.00, 0.05, 0.80)),
    "H": ((0.95, 0.95, 0.95), (0.70, 0.70, 0.70)),
}
DEFAULT_COLOR = ((0.9, 0.9, 0.9), (0.6, 0.6, 0.6))


def solve_vibrational_states(R_bohr_stack, pot_spline, E_min, mu_amu):
    N_grid = 1000
    R_array = np.asarray(R_bohr_stack).flatten()
    R_min = float(R_array[0])
    R_max = float(R_array[-1])
    R_space = np.linspace(R_min, R_max, N_grid)
    dR = float(R_space[1] - R_space[0])
    mu = float(mu_amu * AMU_TO_ME)

    V = pot_spline(R_space).flatten() - float(E_min)

    H = np.zeros((N_grid, N_grid), dtype=np.float64)
    diag = 1.0 / (mu * dR ** 2) + V
    offdiag = -0.5 / (mu * dR ** 2)
    for i in range(N_grid):
        H[i, i] = diag[i]
        if i > 0:
            H[i, i - 1] = offdiag
        if i < N_grid - 1:
            H[i, i + 1] = offdiag

    evals, evecs = np.linalg.eigh(H)
    psi0 = evecs[:, 0] / np.sqrt(np.trapezoid(evecs[:, 0] ** 2, R_space))
    psi1 = evecs[:, 1] / np.sqrt(np.trapezoid(evecs[:, 1] ** 2, R_space))
    return R_space, float(evals[0]), float(evals[1]), psi0, psi1


def main():
    p = argparse.ArgumentParser(description="CO2 vibration animation (taichi GGUI)")
    p.add_argument("--orbitals", type=Path, required=True, help="output of co2_precompute_orbitals.py")
    p.add_argument("--pes", type=Path, required=True, help="output of co2_dft_pes.py")
    p.add_argument("--mu-amu", type=float, default=None,
                    help="override the reduced mass stored in the orbitals file")
    p.add_argument("--cpu", action="store_true", help="force the CPU backend instead of GPU")
    p.add_argument("--dt", type=float, default=None,
                    help="atomic-time units advanced per frame (default: auto, so one vibrational "
                         "period lasts --period-sec seconds at ~60 fps)")
    p.add_argument("--period-sec", type=float, default=4.0,
                    help="target wall-clock duration of one vibrational period (used when --dt is auto)")
    p.add_argument("--gain", type=float, default=None,
                    help="visual amplitude exaggeration of the <q>(t) swing about its mean "
                         "(default: auto, so the swing fills --target-frac of the scanned q range)")
    p.add_argument("--target-frac", type=float, default=0.7)
    p.add_argument("--headless-phase", type=float, default=0.0, help="phase (rad) for --headless-test")
    p.add_argument("--headless-scale", type=int, default=3, help="integer upscale of the headless PNG")
    p.add_argument("--headless-test", type=Path, default=None,
                    help="render a single frame to this PNG and exit (no window needed)")
    args = p.parse_args()

    ti.init(arch=ti.cpu if args.cpu else ti.gpu)

    print("Loading precomputed CO2 orbitals and PES...")
    orb = np.load(args.orbitals, allow_pickle=True)
    rho_stack = orb["rho"]                       # (nR, nx, nz)
    R_bohr_stack = orb["R_bohr"]                  # (nR,)  == q, bohr
    box_bohr = float(orb["box_bohr"])
    symbols = [str(s) for s in orb["symbols"]]
    natm = len(symbols)
    atom_x_bohr = orb["atom_x_bohr"]              # (nR, natm)
    atom_z_bohr = orb["atom_z_bohr"]              # (nR, natm)
    mode_name = str(orb["mode"])
    mu_amu = args.mu_amu if args.mu_amu is not None else float(orb["reduced_mass_amu"])

    nR, nx, nz = rho_stack.shape

    rho_field = ti.field(dtype=ti.f32, shape=(nR, nx, nz))
    rho_field.from_numpy(rho_stack)

    atom_x_field = ti.field(dtype=ti.f32, shape=(nR, natm))
    atom_z_field = ti.field(dtype=ti.f32, shape=(nR, natm))
    atom_x_field.from_numpy(atom_x_bohr.astype(np.float32))
    atom_z_field.from_numpy(atom_z_bohr.astype(np.float32))

    atom_core_color = ti.Vector.field(3, dtype=ti.f32, shape=(natm,))
    atom_glow_color = ti.Vector.field(3, dtype=ti.f32, shape=(natm,))
    for a, sym in enumerate(symbols):
        core, glow = ELEMENT_COLORS.get(sym, DEFAULT_COLOR)
        atom_core_color[a] = core
        atom_glow_color[a] = glow

    render_texture = ti.Vector.field(4, dtype=ti.f32, shape=(nx, nz))
    prob_field = ti.field(dtype=ti.f32, shape=(1000,))

    pes_df = pd.read_csv(args.pes)
    R_pes = pes_df["R_bohr"].values
    E_pes = pes_df["E_hartree"].values
    pot_spline = CubicSpline(R_pes, E_pes)
    E_min = float(np.min(E_pes))

    print(f"Mode: {mode_name}   mu_eff = {mu_amu:.4f} amu")
    R_space, E0, E1, psi0, psi1 = solve_vibrational_states(R_bohr_stack, pot_spline, E_min, mu_amu)
    omega = E1 - E0
    print(f"Vibrational \u0394E = {omega * HARTREE_TO_EV * 8065.54:.2f} cm\u207b\u00b9")

    R_min_val = float(R_bohr_stack[0])
    R_max_val = float(R_bohr_stack[-1])

    # The physical <q>(t) swing of a 0/1 superposition is ~0.05-0.1 bohr, i.e. under one
    # pixel -- invisible. So the displayed q is exaggerated about its cycle-mean by `gain`
    # (shown in the GUI so it is never mistaken for the true amplitude).
    def q_expect(ph):
        psi_t = (psi0 + psi1 * np.cos(ph)) / np.sqrt(2.0)
        return float(np.trapezoid(psi_t ** 2 * R_space, R_space))

    q_curve = np.array([q_expect(ph) for ph in np.linspace(0.0, 2.0 * np.pi, 721)])
    q_mean = float(q_curve.mean())
    q_half = float(0.5 * (q_curve.max() - q_curve.min()))
    room = min(R_max_val - q_mean, q_mean - R_min_val)
    gain = args.gain if args.gain is not None else max(1.0, args.target_frac * room / max(q_half, 1e-9))
    dt = args.dt if args.dt is not None else 2.0 * np.pi / (omega * 60.0 * args.period_sec)
    print(f"True <q> swing: +/-{q_half:.3f} bohr   display gain: x{gain:.1f}   dt = {dt:.2f} a.u./frame")

    def displayed_q(r_exp, g):
        return float(np.clip(q_mean + g * (r_exp - q_mean), R_min_val, R_max_val))

    @ti.kernel
    def compute_density_frame(R_current: ti.f32, R_min: ti.f32, R_max: ti.f32, box_bohr: ti.f32):
        frac_idx = (R_current - R_min) / (R_max - R_min) * ti.cast(nR - 1, ti.f32)
        idx_low = ti.cast(ti.floor(frac_idx), ti.i32)
        idx_high = ti.cast(ti.ceil(frac_idx), ti.i32)
        weight = frac_idx - ti.cast(idx_low, ti.f32)
        idx_low = ti.max(0, ti.min(idx_low, nR - 1))
        idx_high = ti.max(0, ti.min(idx_high, nR - 1))

        for i, j in render_texture:
            plot_height = nz // 5

            if j > nz - plot_height:
                # -- nuclear wavepacket plot zone --
                grid_idx = ti.cast(i / (nx - 1) * 999, ti.i32)
                psi_val = prob_field[grid_idx]
                pixel_y = (nz - 1) - j
                scaled_y = ti.cast(psi_val * (ti.cast(plot_height, ti.f32) * 0.15), ti.i32)
                if pixel_y == scaled_y:
                    render_texture[i, j] = ti.Vector([1.0, 0.5, 0.1, 1.0])
                elif j == nz - plot_height + 1:
                    render_texture[i, j] = ti.Vector([0.2, 0.2, 0.25, 1.0])
                else:
                    render_texture[i, j] = ti.Vector([0.01, 0.01, 0.03, 1.0])
            else:
                # -- electron density zone --
                j_mapped = ti.cast(j / (nz - plot_height) * (nz - 1), ti.i32)
                j_clamped = ti.max(0, ti.min(j_mapped, nz - 1))

                rho_l = rho_field[idx_low, i, j_clamped]
                rho_h = rho_field[idx_high, i, j_clamped]
                rho = (1.0 - weight) * rho_l + weight * rho_h
                val = ti.pow(rho, 1.0 / 3.0)
                pixel_color = ti.Vector([0.0, val * 0.4, val * 0.9, 1.0])

                # -- N-nucleus tracker (generalizes the single j_H formula) --
                for a in range(natm):
                    ax = (1.0 - weight) * atom_x_field[idx_low, a] + weight * atom_x_field[idx_high, a]
                    az = (1.0 - weight) * atom_z_field[idx_low, a] + weight * atom_z_field[idx_high, a]

                    i_a = ti.cast((ax + box_bohr) / (2.0 * box_bohr) * ti.cast(nx - 1, ti.f32), ti.i32)
                    j_a = ti.cast((az + box_bohr) / (2.0 * box_bohr) * ti.cast(nz - 1, ti.f32), ti.i32)

                    dist = ti.sqrt(ti.cast((i - i_a) ** 2 + (j_mapped - j_a) ** 2, ti.f32))
                    core = atom_core_color[a]
                    glow = atom_glow_color[a]

                    if dist < 3.0:
                        pixel_color = ti.Vector([core[0], core[1], core[2], 1.0])
                    elif dist < 8.0:
                        glow_factor = (8.0 - dist) / 5.0
                        glow4 = ti.Vector([glow[0], glow[1], glow[2], 1.0])
                        pixel_color = glow_factor * glow4 + (1.0 - glow_factor) * pixel_color

                render_texture[i, j] = pixel_color

    if args.headless_test is not None:
        # No window / no GPU display needed: render one frame at --headless-phase to a PNG.
        phase = args.headless_phase
        psi_t = (psi0 + psi1 * np.cos(phase)) / np.sqrt(2.0)
        prob_density = psi_t ** 2
        R_expect = float(np.trapezoid(prob_density * R_space, R_space))
        q_show = displayed_q(R_expect, gain)
        prob_field.from_numpy(prob_density.astype(np.float32))
        compute_density_frame(q_show, R_min_val, R_max_val, float(box_bohr))
        img = render_texture.to_numpy()  # (nx, nz, 4)
        rgb = np.transpose(img[:, :, :3], (1, 0, 2))[::-1]          # y up -> image rows
        rgb = np.repeat(np.repeat(rgb, args.headless_scale, axis=0), args.headless_scale, axis=1)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(args.headless_test, np.clip(rgb, 0, 1))
        print(f"\u2713  Headless test frame \u2192 {args.headless_test.resolve()}")
        print(f"   phase={phase:.2f}  true <q>={R_expect:+.4f} bohr   shown q={q_show:+.4f} bohr")
        return

    window = ti.ui.Window(f"CO2 Vibration ({mode_name}) - Real-time DFT", (800, 800))
    canvas = window.get_canvas()
    gui = window.get_gui()

    phase = 0.0
    paused = False
    speed = 1.0
    gain_ui = float(gain)

    while window.running:
        if not paused:
            phase += omega * dt * speed

        psi_t = (psi0 + psi1 * np.cos(phase)) / np.sqrt(2.0)
        prob_density = psi_t ** 2
        R_expect = float(np.trapezoid(prob_density * R_space, R_space))
        q_show = displayed_q(R_expect, gain_ui)
        prob_field.from_numpy(prob_density.astype(np.float32))

        compute_density_frame(q_show, R_min_val, R_max_val, float(box_bohr))
        canvas.set_image(render_texture)

        with gui.sub_window("Simulation Controls", 0.02, 0.62, 0.40, 0.34):
            if gui.button("Pause/Resume"):
                paused = not paused
            speed = gui.slider_float("Speed", speed, 0.05, 3.0)
            gain_ui = gui.slider_float("Amplitude gain", gain_ui, 1.0, max(20.0, 2.0 * gain))
            gui.text(f"Mode: {mode_name}")
            gui.text(f"True <q>:  {R_expect:+.3f} bohr")
            gui.text(f"Shown q:   {q_show:+.3f} bohr (x{gain_ui:.1f})")
            gui.text(f"Phase: {phase:.2f} rad")

        window.show()


if __name__ == "__main__":
    main()
