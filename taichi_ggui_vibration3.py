"""
taichi_ggui_vibration.py

Real-time 2D electron density and nuclear wavepacket animation using Taichi GGUI.
Loads precomputed PySCF density stack and potential energy surface.
"""

import numpy as np
import pandas as pd
import taichi as ti
from scipy.interpolate import CubicSpline
from scipy.constants import physical_constants

# ── UNIT CONVERSIONS ─────────────────────────────────────────────────────────
HARTREE_TO_EV = 27.211386
AMU_TO_ME = 1822.888486192
ANG_TO_BOHR = 1.8897261246257702

# ── INITIALISE TAICHI ────────────────────────────────────────────────────────
ti.init(arch=ti.gpu)

# ── LOAD PRECOMPUTED DATA ────────────────────────────────────────────────────
print("Loading precomputed orbitals and PES...")
orb_data = np.load("orbitals_hcl.npz")
rho_stack = orb_data["rho"]          # (nR, nx, nz)
R_bohr_stack = orb_data["R_bohr"]    # (nR,)
box_bohr = float(orb_data["box_bohr"])
atom1 = str(orb_data["atom1"])
atom2 = str(orb_data["atom2"])

nR, nx, nz = rho_stack.shape

# Setup Taichi fields for the GPU stack
rho_field = ti.field(dtype=ti.f32, shape=(nR, nx, nz))
rho_field.from_numpy(rho_stack)

render_texture = ti.Vector.field(4, dtype=ti.f32, shape=(nx, nz))

# Load PES for Numerov solver
pes_df = pd.read_csv("pes_hcl.csv")
R_pes = pes_df["R_bohr"].values
E_pes = pes_df["E_hartree"].values
pot_spline = CubicSpline(R_pes, E_pes)
E_min = np.min(E_pes)

# ── NUMEROV VIBRATIONAL SOLVER (DIATOMIC) ────────────────────────────────────
def solve_vibrational_states(mu_amu=0.97959):
    # Setup internal grid for solving the wavepacket
    N_grid = 1000
    
    R_array = np.asarray(R_bohr_stack).flatten()
    # FIX 1: Index individual array elements before converting to scalar floats
    R_min = float(R_array[0])
    R_max = float(R_array[-1])
    
    R_space = np.linspace(R_min, R_max, N_grid)
    
    # FIX 2: Compute real grid step size instead of subtracting array from itself
    dR = float(R_space[1] - R_space[0])
    mu = float(mu_amu * AMU_TO_ME)
    
    # Ensure V is strictly a 1D array of shape (N_grid,)
    V = pot_spline(R_space).flatten() - float(E_min)
    
    # Simple finite difference matrix formulation
    H = np.zeros((N_grid, N_grid), dtype=np.float64)
    diag = 1.0 / (mu * dR**2) + V
    offdiag = -0.5 / (mu * dR**2)
    
    for i in range(N_grid):
        H[i, i] = diag[i]
        if i > 0: 
            H[i, i-1] = offdiag
        if i < N_grid - 1: 
            H[i, i+1] = offdiag
        
    evals, evecs = np.linalg.eigh(H)
    
    # Normalise wavefunctions using NumPy's modern trapezoid integration
    psi0 = evecs[:, 0] / np.sqrt(np.trapezoid(evecs[:, 0]**2, R_space))
    psi1 = evecs[:, 1] / np.sqrt(np.trapezoid(evecs[:, 1]**2, R_space))
    
    # FIX 3: Return scalar values for E0 and E1 instead of duplicating full arrays
    return R_space, float(evals[0]), float(evals[1]), psi0, psi1


R_space, E0, E1, psi0, psi1 = solve_vibrational_states()
omega = E1 - E0
print(f"Vibrational ΔE = {(omega)*HARTREE_TO_EV*8065.54:.2f} cm⁻¹")


render_texture = ti.Vector.field(4, dtype=ti.f32, shape=(nx, nz))

# ADD THIS: GPU memory allocation for the 1D wavepacket plot
prob_field = ti.field(dtype=ti.f32, shape=(1000,))


# ── TAICHI KERNELS FOR INTERPOLATION AND RENDERING ───────────────────────────
@ti.kernel
def compute_density_frame(R_current: ti.f32, R_min: ti.f32, R_max: ti.f32, box_bohr: ti.f32):
    frac_idx = (R_current - R_min) / (R_max - R_min) * ti.cast(nR - 1, ti.f32)
    idx_low = ti.cast(ti.floor(frac_idx), ti.i32)
    idx_high = ti.cast(ti.ceil(frac_idx), ti.i32)
    weight = frac_idx - ti.cast(idx_low, ti.f32)
    
    idx_low = ti.max(0, ti.min(idx_low, nR - 1))
    idx_high = ti.max(0, ti.min(idx_high, nR - 1))
    
    for i, j in render_texture:
        plot_height = nz // 5  # Reserve bottom 20% of rows for the plot
        
        if j > nz - plot_height:
            # ── PLOT ZONE ────────────────────────────────────────────────────
            grid_idx = ti.cast(i / (nx - 1) * 999, ti.i32)
            psi_val = prob_field[grid_idx]
            
            pixel_y = (nz - 1) - j 
            scaled_y = ti.cast(psi_val * (ti.cast(plot_height, ti.f32) * 0.15), ti.i32)
            
            if pixel_y == scaled_y:
                render_texture[i, j] = ti.Vector([1.0, 0.5, 0.1, 1.0])  # Orange Wavepacket
            elif j == nz - plot_height + 1:
                render_texture[i, j] = ti.Vector([0.2, 0.2, 0.25, 1.0]) # Separator
            else:
                render_texture[i, j] = ti.Vector([0.01, 0.01, 0.03, 1.0]) # Dark Background
                
        else:
            # ── ELECTRON DENSITY ZONE ────────────────────────────────────────
            j_mapped = ti.cast(j / (nz - plot_height) * (nz - 1), ti.i32)
            j_clamped = ti.max(0, ti.min(j_mapped, nz - 1))
            
            rho_l = rho_field[idx_low, i, j_clamped]
            rho_h = rho_field[idx_high, i, j_clamped]
            rho = (1.0 - weight) * rho_l + weight * rho_h
            
            val = ti.pow(rho, 1.0 / 3.0)
            
            # Base color: Glowing cyan electron cloud
            pixel_color = ti.Vector([0.0, val * 0.4, val * 0.9, 1.0])
            
            # ── HYDROGEN NUCLEUS TRACKER ─────────────────────────────────────
            # NOTE: If your PySCF generation centered the *midpoint* of the bond, use 0.5.
            # If it fixed the Chlorine atom dead-center at (0,0), change 0.5 to 1.0!
            j_H = (nz // 2) - ti.cast(0.5 * R_current * (ti.cast(nz, ti.f32) / box_bohr), ti.i32)
            i_H = nx // 2
            
            # Calculate distance from current pixel to the Hydrogen nucleus
            dist_to_H = ti.sqrt(ti.cast((i - i_H)**2 + (j_mapped - j_H)**2, ti.f32))
            
            if dist_to_H < 3.0:
                # Solid red core for the nucleus
                pixel_color = ti.Vector([1.0, 0.1, 0.1, 1.0])
            elif dist_to_H < 8.0:
                # Anti-aliased soft red glow blending into the cyan cloud
                glow_factor = (8.0 - dist_to_H) / 5.0
                red_glow = ti.Vector([0.8, 0.0, 0.0, 1.0])
                pixel_color = glow_factor * red_glow + (1.0 - glow_factor) * pixel_color
                
            render_texture[i, j] = pixel_color

"""
def compute_density_frame(R_current: ti.f32, R_min: ti.f32, R_max: ti.f32):
    # Map R_current to fractional index in the precomputed stack
    frac_idx = (R_current - R_min) / (R_max - R_min) * ti.cast(nR - 1, ti.f32)
    idx_low = ti.cast(ti.floor(frac_idx), ti.i32)
    idx_high = ti.cast(ti.ceil(frac_idx), ti.i32)
    weight = frac_idx - ti.cast(idx_low, ti.f32)
    
    idx_low = ti.max(0, ti.min(idx_low, nR - 1))
    idx_high = ti.max(0, ti.min(idx_high, nR - 1))
    
    for i, j in render_texture:
        # Linear interpolation between the two closest DFT snapshots
        rho_l = rho_field[idx_low, i, j]
        rho_h = rho_field[idx_high, i, j]
        rho = (1.0 - weight) * rho_l + weight * rho_h
        
        # FIX ALIASING FLICKER: Use power scaling (cubic root) to smooth the core
        val = ti.pow(rho, 1.0 / 3.0)
        
        # Colour mapping: Glowing cyan/blue electron cloud
        render_texture[i, j] = ti.Vector([0.0, val * 0.4, val * 0.9, 1.0])
"""


# ── MAIN ANIMATION LOOP ──────────────────────────────────────────────────────
window = ti.ui.Window("Quantum Diatomic Vibration - Real-time DFT", (800, 800))
canvas = window.get_canvas()
gui = window.get_gui()

time = 0.0
dt = 0.15 # Time step for wavepacket evolution
paused = False

# FIX 4: Safely extract bounds without relying on .item()
R_min_val = float(R_bohr_stack[0])
R_max_val = float(R_bohr_stack[-1])

"""

while window.running:
    if not paused:
        time += dt
        
    # 1. Quantum Wavepacket Evolution: coherent superposition state
    # |Ψ(t)> = 1/sqrt(2) * (|0> + |1> * e^(-i*omega*t))
    phase = omega * time
    psi_t = (psi0 + psi1 * np.cos(phase)) / np.sqrt(2.0)
    prob_density = psi_t**2
    
    # Expectation value of the bond length <R>(t)
    # FIX 5: Updated from deprecated np.trapz to modern np.trapezoid
    R_expect = np.trapezoid(prob_density * R_space, R_space)
    
    # 2. Trigger GPU reconstruction of electron density
    compute_density_frame(float(R_expect), R_min_val, R_max_val)
    
    # 3. Render Electron Cloud Texture
    canvas.set_image(render_texture)
    
    # 4. Drawing User Interface & Analytics Plots
    with gui.sub_window("Simulation Controls", 0.05, 0.05, 0.4, 0.2):
        if gui.button("Pause/Resume"):
            paused = not paused
        gui.text(f"Molecule: {atom1}-{atom2}")
        gui.text(f"Expectation <R>: {R_expect / ANG_TO_BOHR:.4f} Å")
        gui.text(f"Oscillation Phase: {phase:.2f} rad")
        
    # Render the dynamic Nuclear Wavepacket Profile at the bottom
    with gui.sub_window("Nuclear Wavepacket |Ψ(R,t)|²", 0.05, 0.75, 0.9, 0.2):
        # Convert data safely to display on Taichi graph
        gui.plot_profile(title="Probability Density", val=prob_density.astype(np.float32))

    window.show()
    
    """
    
while window.running:
    if not paused:
        time += dt
        
    # 1. Quantum Wavepacket Evolution
    phase = omega * time
    psi_t = (psi0 + psi1 * np.cos(phase)) / np.sqrt(2.0)
    prob_density = psi_t**2
    
    # Expectation value of the bond length <R>(t)
    R_expect = np.trapezoid(prob_density * R_space, R_space)
    
    # Send calculated wavepacket distribution data straight to the GPU
    prob_field.from_numpy(prob_density.astype(np.float32))
    
    # 2. Trigger GPU reconstruction of electron density & plot
    compute_density_frame(float(R_expect), R_min_val, R_max_val, float(box_bohr))
    #compute_density_frame(float(R_expect), R_min_val, R_max_val)
    
    
    # 3. Render integrated texture
    canvas.set_image(render_texture)
    
    # 4. Drawing User Interface Controls
    with gui.sub_window("Simulation Controls", 0.05, 0.05, 0.4, 0.2):
        if gui.button("Pause/Resume"):
            paused = not paused
        gui.text(f"Molecule: {atom1}-{atom2}")
        gui.text(f"Expectation <R>: {R_expect / ANG_TO_BOHR:.4f} Å")
        gui.text(f"Oscillation Phase: {phase:.5f} rad")

    window.show()