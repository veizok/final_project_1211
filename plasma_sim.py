import numpy as np
import matplotlib.pyplot as pl
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.animation as animation
import datetime
import time

# Simulation Parameters
L = 0.05                # Simulation box size (m)
Nx = 64                 # Number of grid cells per axis
dx = L / Nx             # Grid spacing
dt = 2e-11              # Time step (s)
T_plasma = 1000         # Plasma temperature (K)
R_plasma = 0.0125        # Initial radius of plasma (m)
N_particles = int(1e6)       # Number of each particle type simulated

# Physical Parameters
q_e_real = -1.602e-19   # Electron charge (C)
q_p_real = 1.602e-19    # Proton charge (C)
eps0 = 8.854e-12        # Permittivity of free space (F/m)
kB = 1.380649e-23       # Boltzmann constant (J/K)
m_e_real = 9.109e-31    # Electron mass (kg)
m_p_real = 1.672e-27    # Proton mass (kg)

# Macroparticle Parameters
lambda_D = 2 * dx       # Debye length - must have multiple lengths per cell to accurately portray the physics
scale_n = (eps0 * kB * T_plasma)/(lambda_D**2 * q_e_real**2) # Calculating true number of particles to simulate for accurate physics, based on calculated Debye length

plasma_area = np.pi * R_plasma**2 # Initial area of plasma (m^2)
actual_charge = scale_n * plasma_area * q_p_real # Total actual charge to simulate based on the scale factor from debye length
weight = actual_charge / (N_particles * q_p_real) # Weight calculation for both species of particle

# Weighted quantities for particles
q_e = q_e_real * weight
q_p = q_p_real * weight
m_e = m_e_real * weight
m_p = m_p_real * weight

# Animation Parameters
total_frames = 500

# Building matrix for Poisson solver and enforcing phi=0 at the edges of the simulation box
def build_matrix(nx):
    '''
    Builds the matrix A, which represents the Laplacian operator in the Poisson equation
       Inputs:
        nx: the number of grid points per axis
       Returns:
        A: the matrix representing the Laplacian operator
    '''
    # Length of diag arrays (number of unknowns in system of equations)
    N = nx * nx
    # Forming diagonals based on coefficients from discretized Poisson equation (shown in notebook)
    main_diag = -4 * np.ones(N)
    off_diag_x = np.ones(N - 1)
    off_diag_y = np.ones(N - nx)
    # Preventing flattened array from wrapping matrix rows into each other.
    for i in range(1, N):
        if i % nx == 0:
            off_diag_x[i-1] = 0
    # Building the sparse matrix
    diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
    offsets = [0, -1, 1, -nx, nx]
    A = diags(diagonals, offsets, shape=(N, N), format='csc')
    return A

matrix = build_matrix(Nx)

# Poisson solver
def solve_poisson(rho):
    '''
    Solves Poisson equation with phi = 0 at simulation walls in the form of a system of linear equations A*phi = b
       Input:
        rho: Nx x Nx array containing the charge density at each point on the grid
       Returns:
        phi: Nx x Nx array containing the electric potential at each point on the grid
    '''
    # Constructing b vector from right hand side of discretized Poisson equation
    b = -rho.flatten() / eps0 * dx**2
    # Solving the system for the potential and reshaping the flattened (1D) vector into an Nx x Nx array
    phi_flat = spsolve(matrix, b)
    phi = phi_flat.reshape((Nx, Nx))
    return phi

# Initializing particles
def initialize_particles():
    '''Generates initial particle positions and velocities and distributes them uniformly in a circle of radius R_plasma'''
    # Calculating standard deviation for velocity initialization
    sig_e = np.sqrt(kB * T_plasma / m_e_real)
    sig_p = np.sqrt(kB * T_plasma / m_p_real)
    # Generating empty pos and vel arrays
    pos_e = np.zeros((N_particles, 2))
    vel_e = np.zeros((N_particles, 2))
    # Generating distance from center of plasma for each particle
    r = R_plasma * np.sqrt(np.random.rand(N_particles))
    # Generating angle at which each particle is placed relative to center of plasma
    theta = 2 * np.pi * np.random.rand(N_particles)
    # Converting the particle positions to cartesian coordinates and placing them relative to the center of the simulation box
    pos_e[:, 0] = L/2 + r * np.cos(theta)
    pos_e[:, 1] = L/2 + r * np.sin(theta)
    # Generating initial velocity components independently for each particle from Gaussian distribution with mean=0 (no bulk motion), sigma from previous calculation
    vel_e[:, 0] = np.random.normal(0, sig_e, N_particles)
    vel_e[:, 1] = np.random.normal(0, sig_e, N_particles)

    # Repeating steps from above for other particle species (protons)
    pos_p = np.zeros((N_particles, 2))
    vel_p = np.zeros((N_particles, 2))
    r = R_plasma * np.sqrt(np.random.rand(N_particles))
    theta = 2 * np.pi * np.random.rand(N_particles)
    pos_p[:, 0] = L/2 + r * np.cos(theta)
    pos_p[:, 1] = L/2 + r * np.sin(theta)
    vel_p[:, 0] = np.random.normal(0, sig_p, N_particles)
    vel_p[:, 1] = np.random.normal(0, sig_p, N_particles)

    return pos_e, vel_e, pos_p, vel_p

# Grid assignment and charge density
def calculate_density(pos, q_eff):
    '''
    Assigns the charge of each particle to the nearest node; calculates rho at each node
       Inputs:
        pos: particle positions array
        q_eff: the weighted charge of the each macroparticle
       Returns: array of charge density values for each node
    '''
    # Creating Nx x Nx array to store the charge value at each node
    density = np.zeros((Nx, Nx))
    # Converting particle positions to grid coordinates, assigns particle to the lower-left node of the cell containing the particle
    x_norm = pos[:, 0] / dx - 0.5
    y_norm = pos[:, 1] / dx - 0.5
    # Obtaining integer indices for the node to the lower-left of the particle
    i = np.floor(x_norm).astype(int)
    j = np.floor(y_norm).astype(int)
    # Creating weighting terms for each particle by calculating distance from particle to its node
    wx = 1 - (x_norm - i)
    wy = 1 - (y_norm - j)
    # Checking that the indices for each particle are within the grid, particles that are outside the simulation area or directly on the edge are excluded from the density calculation
    mask = (i >= 0) & (i < Nx-1) & (j >= 0) & (j < Nx-1)
    i = i[mask]; j = j[mask]; wx = wx[mask]; wy = wy[mask]
    # Distributing each particle's charge to the four surrounding nodes
    np.add.at(density, (i, j), q_eff * wx * wy)
    np.add.at(density, (i+1, j), q_eff * (1-wx) * wy)
    np.add.at(density, (i, j+1), q_eff * wx * (1-wy))
    np.add.at(density, (i+1, j+1), q_eff * (1-wx) * (1 - wy))

    return density / (dx**2)

def calculate_electric_field(phi):
    # Building arrays of same shape as phi to store E field values
    Ex = np.zeros_like(phi)
    Ey = np.zeros_like(phi)
    # Calculating E field for interior points - the slices exclude the points around the edge of the simulation to enforce the boundary condition of phi=0
    Ex[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2*dx)
    Ey[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2*dx)
    return Ex, Ey

def interpolate_field(pos, Ex, Ey):
    '''Using bilinear interpolation to distribute E field values from the grid nodes back to the particles'''
    # Calculating index of grid node to the lower-left of the particle, as before
    i = np.floor(pos[:, 0] / dx - 0.5).astype(int)
    j = np.floor(pos[:, 1] / dx - 0.5).astype(int)
    # Weighting using particle distance from node
    dx_local = (pos[:, 0] / dx - 0.5) - i
    dy_local = (pos[:, 1] / dx - 0.5) - j
    # Making sure calculated indices stay within the boundaries
    i = np.clip(i, 0, Nx-2)
    j = np.clip(j, 0, Nx-2)
    # Performing bilinear interpolation: each particle's E field value is a weighted average of the values at the four surrounding grid nodes
    Ex_p = (Ex[i, j] * (1-dx_local) * (1-dy_local) + 
            Ex[i+1, j] * dx_local * (1-dy_local) + 
            Ex[i, j+1] * (1-dx_local) * dy_local +
            Ex[i+1, j+1] * dx_local * dy_local)
    
    Ey_p = (Ey[i, j] * (1-dx_local) * (1-dy_local) +
            Ey[i+1, j] * dx_local * (1-dy_local) + 
            Ey[i, j+1] * (1-dx_local) * dy_local + 
            Ey[i+1, j+1] * dx_local * dy_local)
    
    return np.stack((Ex_p, Ey_p), axis = 1)

# Particle push function
def verlet_step(pos, vel, acc, q_eff, m_eff, Ex, Ey):
    '''Calculates the positions and velocities of the particles at each timestep using the calculated E field'''
    # Calculating the velocity at time t + 0.5dt
    vel_half = vel + 0.5 * acc * dt
    # Calculating position at time t + dt
    pos_new = pos + vel_half * dt
    # Calculating the electric field acting on the particle at its new position
    E_at_pos = interpolate_field(pos_new, Ex, Ey)
    # Calculating the new acceleration at time t + dt; uses effective (scaled) charge and mass of each particle
    acc_new = (q_eff / m_eff) * E_at_pos
    # Calculates the new velocity after the second half of the timestep (t + dt)
    vel_new = vel_half + 0.5 * acc_new * dt
    return pos_new, vel_new, acc_new

def remove_particles(pos, vel, acc):
    # Mask to check that particles are within simulation boundaries: must have 0 < x < L and 0 < y < L.
    mask = (pos[:, 0] > 0) & (pos[:, 0] < L) & (pos[:, 1] > 0) & (pos[:, 1] < L)
    return pos[mask], vel[mask], acc[mask]

# Simulation Loop
if __name__ == '__main__':
    # Initializing particles
    pos_e, vel_e, pos_p, vel_p = initialize_particles()
    # Initializing Figure
    fig, (ax1, ax2, ax3) = pl.subplots(1, 3, figsize=(18, 5))

    # Calculating initial state
    rho_e = calculate_density(pos_e, q_e)
    rho_p = calculate_density(pos_p, q_p)

    # Calculating particle densities (m^-2)
    n_e = rho_e / q_e_real
    n_p = rho_p / q_p_real

    # Calculating electric potential and field
    phi = solve_poisson(rho_e + rho_p)
    Ex, Ey = calculate_electric_field(phi)

    # Calculating electric field magnitude for figure
    E_mag = np.sqrt(Ex**2 + Ey**2)

    acc_e = (q_e / m_e) * interpolate_field(pos_e, Ex, Ey)
    acc_p = (q_p / m_p) * interpolate_field(pos_p, Ex, Ey)

    # Making plots for figure
    # Electron density
    pl1 = ax1.imshow(n_e.T, origin='lower', extent=[0, L*100, 0, L*100], cmap='plasma', interpolation='bilinear')
    ax1.set_title('Electron Density ($m^{-2}$)')
    ax1.set_xlabel('x (cm)'); ax1.set_ylabel('y (cm)')
    pl.colorbar(pl1, ax=ax1)

    # Proton density
    pl2 = ax2.imshow(n_p.T, origin='lower', extent=[0, L*100, 0, L*100], cmap='plasma', interpolation='bilinear')
    ax2.set_title('Proton Density ($m^{-2}$)')
    ax2.set_xlabel('x (cm)'); ax2.set_ylabel('y (cm)')
    pl.colorbar(pl2, ax=ax2)

    # Electric field strength
    pl3 = ax3.imshow(E_mag.T, origin='lower', extent=[0, L*100, 0, L*100], cmap='plasma', interpolation='bilinear')
    ax3.set_title('Electric Field Strength (V/m)')
    ax3.set_xlabel('x (cm)'); ax3.set_ylabel('y (cm)')
    pl.colorbar(pl3, ax=ax3)

    # Update function to repeat steps
    def update(frame):
        t0 = time.time()
        global pos_e, vel_e, acc_e, pos_p, vel_p, acc_p

        rho_e = calculate_density(pos_e, q_e)
        rho_p = calculate_density(pos_p, q_p)
        rho_total = rho_e + rho_p

        phi = solve_poisson(rho_total)
        Ex, Ey = calculate_electric_field(phi)
        E_mag = np.sqrt(Ex**2 + Ey**2)

        pos_e, vel_e, acc_e = verlet_step(pos_e, vel_e, acc_e, q_e, m_e, Ex, Ey)
        pos_p, vel_p, acc_p = verlet_step(pos_p, vel_p, acc_p, q_p, m_p, Ex, Ey)

        pos_e, vel_e, acc_e = remove_particles(pos_e, vel_e, acc_e)
        pos_p, vel_p, acc_p = remove_particles(pos_p, vel_p, acc_p)

        # Density calculations for figure
        n_e = rho_e / q_e_real
        n_p = rho_p / q_p_real

        # Updating plots
        pl1.set_data(n_e.T)
        pl1.set_clim(vmin=0, vmax=np.max(n_e))

        pl2.set_data(n_p.T)
        pl2.set_clim(vmin=0, vmax=np.max(n_p))

        pl3.set_data(E_mag.T)
        pl3.set_clim(vmin=0, vmax=np.max(E_mag))

        dt_calc = (time.time() - t0) * 1000
        print(f'Frame {frame}/{total_frames} | Time: {dt_calc:.1f}ms | Electrons: {len(pos_e)} | Protons: {len(pos_p)}')

        return pl1, pl2, pl3
    
    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=False)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ani.save(f'plasma_sim_{timestamp}.gif', writer='Pillow', fps=15)