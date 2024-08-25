# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from Helper import Lattice_Boltzmann_Method as lbm
from Helper import Boundary_Conditions as bound

# Define paths for storing results
PATH = "results"  # Main result directory
path = os.path.join(PATH, 'M4_Couette_Flow')  # Path for storing Couette flow results
os.makedirs(path, exist_ok=True)  # Create the directory if it doesn't exist

# Function to plot Couette flow velocity profiles
def plot_couette_flow(u_ij, wall_velocity, step, nx, ny):
    # Create array for y coordinates
    y = np.arange(ny)
    # Calculate analytical velocity profile for Couette flow
    #analytical = wall_velocity[1] * np.arange(ny) / (ny - 1)
    analytical = (ny - 1 - y) / (ny - 1) * wall_velocity[1]

    # Create a new figure
    plt.figure(figsize=(10, 6))  # Set the figure size

    # Create a color gradient for the simulated velocity profile
    sim_cmap = plt.get_cmap('viridis')
    sim_colors = sim_cmap(np.linspace(0, 1, nx))

    # Set x-axis limits and add reference lines
    plt.xlim([-0.01, wall_velocity[1]])
    plt.axhline(0.0, color='k')# Add a horizontal line at y=0 (Bottom Wall)
    plt.axhline(ny - 1, color='r')# Add a horizontal line at y=ny-1 (TOP wall)

    # Plot the simulated velocity profile with color gradient and shading
    plt.plot(u_ij[0, nx // 2, :], y, linestyle='dashed', color='black', linewidth=2)
    plt.scatter(u_ij[0, nx // 2, :], y, linestyle='dashed',c=sim_colors, s=30, edgecolors='none')

    # Plot the analytical velocity profile with a dashed line
    plt.plot(analytical, y, linestyle='dotted', color='blue', linewidth=2)

    # Add labels, title, grid, colorbar, and legend
    plt.ylabel('y')
    plt.xlabel('velocity')
    plt.title(f'Couette Flow Simulation (Step {step})')
    plt.grid(True, linestyle='--', alpha=0.7)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=wall_velocity[1]))
    sm.set_array([])
    cbar_ax = plt.gca().inset_axes([1.01, 0.1, 0.02, 0.8])  # Adjust position and size as needed
    cbar = plt.colorbar(sm, cax=cbar_ax, label='Velocity')
    plt.legend(
        [ 'Moving Wall','Rigid Wall', 'Simulated Velocity', 'Analytical Velocity'],
        loc='upper right'
    )

    # Save the plot with better spacing and display it
    save_path = os.path.join(path, f'couette_flow_{step}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)

# Function to simulate Couette flow with Lattice Boltzmann method
def couette_flow(nx, ny, omega, num_steps):
    print(f"Simulation started: {num_steps} steps.")
    # Initialize density and velocity arrays
    rho_ij = np.ones((nx, ny))  # Initialize the density array with ones (constant density)
    u_cij = np.zeros((2, nx, ny))  # Initialize the velocity array with zeros
    wall_velocity = np.array([0.0, 0.1])  # Set the velocity of the moving wall
    latticeBoltzmann = lbm.LatticeBoltzmann(rho_ij, u_cij, omega)  # Create a LatticeBoltzmann object
    latticeBoltzmann.add_simulation_boundary(bound.Rigid_Bottom_Wall())  # add a rigid bottom wall boundary
    latticeBoltzmann.add_simulation_boundary(bound.Moving_Top_Wall(wall_velocity, rho_ij))  # add a moving top wall boundary

    f_inm = lbm.equilibrium_distribution(latticeBoltzmann.rho_ij, latticeBoltzmann.u_cij)  # Calculate initial equilibrium distribution
    f_eq = f_inm

    for step in range(num_steps):
        # Cache boundary conditions before streaming and collision steps
        latticeBoltzmann.boundaries_cache(f_inm, f_eq, latticeBoltzmann.u_cij)
        # Streaming step to move particles to neighboring cells
        f_inm = lbm.streaming(f_inm)
        # Apply boundary conditions after streaming and collision steps
        latticeBoltzmann.apply_boundaries(f_inm)
        # Collision step to update particle distribution based on equilibrium
        f_inm = lbm.collision(f_inm, omega)
        # Update density and velocity arrays after streaming and collision
        latticeBoltzmann.rho_ij = lbm.compute_density(f_inm)
        latticeBoltzmann.u_cij = lbm.compute_velocity(f_inm, latticeBoltzmann.rho_ij)

        # Visualization and output at certain steps
        if step % 200 == 0:
            print(f"Step {step}/{num_steps} completed")
            # Plot the Couette flow with current velocity and moving wall velocity
            plot_couette_flow(latticeBoltzmann.u_cij, np.array([0.0, 0.1]), step, nx, ny)

    print(f"Simulation completed: {num_steps} steps.")

if __name__ == "__main__":
    # Run the Couette flow simulation with default parameters
    nx= 50
    ny= 50
    omega = 0.3
    num_steps = 2001
    couette_flow(nx,ny,omega,num_steps)

