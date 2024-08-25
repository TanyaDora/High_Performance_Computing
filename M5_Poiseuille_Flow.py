# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Import custom helper modules
from Helper import Lattice_Boltzmann_Method as lbm
from Helper import Boundary_Conditions as bound

# Define paths for storing results
PATH = "results"  # Main result directory
path = os.path.join(PATH, 'M5_Poiseuille_Flow')  # Path for storing Poiseuille flow results
os.makedirs(path, exist_ok=True)  # Create the directory if it doesn't exist
cs = 1/np.sqrt(3)

# Function to plot velocity profiles
def plot(pressure_right, pressure_left, rho_ij, u_cij, step, nx, ny):
    y = np.arange(ny)
    plt.cla()  # Clear current plot
    # Calculate dynamic viscosity based on the LBM parameter omega
    viscosity = 1 / 3 * (1 / omega - 0.5)
    # Calculate dynamic viscosity for inlet and outlet conditions
    dynamic_viscosity = rho_ij[nx // 2, :].mean() * viscosity
    # Calculate partial derivative of pressure along the y direction
    partial_derivative = cs ** 2 * (pressure_right - pressure_left) / nx
    # Calculate analytical velocity profile using the Poiseuille formula
    analytical = -0.5 * (1 / dynamic_viscosity) * partial_derivative * y * (ny - 1 - y)

    plt.figure(figsize=(10, 6))  # Set the figure size

    # Plot the analytical velocity profile with a solid line and label
    plt.plot(analytical, y, label="Analytical", color='blue', linewidth=2)

    # Plot the simulated velocity profile with dashed lines and label
    plt.plot(u_cij[0, nx // 2, :], y, linestyle='dashed', label="Simulated", color='black', linewidth=2)

    plt.ylabel('y')
    plt.xlabel('velocity')
    plt.title(f'Poiseuille Flow Simulation (Step {step})')
    plt.legend()

    # Customize the plot appearance
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.tick_params(axis='both', which='both', direction='in', width=0.5, bottom=True, top=False, left=True, right=False)
    plt.minorticks_on()
    plt.tick_params(which='minor', axis='both', width=0.5, direction='in')
    plt.legend(frameon=False, loc='upper right')

    # Save the plot with better spacing and display it
    save_path = os.path.join(path, f'Poiseuille_Flow_{step}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)

# Function to perform Poiseuille flow simulation
def poiseuille_flow(nx, ny, pressure_left, pressure_right, omega,num_steps):
    # Initialize density and velocity arrays
    rho_ij = np.ones((nx, ny))
    u_cij = np.zeros((2, nx, ny))

    # Initialize the Lattice Boltzmann simulation
    latticeBoltzmann = lbm.LatticeBoltzmann(rho_ij, u_cij, omega)

    # Add simulation boundaries (periodic, rigid walls)
    latticeBoltzmann.add_simulation_boundary(bound.Left_Periodic_Pressure(ny, pressure_left))
    latticeBoltzmann.add_simulation_boundary(bound.Right_Periodic_Pressure(ny, pressure_right))
    latticeBoltzmann.add_simulation_boundary(bound.Rigid_Top_Wall())
    latticeBoltzmann.add_simulation_boundary(bound.Rigid_Bottom_Wall())

    # Main simulation loop
    for step in range(num_steps):
        latticeBoltzmann.Poiseuille_streaming_and_colliding()  # lbm step
        u_cij = latticeBoltzmann.u_cij  # update velocity
        rho_ij = latticeBoltzmann.rho_ij # update density

        # plot every 200 steps
        if ((step % 1000 == 0)):
            plot(pressure_right,pressure_left,rho_ij,u_cij,step,nx,ny)
            print(f"Step {step}/{num_steps} completed")

# Main block to run the simulation
if __name__ == "__main__":
    # Run the Poiseuille flow simulation with default parameters
    nx= 50
    ny= 50
    pressure_left = 0.755
    pressure_right = 0.75
    omega = 1.0
    num_steps = 10001
    # Call the Poiseuille flow simulation function
    poiseuille_flow(nx,ny, pressure_left, pressure_right, omega, num_steps)