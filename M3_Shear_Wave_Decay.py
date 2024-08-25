# Import necessary libraries
import os
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from Helper import Lattice_Boltzmann_Method as lbm
import numpy as np
import matplotlib.pyplot as plt

# Define paths for storing results
PATH = "results"
path = os.path.join(PATH, 'M3_Shear_Wave_Decay')  # Path for storing shear wave decay results
density_path = os.path.join(path, 'Density')  # Subpath for density profiles
velocity_path = os.path.join(path, 'Velocity')  # Subpath for velocity profiles
viscosity_path = os.path.join(path, 'Viscosity')  # Subpath for viscosity data
os.makedirs(PATH, exist_ok=True)  # Create the main result directory if it doesn't exist
os.makedirs(density_path, exist_ok=True)  # Create the density profile directory if it doesn't exist
os.makedirs(velocity_path, exist_ok=True)  # Create the velocity profile directory if it doesn't exist
os.makedirs(viscosity_path, exist_ok=True)  # Create the viscosity directory if it doesn't exist


# Function to simulate shear wave experiment and calculate viscosity for density or velocity profiles
def shear_wave_viscosity(experiment: str = "density", NX: int = 50, NY: int = 50, omega: float = 1.0,
                         epsilon: float = 0.01,num_steps: int = 2000, rho0: float = 1.0):
    """Simulate shear wave experiment and calculate viscosity.

        Args:
            experiment (str): Experiment type, 'density' or 'velocity'.
            NX (int): Number of grid points in x-direction.
            NY (int): Number of grid points in y-direction.
            omega (float): Relaxation parameter.
            epsilon (float): Amplitude of the perturbation.
            num_steps (int): Number of simulation steps.
            rho0 (float): Initial density .

        Returns:
            simulated_viscosity (float): Simulated viscosity.
            analytical_viscosity (float): Analytical viscosity.
        """

    # Meshgrid for lattice
    X, Y = np.meshgrid(np.arange(NX), np.arange(NY))

    if experiment == "density":
        # Shear wave experiment for density
        rho_ij = rho0 + epsilon * np.sin(2 * np.pi * X / NX)
        u_cij = np.zeros((2, NX, NY))  # Initialize the velocity field array
    elif experiment == "velocity":
        # Shear wave experiment for velocity
        rho_ij = np.ones((NX, NY))  # Constant density for velocity experiment
        u_cij = np.zeros((2, NX, NY))  # Initialize the velocity field array
        u_cij[1, :, :] = epsilon * np.sin(2 * np.pi * Y / NY)  # Set y-component of velocity
    else:
        print("Invalid experiment")
        return
    f_inm = lbm.equilibrium_distribution(rho_ij, u_cij) # Calculate initial equilibrium distribution
    # quantities
    q = [] # List to store amplitude of the shear wave over time


    for step in range(num_steps):
        # Update the lattice using streaming and collision steps
        f_inm = lbm.streaming(f_inm)
        f_inm = lbm.collision(f_inm, omega)
        rho_ij = lbm.compute_density(f_inm)  # Compute density from updated distribution
        u_cij = lbm.compute_velocity(f_inm, rho_ij)  # Compute velocity from updated distribution

        # Calculate amplitude of the wave
        if experiment == "density":
            q.append(np.max(np.abs(rho_ij - rho0)))
            #profile.append(rho_ij[X // 4, 0])
            # Append max density deviation from equilibrium
        else:
            q.append(np.max(np.abs(u_cij[1, :, :])))
            #profile.append(u_cij[0, X//4, 0].max())# Append max y-velocity deviation from equilibrium
    if experiment == 'density':
        q = np.array(q)
        x = argrelextrema(q, np.greater)[0]  # Find local maxima of the amplitude
        q = q[x]
    else:
        x = np.arange(num_steps)  # Create an array of steps for velocity experiment

    coef = 2 * np.pi / NX  # Coefficient for viscosity calculation

    # Define a function to fit the data and extract viscosity
    def visc(t, viscosity):
        return epsilon * np.exp(-viscosity * t * coef ** 2)

    simulated_viscosity = curve_fit(visc, xdata=x, ydata=q)[0][0] # Fit the data to extract simulated viscosity
    analytical_viscosity = (1 / 3) * ((1 / omega) - 0.5)  # Analytical viscosity for comparison

    return simulated_viscosity, analytical_viscosity

# Function to perform the shear wave simulation
def shear_wave_simulation(NX=50, NY=50, w=1.0, epsilon=0.01,num_steps=3001, omega: float = 1.0, experiment_type='velocity', rho0 =1.0):
    """Perform shear wave simulation and plot results.

        Args:
            NX (int): Number of grid points in x-direction.
            NY (int): Number of grid points in y-direction.
            w (float): Frequency of the shear wave.
            epsilon (float): Amplitude of the perturbation.
            num_steps (int): Number of simulation steps.
            omega (float): Relaxation parameter.
            experiment_type (str): Experiment type, 'density' or 'velocity'.
            rho0 (float): Initial density.
        """
    x, y = np.meshgrid(np.arange(NX), np.arange(NY))
    viscosity = (1 / 3) * ((1 / omega) - 0.5)
    if experiment_type == 'density':
        rho_ij = rho0 + epsilon * np.sin(2 * np.pi / NX * x)  # Density with a sinusoidal perturbation
        u_cij = np.zeros((2, NX, NY), dtype=np.float32)  # Initialize the velocity field array
    else:
        rho_ij = np.ones((NX, NY), dtype=np.float32)  # Constant density for velocity experiment
        u_cij = np.zeros((2, NX, NY), dtype=np.float32)  # Initialize the velocity field array
        u_cij[1, :, :] = epsilon * np.sin(2 * np.pi / NY * y)  # Set y-component of velocity

    f_inm = lbm.equilibrium_distribution(rho_ij, u_cij)  # Calculate initial equilibrium distribution

    omega = 1 / w  # Convert relaxation time to collision frequency
    profile = []
    for step in range(num_steps):
        f_inm = lbm.streaming(f_inm)  # Streaming step
        f_inm = lbm.collision(f_inm, omega)  # Collision step
        rho_ij = lbm.compute_density(f_inm)  # Compute density from updated distribution
        u_cij = lbm.compute_velocity(f_inm, rho_ij)  # Compute velocity from updated distribution

        if step % 200 == 0:
            print("time Step" + str(step))
            plt.cla()

            if experiment_type == 'density':
                # Plot Density profile for density shear wave experiment
                plt.cla()
                plt.ylim([-epsilon + rho0, epsilon + rho0])
                plt.plot(np.arange(NX), rho_ij[NY // 2, :], color='red')
                plt.xlabel('x X ' + str(step))
                plt.ylabel('Density')
                plt.title('Density Profile')
                plt.grid(True, linestyle='--', alpha=0.5)
                save_path = os.path.join(density_path, f'Density_shear_wave_decay_{step}.png')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=.2)
            else:
                # Plot Velocity profile for density shear wave experiment
                plt.cla()
                plt.ylim([-epsilon, epsilon])
                plt.plot(np.arange(NY), u_cij[1, :, NX // 2], color='red')
                plt.xlabel('y X ' + str(step))
                plt.ylabel('Velocity')
                plt.title('Velocity Profile')
                plt.grid(True, linestyle='--', alpha=0.5)
                save_path = os.path.join(velocity_path, f'Velocity_shear_wave_decay_{step}.png')
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
        if experiment_type == 'density':
            profile.append(rho_ij[:, NX//4].max())
        else:
            profile.append(u_cij[1, :, NY // 4].max())
    # plot simulated results
    # set up plot
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlim([0, num_steps])
    ax.set_xlabel('Time t')
    ax.legend(['Simulated', 'Analytical'])
    simulated_results = np.array(profile)
    ax.plot(np.arange(num_steps), simulated_results, label="Simulated")
    time_space = np.arange(num_steps)
    def calculate_analytical_sol(epsilon,visc,t, x: np.ndarray, X: int) -> float:
        lx = 2.0 * np.pi / X
        return epsilon * np.exp(- visc * lx ** 2 * t) * np.sin(lx * x)

    # plot analytical results
    if experiment_type == 'density':
        ax.set_ylim(simulated_results.min() - epsilon * 0.1, simulated_results.max() + epsilon * 0.1)
        ax.plot(time_space, rho0 + calculate_analytical_sol(epsilon, viscosity, time_space, NX // 4, NX),
                label="Analytical", linestyle='--',
                dashes=(4, 7), color="orange")
        ax.plot(time_space, rho0 - calculate_analytical_sol(epsilon, viscosity, time_space, NX // 4, NX),
                linestyle='--',
                dashes=(4, 7),
                color="orange")
        plt.xlabel('Time Step')
        plt.ylabel('Density')
        plt.title('Density Time Evolution')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        save_path = os.path.join(density_path, f'Density_Time_Evolution.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    else:
        ax.plot(time_space, calculate_analytical_sol(epsilon, viscosity, time_space, NY // 4, NY),
                label="Analytical",
                linestyle='--',
                dashes=(4, 7))
        ax.set_ylim([-0.0001, +epsilon])
        plt.xlabel('Time Step')
        plt.ylabel('Velocity')
        plt.title('Velocity Time Evolution')
        plt.grid(True, linestyle='--', alpha=0.5)
        save_path = os.path.join(velocity_path, f'Velocity_Time_Evolution.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)

if __name__ == "__main__":
    print("Running shear wave simulations for Density")
    # Run shear wave simulations for both density and velocity profiles
    #shear_wave_simulation(experiment_type="density")
    print("Running shear wave simulations for Velocity")
    #shear_wave_simulation(experiment_type="velocity")
    # Experiments setup
    omegas = np.arange(0.1, 2.01, 0.1)
    print("Running viscosity analysis")
    print("Calculating density viscosity")
    # Shear wave experiment 1 (density)
    density_simulated_viscosities = []
    density_analytical_viscosities = []
    for omega in omegas:
        # Calculate viscosity for density profile experiment at different omega values
        simulated_viscosity, analytical_viscosity = shear_wave_viscosity(omega=omega, experiment='density')
        density_simulated_viscosities.append(simulated_viscosity)
        density_analytical_viscosities.append(analytical_viscosity)
        print(f"Omega = {omega:.2f}: Simulated viscosity = {simulated_viscosity:.6f}, Analytical viscosity = {analytical_viscosity:.6f}")

    print("Calculating velocity viscosity")
    # Shear wave experiment 2 (velocity)
    velocity_simulated_viscosities = []
    velocity_analytical_viscosities = []
    for omega in omegas:
        # Calculate viscosity for velocity profile experiment at different omega values
        simulated_viscosity, analytical_viscosity = shear_wave_viscosity(omega=omega, experiment='velocity')
        velocity_simulated_viscosities.append(simulated_viscosity)
        velocity_analytical_viscosities.append(analytical_viscosity)
        print(f"Omega = {omega:.2f}: Simulated viscosity = {simulated_viscosity:.6f}, Analytical viscosity = {analytical_viscosity:.6f}")

    print("Analysis completed.")
    # Plot density viscosity vs omega
    plt.figure(figsize=(10, 6))
    plt.scatter(omegas, np.log(density_simulated_viscosities), marker='X', color='blue', label='Simulated',
                edgecolors='black')
    plt.scatter(omegas, np.log(density_analytical_viscosities), marker='X', color='red', label='Analytical',
                edgecolors='black')
    plt.xlabel('Omega')
    plt.ylabel('Log(Viscosity)')
    plt.title("Density Viscosity vs Omega")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(viscosity_path, 'Viscosity_density.png')
    plt.savefig(save_path)

    # Plot velocity viscosity vs omega
    plt.figure(figsize=(10, 6))
    plt.scatter(omegas, np.log(velocity_simulated_viscosities), marker='X', color='green', label='Simulated',
                edgecolors='black')
    plt.scatter(omegas, np.log(velocity_analytical_viscosities), marker='X', color='orange', label='Analytical',
                edgecolors='black')
    plt.xlabel('Omega')
    plt.ylabel('Log(Viscosity)')
    plt.title("Velocity Viscosity vs Omega")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(viscosity_path, 'Viscosity_velocity.png')
    plt.savefig(save_path)

    print("Plots saved in the 'result' folder.")