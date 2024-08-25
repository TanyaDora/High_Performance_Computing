# Import necessary libraries
import os
from PIL import Image
import glob
import numpy as np
# Import custom helper modules
from Helper import Lattice_Boltzmann_Method as lbm
from Helper import Boundary_Conditions as bound
import matplotlib.pyplot as plt
# Define paths for storing results
PATH = "results"
path = os.path.join(PATH, 'M6_Sliding_Lid')
os.makedirs(path, exist_ok=True)
cs = 1/np.sqrt(3)


def plot(velocities, step, nx, ny):
    plt.cla()
    plt.xlim([0, nx])
    plt.xlim([0, ny])
    # Calculate velocity magnitude from velocity components
    v = np.sqrt(velocities.T[:, :, 0] ** 2 + velocities.T[:, :, 1] ** 2)
    # Display velocity magnitude as an image with streamlines
    plt.imshow(v, cmap='viridis', vmin=0, interpolation='spline16')
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    # Add streamlines to visualize the flow
    plt.streamplot(x, y, velocities.T[:, :, 0], velocities.T[:, :, 1], color='black', linewidth=1, density=2)
    plt.legend(['Analytical', 'Simulated'], loc = "lower left")
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title(f'Sliding Lid Simulation (Step {step})')

    # Customize the plot appearance
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.tick_params(axis='both', which='both', direction='in', width=0.5, bottom=True, top=False, left=True,
                    right=False)
    plt.minorticks_on()
    plt.tick_params(which='minor', axis='both', width=0.5, direction='in')
    save_path = os.path.join(path, f'sliding_lid_{step}')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)

def animation():
    print("Creating animation")
    print(path)
    images =[]
    filenames = glob.glob(path + '/sliding_lid_*.png')
    print(filenames)
    for i in filenames:
        new_frame = Image.open(i)
        images.append(new_frame)
    # Save images as a GIF animation
    if images:
        images[0].save(path + '/Sliding_lid.gif', format='GIF', append_images=images[1:], save_all=True, duration=300,
                       loop=0)
        print("Animation created successfully.")
    else:
        print("No images found.")

def sliding_lid(nx, ny, omega, steps, reynold_number):
    print("Running sliding lid simulation")
    rho_ij = np.ones((nx, ny))
    u_cij = np.zeros((2, nx, ny))
    wall_velocity = np.array([0.0, 0.05])
    latticeBoltzmann = lbm.LatticeBoltzmann(rho_ij, u_cij, omega)
    # Add simulation boundaries (moving top wall, rigid walls)
    latticeBoltzmann.add_simulation_boundary(bound.Moving_Top_Wall(wall_velocity, rho_ij))
    latticeBoltzmann.add_simulation_boundary(bound.Rigid_Bottom_Wall())  # add a rigid bottom wall boundary
    latticeBoltzmann.add_simulation_boundary(bound.Rigid_Left_Wall())
    latticeBoltzmann.add_simulation_boundary(bound.Rigid_Right_Wall())
    omega = 1 / (0.5 + ((wall_velocity[1] * nx) / reynold_number) / (1/3))
    # Initialize equilibrium distributions
    f_inm = lbm.equilibrium_distribution(rho_ij, u_cij)
    f_eq = lbm.equilibrium_distribution(rho_ij, u_cij)

    for (step) in range(steps):
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

        # plot sliding lid every 1000 steps
        if ((step % 1000 == 0)):
            plot(latticeBoltzmann.u_cij, step, nx, ny)
            print(f"Step {step}/{steps} completed")
    print("Sliding lid simulation completed.")

# Main block to run the sliding lid simulation
if __name__ == "__main__":
    nx = 300
    ny = 300
    omega= 0.3
    steps =100000
    reynold_number= 1000
    sliding_lid(nx, ny, omega, steps, reynold_number )
    animation()