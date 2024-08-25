import os

import numpy as np
from matplotlib import pyplot as plt
import imageio
# Constants
NX = 15  # Number of grid points in the x direction
NY = 10  # Number of grid points in the y direction
# Define paths for storing results
PATH = "results"
Milestone1_path = os.path.join(PATH, 'Milestone_1')
os.makedirs(PATH, exist_ok=True)  # Create the main result directory if it doesn't exist
os.makedirs(Milestone1_path, exist_ok=True)  # Create the density profile directory if it doesn't exist

# Lattice weights and velocity vectors
w_i = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
c_ai = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
                 [0, 0, 1, 0, -1, 1, 1, -1, -1]])
c_ai.setflags(write=False)
w_i.setflags(write=False)

# Initialize density and velocity arrays
X, Y = np.meshgrid(np.arange(NX), np.arange(NY))

#Compute Velocity
def compute_velocity(f, rho):
    u = np.einsum('ij,jkl->ikl', c_ai, f) / rho
    return u

#Compute Density
def compute_density(f):
    rho = np.einsum('ijk->jk', f)
    return rho

# Streaming step
def streaming(f):
        for i in np.arange(1, 9):
            f[i] = np.roll(f[i], shift=c_ai.T[i], axis=(0, 1))
        return f

#Collision step
def collision(f_inm, omega):
    # Update density and velocity arrays
    rho_ij = compute_density(f_inm)
    # Compute velocity field at each lattice point
    u_cij = compute_velocity(f_inm, rho_ij)
    # Calculate equilibrium distribution function
    f_eqm = equilibrium_distribution(rho_ij , u_cij)
    f_new = f_inm + omega * (f_eqm - f_inm)
    return f_new

#Equlibrium Distribution Function
def equilibrium_distribution(rho_nm , u_anm):
    f_eqm = np.einsum('i,nm->inm',w_i , rho_nm) * (1 + 3 * np.einsum('ij,ikl->jkl', c_ai, u_anm) +9/2 * np.einsum('ij,ikl->jkl', c_ai, u_anm)**2 -3/2 * np.einsum('anm,anm->nm', u_anm, u_anm))
    return f_eqm

# Milestone 1 - Density Plot
def visualize_density(rho, title='Density Plot'):
    plt.figure(figsize=(10, 8))
    plt.imshow(rho, cmap='YlGnBu', origin='lower', extent=[0, NX, 0, NY])
    plt.colorbar(label='Density', pad=0.02)
    plt.title(title, fontsize=16)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    #plt.show()

#Milestone 1 - Velocity Plot
def visualize_velocity(u, titles=['Velocity Field (x-component)', 'Velocity Field (y-component)']):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(2):
        im = ax[i].imshow(u[i], cmap='coolwarm', origin='lower', extent=[0, NX, 0, NY])
        fig.colorbar(im, ax=ax[i], label=titles[i], fraction=0.046, pad=0.04)
        ax[i].set_title(titles[i], fontsize=16)
        ax[i].set_xlabel('x', fontsize=14)
        ax[i].set_ylabel('y', fontsize=14)
        ax[i].set_xticks(np.linspace(0, NX, 6))
        ax[i].set_yticks(np.linspace(0, NY, 6))
        ax[i].tick_params(labelsize=12)
        ax[i].grid(False)

    fig.tight_layout()
    #plt.show()

#Milestone -1 Mass Conservation
def mass_conservation(f):
    f_copy = f.copy()
    f = streaming(f)
    mass_conserved = np.allclose(np.sum(f, axis=0), np.sum(f_copy, axis=0))
    return mass_conserved

#MIlestone 2 - Streaming and collision
def boltzmann_equation(f, omega):
    # Streaming step
    f = streaming(f)
    # Collision step
    f = collision(f, omega)
    return f

if __name__ == "__main__":
    f_inm = np.einsum('i,jk->ijk', w_i, np.ones((NX, NY)))
    rho_ij = compute_density(f_inm)  # density

    u_cij = compute_velocity(f_inm, rho_ij)  # velocity field

    # initialize the probability density function with values of your choice.
    f_inm = np.einsum('i,jk->ijk', w_i, np.ones((NX, NY)))

    # visualize density and velocity field
    visualize_density(rho_ij)
    visualize_velocity(u_cij)

    # just random changes in function for better understanding
    # set a simple pattern in f_inm for visualization
    eps = 0.01
    f_inm[1, :, :] = f_inm[1, :, :] + eps
    f_inm[2, :, :] = f_inm[2, :, :] + eps
    f_inm[5] = f_inm[5] * 1.01
    f_inm[7] = f_inm[7] * 0.99
    f_inm[1, NX // 2 - 1, NY // 2 - 1] = 1.0
    f_inm[2, NX // 2, NY // 2 - 1] = 1.0
    f_inm[3, NX // 2 + 1, NY // 2 - 1] = 1.0
    f_inm[1] = np.roll(f_inm[1], (0, 1))

    number_of_iterations = 5
    for n in range(number_of_iterations):
        f_inm = streaming(f_inm)
        rho_ij = compute_density(f_inm)
        visualize_density(rho_ij)
        n += 1

    # test for mass conservation
    print("Mass conserved:", mass_conservation(f_inm))

    # create a short animation that shows how the velocity field travels in time.
    import imageio

    n = 20
    x, y = np.meshgrid(np.arange(NX), np.arange(NY))

    frames = []

    for i in range(n):
        # Streaming step
        f_inm = streaming(f_inm)
        # Density and velocity computation
        rho_ij = compute_density(f_inm)
        u_cij = compute_velocity(f_inm, rho_ij)

        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Density plot
        im1 = ax1.imshow(rho_ij, cmap='YlGnBu', origin='lower', extent=[0, NX, 0, NY])
        ax1.set_title('Density', fontsize=16)
        ax1.set_xlabel('x', fontsize=14)
        ax1.set_ylabel('y', fontsize=14)
        ax1.tick_params(labelsize=12)
        ax1.grid(False)
        plt.colorbar(im1, ax=ax1, label='Density', fraction=0.046, pad=0.04)

        # Velocity streamplot and quiver plot
        strm = ax2.streamplot(x, y, u_cij[0, :, :].T, u_cij[1, :, :].T, color=u_cij[0, :, :].T, cmap='coolwarm')
        ax2.quiver(x, y, u_cij[0, :, :].T, u_cij[1, :, :].T, color='black', scale=10, width=0.002)
        ax2.set_title('Velocity Field', fontsize=16)
        ax2.set_xlabel('x', fontsize=14)
        ax2.set_ylabel('y', fontsize=14)
        ax2.tick_params(labelsize=12)
        ax2.grid(False)

        plt.tight_layout()

        # Save the frame as an image
        frame_filename = os.path.join(Milestone1_path,f'frame_{i}.png')
        plt.savefig(frame_filename)
        plt.close()

        frames.append(imageio.imread(frame_filename))

    # Create a GIF from the frames
    gif_filename = os.path.join(Milestone1_path, 'simulation_animation.gif')
    imageio.mimsave(gif_filename, frames, duration=0.5)

