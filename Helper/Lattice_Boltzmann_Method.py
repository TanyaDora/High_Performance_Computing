import numpy as np
from Helper import Boundary_Conditions as bound

# Constants
NX = 50  # Number of grid points in the x direction
NY = 50  # Number of grid points in the y direction

# Lattice weights and velocity vectors using a non-alternative (original) formulation
w_i = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])  # Weight for each lattice direction
c_ai = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],  # X component of velocity for each lattice direction
                 [0, 0, 1, 0, -1, 1, 1, -1, -1]])  # Y component of velocity for each lattice direction
c_ai.setflags(write=False)  # Ensure that the c_ai array is read-only (to avoid accidental changes)
w_i.setflags(write=False)  # Ensure that the w_i array is read-only

# Initialize density and velocity arrays using meshgrid
X, Y = np.meshgrid(np.arange(NX), np.arange(NY))

# Lattice Boltzmann class
class LatticeBoltzmann():
    def __init__(self, rho_ij, u_cij, omega: float) -> None:
        """Initializes the LatticeBoltzmann simulation.
                :param rho_ij: Initial lattice density array.
                :param u_cij: Initial lattice velocity array.
                :param omega: Relaxation parameter for collision step.
                """
        assert 0 < omega < 2  # check that omega lies in boundaries between 0 and 2
        self.rho_ij = rho_ij  # Lattice density array
        self.u_cij = u_cij  # Lattice velocity array
        self.omega = omega  # Relaxation parameter
        self.boundaries = [] # List of boundary objects
        self.f_inm = equilibrium_distribution(self.rho_ij, self.u_cij)  # Initial distribution function
        self.f_eq = self.f_inm  # Equilibrium distribution function

    def Poiseuille_stream(self):
        #print("Performing streaming step for Poiseuille flow")
        """
                ->Move particle distributions by one grid cell's distance. Adjusts the distribution components in response to their respective directions along the grid.
                ->Perform streaming step to propagate channel occupations by one cell distance.
                f_inm : Occupancy numbers are organized within a 3-dimensional array, with the initial dimension representing distinct channels from 0 to 8.
            The other two dimensions correspond to the x and y positions.
        """
        for i in np.arange(1, 9):
            self.f_inm[i] = np.roll(self.f_inm[i], shift=c_ai.T[i], axis=(0, 1))

    def Poiseuille_collide(self):
        #print("Performing collision step for Poiseuille Flow")
        """
        Execute the collision step to enhance and update the Probability Distribution Functions (PDFs) at every point within the lattice grid.
        During this process, the PDFs are adjusted to account for particle collisions and interactions, leading to the evolution of the simulated fluid flow dynamics and properties.
        This step contributes to the accurate modeling of physical phenomena within the lattice Boltzmann simulation.
        """
        self.rho_ij = compute_density(self.f_inm)  # calculate average density
        self.u_cij = compute_velocity(self.f_inm, self.rho_ij)  # calculate average velocity
        feq_ixy = equilibrium_distribution(self.rho_ij, self.u_cij)  # calculate equilibrium distribution
        self.f_inm += self.omega * (feq_ixy - self.f_inm)  # update f_inm

    def Poiseuille_streaming_and_colliding(self):# boundaries, f_inm, f_eq, u_cij, omega
        #print("Starting Poiseuille streaming and colliding")
        """
        Execute streaming and collision steps to simulate fluid flow using the lattice Boltzmann method.
        """
        self.Poiseuille_before_streaming()
        self.Poiseuille_stream()
        self.Poiseuille_after_streaming()
        self.Poiseuille_collide()
    def add_simulation_boundary(self, boundary):
        """
        Add a boundary object to the simulation.
        :param boundary: Boundary to be added to the simulation.
        """
        self.boundaries.append(boundary)

    #Helper function to apply pre-streaming boundary conditions
    def boundaries_cache(self, f_inm, f_eq, u_cij):
        #print("Applying pre-streaming boundary conditions")
        """
                Apply pre-streaming boundary conditions for all boundaries in the simulation.

                :param f_inm: Distribution function before streaming.
                :param f_eq: Equilibrium distribution function.
                :param u_cij: Lattice velocity array.
        """
        for boundary in self.boundaries:
                boundary.pre_streaming_boundary_function(f_inm)

    #Helper function to apply post-streaming boundary conditions (bounce-back particles)
    def apply_boundaries(self,f_inm):
        #print("Applying post-streaming boundary conditions")
        """
                Apply post-streaming boundary conditions (bounce-back particles) for all boundaries in the simulation.

                :param f_inm: Distribution function after streaming.
                """
        for boundary in self.boundaries:
                boundary.post_streaming_boundary_function(f_inm)


    def Poiseuille_before_streaming(self,):
        #print("Applying pre-streaming boundary conditions (Poiseuille)")
        """
        Apply appropriate boundary conditions to the distribution functions in the grid nodes adjacent to the boundaries
        """
        for boundary in self.boundaries:
            if isinstance(boundary, bound.Right_Periodic_Pressure) or isinstance(boundary,bound.Left_Periodic_Pressure):
                boundary.pre_streaming_boundary_function(self.f_inm, self.f_eq, self.u_cij)
            else:
                boundary.pre_streaming_boundary_function(self.f_inm)

    # Bounce back particles from a wall
    def Poiseuille_after_streaming(self):
        #print("Applying post-streaming boundary conditions (Poiseuille)...")
        """
        Update the distribution functions at the boundary nodes according to the boundary conditions
        """
        for boundary in self.boundaries:
            boundary.post_streaming_boundary_function(self.f_inm)

def before_streaming(boundaries, f_inm,f_eq,velocity):
    """
    Apply appropriate boundary conditions to the distribution functions in the grid nodes adjacent to the boundaries.
    This function is called before the streaming step.
    """
    for boundary in boundaries:
        if isinstance(boundary, bound.Right_Periodic_Pressure) or isinstance(boundary,
                                                                              bound.Left_Periodic_Pressure):
            boundary.pre_streaming_boundary_function(f_inm, f_eq, velocity)
        else:
            boundary.pre_streaming_boundary_function(f_inm)

# Bounce back particles from a wall
def after_streaming(boundaries,f_inm):
    """
    Update the distribution functions at the boundary nodes according to the boundary conditions.
    This function is called after the streaming step and collision step.
    """
    for boundary in boundaries:
        boundary.post_streaming(f_inm)
# Streaming step to move particles
def streaming(f_inm):
    #print("Performing streaming step")
    """
        Perform the streaming step to move particles within the lattice.
        This function updates the distribution functions by shifting particle distributions along the grid based on their respective directions.

        :param f_inm: Distribution function before streaming.
        :return: Updated distribution function after streaming.
        """
    for i in np.arange(1, 9):
        f_inm[i] = np.roll(f_inm[i], shift=c_ai.T[i], axis=(0, 1))
    return f_inm


# Collision step to update the distribution function
def collision(f_inm, omega):
    ##print("Performing scollision step")
    """
        Perform the collision step of the Lattice Boltzmann simulation.

        :param f_inm: Distribution function before collision.
        :param omega: Relaxation parameter for collision step.
        :return: Updated distribution function after collision.
        """
    # Update density and velocity arrays
    rho_ij = compute_density(f_inm)
    # Compute velocity field at each lattice point
    u_cij = compute_velocity(f_inm, rho_ij)
    # Calculate equilibrium distribution function
    f_eqm = equilibrium_distribution(rho_ij, u_cij)
    # Update the distribution function using the collision formula
    f_new = f_inm + omega * (f_eqm - f_inm)
    return f_new

#Compute Velocity at each lattice point
def compute_velocity(f, rho) ->np.ndarray:
    #print("Calculate the velocity field")
    """
        Calculate the velocity field at each lattice point based on distribution functions and density.

        :param f: Distribution functions.
        :param rho: Density at each lattice point.
        :return: Velocity field.
        """
    u = np.einsum('ij,jkl->ikl', c_ai, f) / rho
    return u

#Compute Density at each lattice point
def compute_density(f) -> np.ndarray:
    #print("Calculate the density")
    """
        Calculate the density at each lattice point based on distribution functions.

        :param f: Distribution functions.
        :return: Density at each lattice point.
        """
    #rho = np.einsum('ijk->jk', f)
    #return rho
    rho = np.sum(f, axis=0)
    return rho

#Equlibrium Distribution Function
def equilibrium_distribution(rho_nm , u_anm) -> np.ndarray:
    #print("Calculate the equilibrium distribution function ")
    """
        Calculate the equilibrium distribution function based on density and velocity.

        :param rho_nm: Density at each lattice point.
        :param u_anm: Velocity at each lattice point.
        :return: Equilibrium distribution function.
        """
    #f_eqm = np.einsum('i,nm->inm',w_i , rho_nm) * (1 + 3 * np.einsum('ij,ikl->jkl', c_ai, u_anm) +9/2 * np.einsum('ij,ikl->jkl', c_ai, u_anm)**2 -3/2 * np.einsum('anm,anm->nm', u_anm, u_anm))
    #return f_eqm
    Value_1 = np.dot(u_anm.T, c_ai).T
    Value_2 = np.sum(u_anm ** 2, axis=0)
    return (w_i * (rho_nm * (1 + 3 * Value_1 + (9 / 2) * Value_1 ** 2 - (3 / 2) * Value_2)).T).T

