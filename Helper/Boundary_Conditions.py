from abc import abstractmethod, ABC
import numpy as np
from Helper import Lattice_Boltzmann_Method as lbm

# Constants
NX = 50  # Number of grid points in the x direction
NY = 50  # Number of grid points in the y direction
cs = 1/np.sqrt(3)
# Lattice weights and velocity vectors using a non-alternative (original) formulation
w_i = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])  # Weight for each lattice direction
c_ai = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],  # X component of velocity for each lattice direction
                 [0, 0, 1, 0, -1, 1, 1, -1, -1]])  # Y component of velocity for each lattice direction
c_ai.setflags(write=False)  # Ensure that the c_ai array is read-only (to avoid accidental changes)
w_i.setflags(write=False)  # Ensure that the w_i array is read-only


class Boundary_Conditions:
    def __init__(self):
        """
        Initialize a Boundary object.
        """
        self.boundary_cached_pdf = None  # Cache for storing probability density function
        self.input_channel_indices = None  # Indices of incoming populations
        self.output_channel_indices = None  # Indices of outgoing populations

    def pre_streaming_boundary_function(self) -> None:
        """
        Apply pre-streaming boundary conditions.
        """
        raise NotImplementedError("Subclasses must implement the pre_streaming method.")

    def post_streaming_boundary_function(self) -> None:
        """
        Apply post-streaming boundary conditions.
        """
        raise NotImplementedError("Subclasses must implement the post_streaming method.")

    def set_channel_indices(self, input, output):
        """
        Set the input and output channel indices for a boundary condition.

        :param input_indices: Array of input channel indices.
        :param output_indices: Array of output channel indices.
        """
        self.input_channel_indices = input
        self.output_channel_indices = output
class Rigid_Left_Wall(Boundary_Conditions):
    def __init__(self):
        # Input and output channel indexes for the RigidLeftWallBoundary
        super().__init__()
        # Set input and output channel indices for the RigidLeftWallBoundary
        self.set_channel_indices(np.array([
            1,  # Right-moving particle (left to right)
            5,  # Up-moving particle (bottom to top)
            8  # Down-left-moving particle (top-left to bottom-right diagonal)
        ]),np.array([
            3,  # Left-moving particle (right to left)
            7,  # Down-right-moving particle (top-right to bottom-left diagonal)
            6  # Down-moving particle (top to bottom)
        ]))

    def pre_streaming_boundary_function(self, f_inm):
        """
                Cache the distribution function values at the boundary before the streaming step.

                This function caches the distribution function values at the boundary for the pre-streaming boundary conditions.
                It stores the distribution function values at the leftmost lattice site for each channel.

                :param f: Distribution function to be cached.
                """
        # Cache the distribution function values at the leftmost lattice site for each channel

        self.boundary_cached_pdf = f_inm[:, 0, :]

    def post_streaming_boundary_function(self, f_inm):
        """
                Apply post-streaming boundary conditions for a rigid left wall.

                This function updates the distribution functions after the streaming step based on the rigid left wall boundary
                conditions. It adjusts the distribution function values in the input channels.

                :param f: Distribution function after streaming and collision.
                """
        # Update the distribution function's input channels based on the cached values

        f_inm[self.input_channel_indices,0, :] = self.boundary_cached_pdf[self.output_channel_indices, :]
class Rigid_Right_Wall(Boundary_Conditions):
    def __init__(self):
        super().__init__()
        # Set input and output channel indices for the RigidRightWallBoundary
        self.set_channel_indices(
            np.array([
                3,  # Left-moving particle (right to left)
                6,  # Down-moving particle (top to bottom)
                7  # Down-right-moving particle (top-right to bottom-left diagonal)
            ]),
            np.array([
                1,  # Right-moving particle (left to right)
                8,  # Down-left-moving particle (top-left to bottom-right diagonal)
                5  # Up-moving particle (bottom to top)
            ])
        )

    def pre_streaming_boundary_function(self, f_inm):
        """
                Cache the distribution function values at the boundary before the streaming step.

                This function caches the distribution function values at the boundary for the pre-streaming boundary conditions.
                It stores the distribution function values at the rightmost lattice site for each channel.

                :param f_inm: Distribution function to be cached.
                """
        # Cache the distribution function values at the rightmost lattice site for each channel
        self.boundary_cached_pdf = f_inm[:, -1, :]

    def post_streaming_boundary_function(self, f_inm):
        """
                Apply post-streaming boundary conditions for a rigid right wall.

                This function updates the distribution functions after the streaming step based on the rigid right wall boundary
                conditions. It adjusts the distribution function values in the input channels.

                :param f_inm: Distribution function after streaming and collision.
                """
        # Update the distribution function's input channels based on the cached values
        f_inm[self.input_channel_indices, -1,:] = self.boundary_cached_pdf[self.output_channel_indices, :]
class Rigid_Bottom_Wall(Boundary_Conditions):
    def __init__(self):
        super().__init__()
        # Set input and output channel indices for the RigidBottomWallBoundary
        self.set_channel_indices(
            np.array([
                4,  # Up-moving particle (bottom to top)
                7,  # Down-right-moving particle (top-right to bottom-left diagonal)
                8  # Down-left-moving particle (top-left to bottom-right diagonal)
            ]),
            np.array([
                2,  # Left-moving particle (right to left)
                5,  # Up-moving particle (bottom to top)
                6  # Down-moving particle (top to bottom)
            ])
        )

    def pre_streaming_boundary_function(self, f_inm):
        """
                Cache the distribution function values at the boundary before the streaming step.

                This function caches the distribution function values at the boundary for the pre-streaming boundary conditions.
                It stores the distribution function values at the bottommost lattice site for each channel.

                :param f_inm: Distribution function to be cached.
                """
        # Cache the distribution function values at the bottommost lattice site for each channel
        self.boundary_cached_pdf = f_inm[:, :, -1]

    def post_streaming_boundary_function(self, f_inm):
        """
                Apply post-streaming boundary conditions for a rigid bottom wall.

                This function updates the distribution functions after the streaming step based on the rigid bottom wall boundary
                conditions. It adjusts the distribution function values in the input channels.

                :param f_inm: Distribution function after streaming and collision.
                """
        # Update the distribution function's input channels based on the cached values
        f_inm[self.input_channel_indices, :, -1] = self.boundary_cached_pdf[self.output_channel_indices, :]
class Rigid_Top_Wall(Boundary_Conditions):
    def __init__(self):
        super().__init__()
        # Set input and output channel indices for the RigidTopWallBoundary
        self.set_channel_indices(
            np.array([
                2,  # Left-moving particle (right to left)
                5,  # Up-moving particle (bottom to top)
                6  # Down-moving particle (top to bottom)
            ]),
            np.array([
                4,  # Right-moving particle (left to right)
                7,  # Down-right-moving particle (top-right to bottom-left diagonal)
                8  # Down-left-moving particle (top-left to bottom-right diagonal)
            ])
        )

    def pre_streaming_boundary_function(self, f_inm):
        """
                Cache the distribution function values at the boundary before the streaming step.

                This function caches the distribution function values at the boundary for the pre-streaming boundary conditions.
                It stores the distribution function values at the topmost lattice site for each channel.

                :param f_inm: Distribution function to be cached.
                """
        # Cache the distribution function values at the topmost lattice site for each channel
        self.boundary_cached_pdf = f_inm[:, :, 0]

    def post_streaming_boundary_function(self, f_inm):
        """
                Apply post-streaming boundary conditions for a rigid top wall.

                This function updates the distribution functions after the streaming step based on the rigid top wall boundary
                conditions. It adjusts the distribution function values in the input channels.

                :param f_inm: Distribution function after streaming and collision.
                """
        # Update the distribution function's input channels based on the cached values
        f_inm[self.input_channel_indices, :, 0] = self.boundary_cached_pdf[self.output_channel_indices, :]

class Moving_Top_Wall():
    class Moving_Wall(Boundary_Conditions, ABC):

        def __init__(self, velocity, density):
            super().__init__()
            self.velocity = velocity
            self.density = density[1]
            """
                    Calculate the momentum exerted on a moving wall due to particle collisions.
        
                    This function calculates the momentum for a moving wall boundary condition by computing the change in momentum
                    resulting from particle collisions. The momentum is computed based on the given velocity and density of the
                    moving wall.
        
                    :param f: Distribution function containing particle distributions.
                    :return: Momentum exerted on the moving wall due to particle collisions.
                    """

        def calculate_momentum_for_moving_wall(self, f_inm):
            # Calculate the density of particles impacting the wall at each lattice site
            density = np.sum(f_inm[:, :, 0], axis=0)
            # Calculate the momentum exerted on the moving wall due to particle collisions
            # The momentum is computed based on the given velocity, density, lattice weights, and lattice speed of sound
            momentum_for_moving_wall = (np.matmul(np.flip(c_ai.T, axis=1), self.velocity)) * (w_i * density[:, None]) * (
                        2 * (1 / cs) ** 2)
            # Extract and return the momentum values at the output_channel_indices
            return momentum_for_moving_wall[:, self.output_channel_indices]
    def __init__(self, velocity, density):
        self.moving_wall = self.Moving_Wall(velocity, density)
        """
                Initialize a moving top wall boundary condition.

                This class inherits from the Moving_Wall class and customizes the input and output channel indexes
                for a moving top wall boundary condition.

                :param velocity: Velocity of the moving wall.
                :param density: Density of particles at the moving wall.
                """

        #super().__init__(velocity, density)
        # Input and output channel indices for the Moving Top Wall Boundary Condition
        self.moving_wall.set_channel_indices(
            np.array([
                2,  # Left-moving particle (top-left to bottom-right diagonal)
                5,  # Up-moving particle (bottom to top)
                6  # Down-moving particle (top to bottom)
            ]),
            np.array([
                4,  # Right-moving particle (top-right to bottom-left diagonal)
                7,  # Down-right-moving particle (top-right to bottom-left diagonal)
                8  # Down-left-moving particle (top-left to bottom-right diagonal)
            ])
        )

    def post_streaming_boundary_function(self, f_inm):
        """
                Apply post-streaming boundary conditions for a moving top wall.

                This function updates the distribution functions after the streaming step based on the moving top wall boundary
                conditions. It calculates the momentum exerted on the wall and adjusts the relevant PDF values.

                :param f: Distribution function after streaming and collision.
                """
        momentum_for_moving_top_wall = self.moving_wall.calculate_momentum_for_moving_wall(f_inm)
        boundary_adjustment = (self.boundary_cached_pdf.T[:, self.moving_wall.output_channel_indices] - momentum_for_moving_top_wall).T  # value of PDF before streaming - momentum
        f_inm[self.moving_wall.input_channel_indices, :, 0] = boundary_adjustment


    def pre_streaming_boundary_function(self, f_inm):
        """
           Cache the distribution function values at the boundary before the streaming step.

           This function caches the distribution function values at the boundary for the pre-streaming boundary conditions.
           It stores the distribution function values at the zero-th lattice site for each channel.

           :param f: Distribution function to be cached.
           """
        # Cache the distribution function values at the zero-th lattice site for each channel
        self.boundary_cached_pdf = f_inm[:, :, 0]
class Left_Periodic_Pressure(Boundary_Conditions):
    def __init__(self, Left_pressure, Length_of_Boundary):
        super().__init__()
        self.left_pressure = Left_pressure
        self.length_of_boundary = Length_of_Boundary
        # Input and output channel indices for the LeftPeriodicBoundaryCondition
        self.set_channel_indices(
            np.array([
                3,  # Left-moving particle (right to left)
                6,  # Down-moving particle (top to bottom)
                7  # Down-right-moving particle (top-right to bottom-left diagonal)
            ]),
            np.array([
                1,  # Right-moving particle (left to right)
                8,  # Down-left-moving particle (top-left to bottom-right diagonal)
                5  # Up-moving particle (bottom to top)
            ])
        )
        self.input_density = np.full(self.left_pressure,self.length_of_boundary)

    def pre_streaming_boundary_function(self, f_inm, f_eq, u_cij):
        """
                Apply pre-streaming periodic boundary conditions for the left boundary.

                This function adjusts the distribution function values at the left boundary before the streaming step based
                on the periodic boundary conditions. It updates the input channels to ensure continuity with the corresponding
                lattice sites on the right boundary.

                :param f_inm: Distribution function before streaming.
                :param f_eq: Equilibrium distribution function.
                :param velocities: Lattice velocities.
                """
        # Calculate the equilibrium distribution function for the input density and velocity
        f_eq_input = lbm.equilibrium_distribution(self.input_density, u_cij[:, -2]).squeeze()
        # Update the distribution function's input channels based on the periodic conditions
        f_inm[:, 0, :] = f_eq_input + (f_inm[:, -2, :] - f_eq[:, -2, :])

    def post_streaming_boundary_function(self, f_inm):
        """
                Apply post-streaming boundary conditions for a left periodic boundary.

                This function is intentionally left empty, as there are no post-streaming conditions to be applied.

                :param f_inm: Distribution function after streaming and collision.
                """
        pass
class Right_Periodic_Pressure(Boundary_Conditions):
    def __init__(self, right_pressure,Length_of_Boundary):
        super().__init__()
        self.right_pressure = right_pressure
        self.length_of_Boundary = Length_of_Boundary
        # Input and output channel indices for the RightPeriodicBoundaryCondition
        self.set_channel_indices(
            np.array([
                1,  # Right-moving particle (left to right)
                8,  # Down-left-moving particle (top-left to bottom-right diagonal)
                5  # Up-moving particle (bottom to top)
            ]),
            np.array([
                3,  # Left-moving particle (right to left)
                6,  # Down-moving particle (top to bottom)
                7  # Down-right-moving particle (top-right to bottom-left diagonal)
            ])
        )
        self.output_density = np.full(self.right_pressure,self.length_of_Boundary)

    def pre_streaming_boundary_function(self, f_inm, f_eq, u_ij):
        """
                Apply pre-streaming periodic boundary conditions for the right boundary.

                This function adjusts the distribution function values at the right boundary before the streaming step based
                on the periodic boundary conditions. It updates the input channels to ensure continuity with the corresponding
                lattice sites on the left boundary.

                :param f_inm: Distribution function before streaming.
                :param f_eq: Equilibrium distribution function.
                :param velocities: Lattice velocities.
                """
        # Calculate the equilibrium distribution function for the output density and velocity
        f_eq_output = lbm.equilibrium_distribution(self.output_density, u_ij[:, 1]).squeeze()
        # Update the distribution function's input channels based on the periodic conditions
        f_inm[:, -1, :] = f_eq_output + (f_inm[:, 1, :] - f_eq[:, 1, :])

    def post_streaming_boundary_function(self, f_inm):
        """
                Apply post-streaming boundary conditions for a right periodic boundary.

                This function is intentionally left empty, as there are no post-streaming conditions to be applied.

                :param f_inm: Distribution function after streaming and collision .
                """
        pass


