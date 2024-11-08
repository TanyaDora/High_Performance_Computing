# High Performance Computing with Python

This repository contains an implementation of the **Lattice Boltzmann Method (LBM)** for simulating fluid dynamics using Python. The project utilizes **High Performance Computing (HPC)** techniques, specifically **MPI** (Message Passing Interface), for parallel execution. The code is optimized to efficiently handle complex fluid dynamics problems, making it ideal for large-scale simulations.

## Project Description

The Lattice Boltzmann Method is a computational technique for fluid dynamics that discretizes the Boltzmann transport equation, allowing for the simulation of fluid flows. This project leverages Python to implement LBM with MPI for parallel computing, making it suitable for handling simulations with high computational demands.

## Key Features

- **Discretization of the Boltzmann Transport Equation**: Enables simulation of fluid flows by discretizing the fundamental equations governing fluid dynamics.
- **Periodic Boundary Conditions**: Allows simulation of flows in a repeating domain, which is useful for simulating endless or cyclical fluid environments.
- **Parallel Processing with MPI**: Utilizes `MPI4Py` for efficient parallel execution, significantly enhancing performance and scalability.

## How to Run

1. **Install Required Python Packages**:
   - Install dependencies using pip:
     ```bash
     pip install numpy mpi4py matplotlib
     ```

2. **Set Up Initial Conditions**:
   - Use the provided scripts to define initial conditions for the simulation. Customize these settings based on the specific requirements of your fluid dynamics problem.

3. **Run the Simulation**:
   - Run the simulation using MPI for parallel processing:
     ```bash
     mpiexec -n <num_processes> python simulation_script.py
     ```
     Replace `<num_processes>` with the number of processes you want to run in parallel.

4. **Visualize Results**:
   - Use `matplotlib` to visualize the simulation results. The code includes plotting functions to generate visual representations of fluid flow patterns and other relevant data.

## Contact

For any questions or additional information, please reach out to **Tanya Dora** at [dora.tanya1210@gmail.com](mailto:dora.tanya1210@gmail.com).
