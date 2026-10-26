import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import basix.ufl

# 1. Define the 2D square cavity
L = 1.0  # Cavity width (m)
H = 1.0  # Cavity height (m)

Nx = 20  # Number of mesh divisions in x-direction
Ny = 20  # Number of mesh divisions in y-direction

domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [
        np.array([0.0, 0.0]),
        np.array([L, H])
    ],
    [Nx, Ny],
    cell_type=mesh.CellType.triangle
)

# 2. Define finite element types
cell = domain.basix_cell()

# Quadratic velocity element: P2
velocity_element = basix.ufl.element(
    "Lagrange", cell, 2, shape=(2,)
)

# Linear pressure element: P1
pressure_element = basix.ufl.element(
    "Lagrange", cell, 1
)

# Quadratic temperature element: P2
temperature_element = basix.ufl.element(
    "Lagrange", cell, 2
)

# 3. Combine velocity and pressure into a mixed space
mixed_element = basix.ufl.mixed_element(
    [velocity_element, pressure_element]
)

W = fem.functionspace(domain, mixed_element)

# 4. Create a separate temperature function space
V_T = fem.functionspace(domain, temperature_element)

# 5. Display basic information
if MPI.COMM_WORLD.rank == 0:
    print("Mesh and function spaces created successfully.")
    print(f"Number of cells: {domain.topology.index_map(domain.topology.dim).size_global}")
    print(f"Velocity-pressure space dimension: {W.dofmap.index_map.size_global}")
    print(f"Temperature space dimension: {V_T.dofmap.index_map.size_global}")
