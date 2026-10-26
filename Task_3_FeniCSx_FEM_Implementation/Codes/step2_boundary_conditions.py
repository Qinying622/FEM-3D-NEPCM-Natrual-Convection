import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import basix.ufl
from dolfinx.mesh import locate_entities_boundary

# 1. Define the 2D square cavity
L = 1.0
H = 1.0

Nx = 20
Ny = 20

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

velocity_element = basix.ufl.element(
    "Lagrange", cell, 2, shape=(2,)
)

pressure_element = basix.ufl.element(
    "Lagrange", cell, 1
)

temperature_element = basix.ufl.element(
    "Lagrange", cell, 2
)

# 3. Create function spaces
mixed_element = basix.ufl.mixed_element(
    [velocity_element, pressure_element]
)

W = fem.functionspace(domain, mixed_element)
V_T = fem.functionspace(domain, temperature_element)

# 4. Identify the four cavity walls
fdim = domain.topology.dim - 1

left_wall = locate_entities_boundary(
    domain, fdim,
    lambda x: np.isclose(x[0], 0.0)
)

right_wall = locate_entities_boundary(
    domain, fdim,
    lambda x: np.isclose(x[0], L)
)

bottom_wall = locate_entities_boundary(
    domain, fdim,
    lambda x: np.isclose(x[1], 0.0)
)

top_wall = locate_entities_boundary(
    domain, fdim,
    lambda x: np.isclose(x[1], H)
)

# 5. Display the number of boundary facets
if MPI.COMM_WORLD.rank == 0:
    print("Boundary identification completed.")
    print(f"Left wall facets: {len(left_wall)}")
    print(f"Right wall facets: {len(right_wall)}")
    print(f"Bottom wall facets: {len(bottom_wall)}")
    print(f"Top wall facets: {len(top_wall)}")
