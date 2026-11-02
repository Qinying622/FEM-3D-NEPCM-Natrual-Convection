from dolfinx import fem, mesh
import numpy as np

from step1_mesh_spaces import domain, W, V_T

from step2_boundary_conditions import (
    left_wall,
    right_wall,
    bottom_wall,
    top_wall
)

# ==================================================
# STEP 2B.1: Combine all boundary facets
# ==================================================

fdim = domain.topology.dim - 1

# Create connectivity between boundary facets and cells
domain.topology.create_connectivity(fdim, domain.topology.dim)

all_boundary_facets = np.unique(
    np.concatenate((
        left_wall,
        right_wall,
        bottom_wall,
        top_wall
    ))
)

# ==================================================
# STEP 2B.2: Apply no-slip velocity boundary conditions
# ==================================================

V_u, _ = W.sub(0).collapse()

u_zero = fem.Function(V_u)
u_zero.x.array[:] = 0.0
u_zero.x.scatter_forward()

dofs_u = fem.locate_dofs_topological(
    (W.sub(0), V_u),
    fdim,
    all_boundary_facets
)

bc_u = fem.dirichletbc(
    u_zero,
    dofs_u,
    W.sub(0)
)

# ==================================================
# STEP 2B.3: Apply hot-wall temperature condition
# ==================================================

theta_hot = fem.Function(V_T)
theta_hot.x.array[:] = 1.0
theta_hot.x.scatter_forward()

dofs_theta_left = fem.locate_dofs_topological(
    V_T,
    fdim,
    left_wall
)

bc_theta_left = fem.dirichletbc(
    theta_hot,
    dofs_theta_left
)

# ==================================================
# STEP 2B.4: Apply cold-wall temperature condition
# ==================================================

theta_cold = fem.Function(V_T)
theta_cold.x.array[:] = 0.0
theta_cold.x.scatter_forward()

dofs_theta_right = fem.locate_dofs_topological(
    V_T,
    fdim,
    right_wall
)

bc_theta_right = fem.dirichletbc(
    theta_cold,
    dofs_theta_right
)

# ==================================================
# STEP 2B.5: Collect boundary conditions
# ==================================================

bcs_velocity = [bc_u]

bcs_temperature = [
    bc_theta_left,
    bc_theta_right
]

print("Step 2B: Boundary conditions applied successfully!")
print("Velocity BCs: No-slip on all four walls")
print("Temperature BCs: Hot wall = 1, Cold wall = 0")
print("Top and bottom walls: Insulated")
