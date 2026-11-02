
from dolfinx import fem
from petsc4py import PETSc
import ufl

from step1_mesh_spaces import domain, W, V_T

# ==================================================
# STEP 3A: Define solution and test functions
# ==================================================

# Mixed solution function
w = fem.Function(W)
# w = the function solution within the mixed function space W

# Split velocity and pressure
u, p = ufl.split(w)
# u = dimentionless velocity vector U = (U,V)
# p= = dimentionless pressure P

# Momentum test functions
v, q = ufl.TestFunctions(W)
# v = momentum vector test fucntion W = (wx, wy)
# q = scalar test function for continuity
# v is used as the Python variable name for the momentum test function, while the physical vertical velocity component is still V.

# Temperature solution and test functions
theta = fem.Function(V_T)
phi = ufl.TestFunction(V_T)

# ==================================================
# STEP 3B: Define dimensionless parameters
# ==================================================

# Temporary values (to be replaced with calculated values)
Ra = fem.Constant(domain, PETSc.ScalarType(1.0))
Pr = fem.Constant(domain, PETSc.ScalarType(1.0))

# Integration measure
dx = ufl.Measure("dx", domain=domain)

# ==================================================
# STEP 3C: Continuity weak form
# ==================================================

F_c = q * ufl.div(u) * dx

# ==================================================
# STEP 3D: Combined momentum weak form
# ==================================================

F_m = (
    ufl.inner(ufl.dot(ufl.grad(u), u), v)
    + Pr * ufl.inner(ufl.grad(u), ufl.grad(v))
    - p * ufl.div(v)
    - Ra * Pr * theta * v[1]
) * dx

# Combined flow residual
F_flow = F_c + F_m

# ==================================================
# STEP 3E: Energy weak form
# ==================================================

F_T = (
    phi * ufl.dot(u, ufl.grad(theta))
    + ufl.dot(ufl.grad(phi), ufl.grad(theta))
) * dx

print("Step 3: Dimensionless weak forms defined successfully!")
