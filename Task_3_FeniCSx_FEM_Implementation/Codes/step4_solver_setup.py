import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
from dolfinx import la
import numpy as np

from step1_mesh_spaces import domain, W, V_T

from step2b_apply_bcs import (
    bcs_velocity,
    bcs_temperature
)

from step3_weak_forms import (
    w,
    u,
    p,
    theta,
    phi,
    F_flow,
    Ra,
    Pr,
    dx
)

# ==================================================
# STEP 4A: Initialise solution fields
# ==================================================

# Initialise velocity and pressure to zero
w.x.array[:] = 0.0
w.x.scatter_forward()

# Initialise temperature with a linear profile
def initial_temperature(x):
    return 1.0 - x[0]

theta.interpolate(initial_temperature)
theta.x.scatter_forward()

# ==================================================
# STEP 4B: Verify initialisation
# ==================================================

print("Step 4A: Initial solution fields created successfully!")
print("Initial velocity: U = 0, V = 0")
print("Initial pressure: P = 0")
print("Initial temperature: theta = 1 - X")

# Create the nonlinear flow problem (STEP 4B THAT PRODUCE OUTPUT 4B)
flow_problem = petsc.NonlinearProblem(
    F_flow,
    w,
    bcs=bcs_velocity,
    petsc_options_prefix="flow_",
    petsc_options={
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-10,
        "snes_max_it": 50,
    }
)

print("Nonlinear flow problem created successfully.")
print("SNES solver type:", flow_problem.solver.getType())

# ==================================================
# Step 4C: Construct the pressure nullspace
# ==================================================

# Create a function in the mixed velocity-pressure space
nullspace_vector = fem.Function(W)

# Set all DOFs to zero initially
nullspace_vector.x.array[:] = 0.0

# Set all pressure DOFs to one
nullspace_vector.sub(1).interpolate(
    lambda x: np.ones(x.shape[1], dtype=PETSc.ScalarType)
)

# Synchronise ghost values
nullspace_vector.x.scatter_forward()

# Calculate the norm of the vector
vector_norm = la.norm(nullspace_vector.x)

# Normalise the vector
nullspace_vector.x.array[:] /= vector_norm

# Synchronise ghost values again
nullspace_vector.x.scatter_forward()

# Create PETSc nullspace object
nullspace = PETSc.NullSpace().create(
    vectors=[nullspace_vector.x.petsc_vec]
)

print("Step 4C: Pressure nullspace created successfully.")

# Step 4D: Attach pressure nullspace to the flow Jacobian

flow_problem.A.setNullSpace(nullspace)

print("Step 4D: Pressure nullspace attached successfully.")

# Step 4E: Create the temperature linear problem (We are using a segregated solution approach, so when we solve the temperature equation, we temporarily treat the velocity field U from the energy equation as known. Therefore, the only unknown in this equation is the temperature (theta), making it a linear problem.

# Define trial and test functions
theta_trial = ufl.TrialFunction(V_T)
phi_T = ufl.TestFunction(V_T) #We use phi_T instead of reusing phi from Step 3 to make it clear that these functions belong to the temperature solver.

# Define the bilinear form (the part at the left hand side of the weak form energy equation is the bilinear form)
a_T = (
    phi_T * ufl.dot(u, ufl.grad(theta_trial))
    + ufl.dot(ufl.grad(phi_T), ufl.grad(theta_trial))
) * dx

# Define the linear form (the rhs part of the weak form energy equation is the linear form which is =0)
L_T = fem.Constant(
    domain, PETSc.ScalarType(0.0)
) * phi_T * dx

# Create the linear temperature problem
temperature_problem = petsc.LinearProblem(
    a_T,
    L_T,
    bcs=bcs_temperature,
    u=theta,
    petsc_options_prefix="temperature_",
    petsc_options={
    "ksp_type": "gmres",
    "ksp_rtol": 1.0e-8,
    "ksp_atol": 1.0e-10,
    "ksp_max_it": 1000,
    "pc_type": "jacobi",
    }
)

print("Step 4E: Linear temperature problem created successfully.")

# Step 4G: Coupled iteration settings

max_iterations = 100
convergence_tolerance = 1.0e-6

print("Step 4G: Coupled iteration settings configured.")

# Step 4G: Coupled iteration loop

print("\nStarting coupled iteration...")

for iteration in range(max_iterations):

    # Store previous solutions
    w_old = w.x.array.copy()
    theta_old = theta.x.array.copy()

    # Step 1: Solve flow equations
    flow_problem.solve()
    w.x.scatter_forward()

    # Check flow solver convergence
    flow_reason = flow_problem.solver.getConvergedReason()

    if flow_reason <= 0:
        print("\nFlow solver failed to converge.")
        print("SNES convergence reason:", flow_reason)
        print(
            "Number of SNES iterations:",
            flow_problem.solver.getIterationNumber()
        )
        print(
            "Final residual norm:",
            flow_problem.solver.getFunctionNorm()
        )

        raise RuntimeError("Flow solver did not converge.")

    # Step 2: Solve temperature equation
    temperature_problem.solve()
    theta.x.scatter_forward()

    # Check temperature solver convergence
    temperature_reason = (
        temperature_problem.solver.getConvergedReason()
    )

    if temperature_reason <= 0:
        raise RuntimeError("Temperature solver did not converge.")

    # Step 3: Calculate relative changes
    flow_error = (
        np.linalg.norm(w.x.array - w_old)
        / max(np.linalg.norm(w.x.array), 1.0e-14)
    )

    temperature_error = (
        np.linalg.norm(theta.x.array - theta_old)
        / max(np.linalg.norm(theta.x.array), 1.0e-14)
    )

    # Display iteration information
    print(
        f"Iteration {iteration + 1}: "
        f"Flow error = {flow_error:.6e}, "
        f"Temperature error = {temperature_error:.6e}"
    )

    # Step 4: Check coupled convergence
    if (
        flow_error < convergence_tolerance
        and temperature_error < convergence_tolerance
    ):
        print("\nCoupled solution converged successfully!")
        break

else:
    print("\nWarning: Maximum iterations reached without convergence.")
