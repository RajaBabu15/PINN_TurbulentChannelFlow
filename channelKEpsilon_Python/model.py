# %%
!pip install deepxde numpy matplotlib torch

# %%
import os
# Set the backend to PyTorch BEFORE importing deepxde
os.environ["DDE_BACKEND"] = "pytorch"

import torch
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import time

# Check and set default device
if torch.cuda.is_available():
    print("CUDA available. Setting default device to CUDA.")
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32) # Ensure consistent dtype
else:
    print("CUDA not available. Using CPU.")
    torch.set_default_device("cpu")


# Function to ensure plots directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ============================================================================
# 1. Define Physical Parameters & Geometry
# ============================================================================
# Fluid Properties and Flow Parameters (Matching OpenFOAM & Paper)
nu = 0.0002       # Kinematic viscosity (m^2/s)
rho = 1.0         # Density (kg/m^3) - Assumed 1 for kinematic pressure
U_inlet = 1.0     # Inlet velocity (m/s)
H = 2.0           # Channel height (m)
h = H / 2.0       # Channel half-height (m)
L = 10.0          # Channel length (m)
Re_H = U_inlet * H / nu  # Reynolds number based on Height (~10000)
print(f"Re_H = {Re_H:.0f}")

# Turbulence Model Constants (Standard k-epsilon)
Cmu = 0.09
Ceps1 = 1.44
Ceps2 = 1.92
sigma_k = 1.0
sigma_eps = 1.3

# Wall Function Parameters
kappa = 0.41      # von Karman constant
E = 9.8           # Log-law constant for smooth walls
y_p = 0.04        # Distance from wall to apply WF (Ideally, match FVM 1st cell y+ ~ 30-100)
                  # This value might need tuning based on FVM y+ results.

# Inlet Turbulence Conditions
I = 0.05          # Turbulence intensity
l_mix = 0.07 * h  # Mixing length scale (0.07 * h)
k_in = 1.5 * (U_inlet * I)**2
eps_in = (Cmu**0.75) * (k_in**1.5) / l_mix
print(f"Calculated Inlet k: {k_in:.6f} m^2/s^2, eps: {eps_in:.6f} m^2/s^3")

# --- Wall Function Target Value Calculation (Simplified Approach) ---
# Estimate u_tau based on expected Re_tau for Re_H=10000 (Re_tau ~ 300-400)
Re_tau_target = 350 # Target friction Reynolds number
u_tau_target = Re_tau_target * nu / h # Target friction velocity
print(f"Target u_tau for Wall Functions (based on Re_tau={Re_tau_target}): {u_tau_target:.4f} m/s")

if u_tau_target <= 0:
    raise ValueError("u_tau_target must be positive.")

# Calculate target y+ at the chosen y_p
yp_plus_target = y_p * u_tau_target / nu
print(f"\nTarget y+ at y_p={y_p:.3f} is: {yp_plus_target:.2f}")
if yp_plus_target < 20 or yp_plus_target > 100: # Wider range check
     print(f"Warning: Target y+ ({yp_plus_target:.2f}) is outside the typical log-law range (30-100).")
     print("         The chosen y_p might be too small/large or the u_tau_target estimate differs")
     print("         significantly from the actual developed flow.")

# Calculate target values at y_p based on standard wall function formulas
# Avoid log(0) or negative arguments if yp_plus_target is extremely small
if yp_plus_target * E > 1e-6: # Check before log
    U_target = (u_tau_target / kappa) * np.log(E * yp_plus_target)
else:
    U_target = 0 # Velocity should be small if y+ is very small
    print("Warning: Very low target y+, setting U_target based on log-law to 0.")

k_target = u_tau_target**2 / np.sqrt(Cmu)
# Ensure eps_target calculation avoids division by zero if y_p is zero
if y_p > 1e-9:
    eps_target = u_tau_target**3 / (kappa * y_p)
else:
    eps_target = 1e-6 # Assign a small value if y_p is effectively zero
    print("Warning: y_p is near zero, setting eps_target to a small value.")


print(f"Target values at y_p={y_p:.3f} (y+={yp_plus_target:.2f}):")
print(f"  U_target   = {U_target:.4f} m/s")
print(f"  k_target   = {k_target:.6f} m^2/s^2")
print(f"  eps_target = {eps_target:.6f} m^2/s^3")
# --- End Wall Function Target Value Calculation ---

eps_small = 1e-10 # Small number to prevent log(0) or division by zero

# Computational Domain
geom = dde.geometry.Rectangle(xmin=[0, -h], xmax=[L, h])

# ============================================================================
# 2. Define the PDE System (RANS k-epsilon)
# ============================================================================
# Ensure device consistency for tensors created inside the function
def pde(x, y):
    """
    Expresses the RANS k-epsilon equations.
    x: spatial coordinates, shape = (N, 2), where x[:, 0] = x and x[:, 1] = y
    y: network output, shape = (N, 5), where y = (u, v, p', k_raw, eps_raw)
    """
    u, v, p_prime, k_raw, eps_raw = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5]

    # Enforce positivity using exp transformation (matches network output definition)
    # Add eps_small AFTER exp for numerical stability if raw output is large negative
    k = dde.backend.exp(k_raw) + eps_small
    eps = dde.backend.exp(eps_raw) + eps_small

    # Calculate derivatives using DeepXDE's automatic differentiation
    # Ensure gradients are computed correctly on the default device
    # Gradients of velocity components
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)
    v_y = dde.grad.jacobian(y, x, i=1, j=1)

    # Gradients of pressure
    p_prime_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_prime_y = dde.grad.jacobian(y, x, i=2, j=1)

    # Gradients and Laplacians using hessian for efficiency where possible
    # Need individual gradients first for some terms
    # Velocity Hessians
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    u_xy = dde.grad.hessian(y, x, component=0, i=0, j=1) # = u_yx
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    v_xy = dde.grad.hessian(y, x, component=1, i=0, j=1) # = v_yx

    # --- Calculate nu_t and its gradients ---
    # Use k and eps directly (already positive)
    k_safe = k
    eps_safe = eps # Use the positive version

    nu_t = Cmu * dde.backend.square(k_safe) / eps_safe
    nu_eff = nu + nu_t

    # Gradients of nu_eff (needed for momentum eq diff terms)
    # Requires gradients of k and eps first
    k_x = dde.grad.jacobian(k, x, i=0, j=0) # Grad of positive k
    k_y = dde.grad.jacobian(k, x, i=0, j=1)
    eps_x = dde.grad.jacobian(eps, x, i=0, j=0) # Grad of positive eps
    eps_y = dde.grad.jacobian(eps, x, i=0, j=1)

    # Chain rule for nu_eff gradients
    # d(nu_t)/dk = 2 * Cmu * k / eps
    # d(nu_t)/deps = - Cmu * k^2 / eps^2
    dnut_dk = 2.0 * Cmu * k_safe / eps_safe
    dnut_deps = -Cmu * dde.backend.square(k_safe) / dde.backend.square(eps_safe)

    nu_eff_x = dnut_dk * k_x + dnut_deps * eps_x
    nu_eff_y = dnut_dk * k_y + dnut_deps * eps_y

    # --- Equation Definitions ---

    # Continuity Equation (Eq. 1)
    eq_continuity = u_x + v_y

    # x-Momentum Equation (Eq. 3)
    adv_u = u * u_x + v * u_y
    # Diffusion term split for clarity (matches paper structure)
    # d/dx (2 * nu_eff * du/dx) + d/dy (nu_eff * (du/dy + dv/dx))
    diff_u_term1 = nu_eff_x * 2 * u_x + nu_eff * 2 * u_xx
    diff_u_term2 = nu_eff_y * (u_y + v_x) + nu_eff * (u_yy + v_xy)
    k_gradient_term_x = (2.0 / 3.0) * k_x # Using grad of positive k
    eq_mom_x = adv_u + p_prime_x - (diff_u_term1 + diff_u_term2) + k_gradient_term_x

    # y-Momentum Equation (Eq. 4)
    adv_v = u * v_x + v * v_y
    # Diffusion term split for clarity
    # d/dy (2 * nu_eff * dv/dy) + d/dx (nu_eff * (du/dy + dv/dx))
    diff_v_term1 = nu_eff_y * 2 * v_y + nu_eff * 2 * v_yy
    diff_v_term2 = nu_eff_x * (u_y + v_x) + nu_eff * (u_xy + v_xx)
    k_gradient_term_y = (2.0 / 3.0) * k_y # Using grad of positive k
    eq_mom_y = adv_v + p_prime_y - (diff_v_term1 + diff_v_term2) + k_gradient_term_y

    # --- k-Equation (Eq. 6) ---
    adv_k = u * k_x + v * k_y
    # Diffusion term: div((nu + nu_t/sigma_k) * grad(k))
    diffusivity_k = nu + nu_t / sigma_k
    # Need gradient of diffusivity_k
    # Chain rule: d(diff_k)/dx = d(nu_t/sigma_k)/dk * kx + d(nu_t/sigma_k)/deps * epsx
    diffusivity_k_x = (dnut_dk / sigma_k) * k_x + (dnut_deps / sigma_k) * eps_x
    diffusivity_k_y = (dnut_dk / sigma_k) * k_y + (dnut_deps / sigma_k) * eps_y
    # Need Laplacian of k (use hessian)
    k_xx = dde.grad.hessian(k, x, i=0, j=0) # Hessian of positive k
    k_yy = dde.grad.hessian(k, x, i=1, j=1)
    laplacian_k = k_xx + k_yy
    diffusion_k = diffusivity_k_x * k_x + diffusivity_k_y * k_y + diffusivity_k * laplacian_k

    # Production Term P_k (Eq. 8)
    S_squared = 2 * (dde.backend.square(u_x) + dde.backend.square(v_y)) + dde.backend.square(u_y + v_x)
    P_k = nu_t * S_squared

    eq_k = adv_k - diffusion_k - P_k + eps # Using positive eps

    # --- Epsilon-Equation (Eq. 7) ---
    adv_eps = u * eps_x + v * eps_y
    # Diffusion term: div((nu + nu_t/sigma_eps) * grad(eps))
    diffusivity_eps = nu + nu_t / sigma_eps
    # Need gradient of diffusivity_eps
    diffusivity_eps_x = (dnut_dk / sigma_eps) * k_x + (dnut_deps / sigma_eps) * eps_x
    diffusivity_eps_y = (dnut_dk / sigma_eps) * k_y + (dnut_deps / sigma_eps) * eps_y
    # Need Laplacian of eps
    eps_xx = dde.grad.hessian(eps, x, i=0, j=0) # Hessian of positive eps
    eps_yy = dde.grad.hessian(eps, x, i=1, j=1)
    laplacian_eps = eps_xx + eps_yy
    diffusion_eps = diffusivity_eps_x * eps_x + diffusivity_eps_y * eps_y + diffusivity_eps * laplacian_eps

    # Source and Sink terms for Epsilon
    source_eps = Ceps1 * (eps_safe / k_safe) * P_k
    sink_eps = Ceps2 * (dde.backend.square(eps_safe) / k_safe)
    eq_eps = adv_eps - diffusion_eps - source_eps + sink_eps

    return [eq_continuity, eq_mom_x, eq_mom_y, eq_k, eq_eps]


# ============================================================================
# 3. Define Boundary Conditions
# ============================================================================
# Helper functions for boundary identification
def boundary_inlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_outlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], L)

def boundary_bottom_wall_physical(x, on_boundary):
    return on_boundary and np.isclose(x[1], -h)

def boundary_top_wall_physical(x, on_boundary):
    return on_boundary and np.isclose(x[1], h)

# Physical Boundary Conditions (Applied at actual boundaries)
# Inlet
bc_u_inlet = dde.DirichletBC(geom, lambda x: U_inlet, boundary_inlet, component=0)
bc_v_inlet = dde.DirichletBC(geom, lambda x: 0, boundary_inlet, component=1)
# Transform target k/eps to raw values using log (inverse of network output activation)
k_in_transformed = np.log(max(k_in, eps_small))
eps_in_transformed = np.log(max(eps_in, eps_small))
bc_k_inlet = dde.DirichletBC(geom, lambda x: k_in_transformed, boundary_inlet, component=3)
bc_eps_inlet = dde.DirichletBC(geom, lambda x: eps_in_transformed, boundary_inlet, component=4)

# Outlet
bc_p_outlet = dde.DirichletBC(geom, lambda x: 0, boundary_outlet, component=2) # Set reference p'=0
# Zero gradient for U, k, eps at outlet is implicitly handled by minimizing PDE residual

# Walls (Physical No-Slip)
bc_u_walls = dde.DirichletBC(geom, lambda x: 0,
                             lambda x, on_boundary: boundary_bottom_wall_physical(x, on_boundary) or boundary_top_wall_physical(x, on_boundary),
                             component=0)
bc_v_walls = dde.DirichletBC(geom, lambda x: 0,
                             lambda x, on_boundary: boundary_bottom_wall_physical(x, on_boundary) or boundary_top_wall_physical(x, on_boundary),
                             component=1)
# Zero gradient for p' at walls is implicitly handled by minimizing PDE residual + continuity

# Wall Function "Boundary" Conditions (Applied at y_p distance)
n_wf_points_per_wall = 200 # Number of points along each wall for WF BCs
x_wf_coords = np.linspace(0 + L * 0.01, L * 0.99, n_wf_points_per_wall)[:, None] # Avoid corners
points_bottom_wf = np.hstack((x_wf_coords, np.full_like(x_wf_coords, -h + y_p)))
points_top_wf = np.hstack((x_wf_coords, np.full_like(x_wf_coords, h - y_p)))
points_wf = np.vstack((points_bottom_wf, points_top_wf))

# Target values for the PointSetBC (using pre-calculated targets)
U_target_vals = np.full((points_wf.shape[0], 1), U_target)
k_target_transformed = np.log(max(k_target, eps_small))
eps_target_transformed = np.log(max(eps_target, eps_small))
k_target_vals = np.full((points_wf.shape[0], 1), k_target_transformed)
eps_target_vals = np.full((points_wf.shape[0], 1), eps_target_transformed)

# Define PointSetBCs for U, k_raw, eps_raw at the wall function interface points
bc_u_wf = dde.PointSetBC(points_wf, U_target_vals, component=0)
bc_k_wf = dde.PointSetBC(points_wf, k_target_vals, component=3) # BC for k_raw
bc_eps_wf = dde.PointSetBC(points_wf, eps_target_vals, component=4) # BC for eps_raw

# Collect all boundary conditions
bcs = [
    # Inlet
    bc_u_inlet, bc_v_inlet, bc_k_inlet, bc_eps_inlet,
    # Outlet
    bc_p_outlet,
    # Physical Walls (No-slip)
    bc_u_walls, bc_v_walls,
    # Wall Function Interface (Simplified Dirichlet)
    bc_u_wf, bc_k_wf, bc_eps_wf
]

# ============================================================================
# 4. Define the Neural Network and Model
# ============================================================================
n_layers = 8
n_neurons = 64
activation = "tanh"  # Tanh generally works well
initializer = "Glorot normal" # Standard initializer

net = dde.maps.FNN(
    layer_sizes=[2] + [n_neurons] * n_layers + [5], # Input (x,y), Hidden layers, Output (u,v,p',k_raw,eps_raw)
    activation=activation,
    kernel_initializer=initializer
)

# Define the model data structure
data = dde.data.PDE(
    geometry=geom,
    pde=pde,
    bcs=bcs,
    num_domain=20000,    # Number of collocation points inside domain
    num_boundary=4000,   # Number of points on physical boundaries (inlet, outlet, walls)
    num_test=5000,       # Number of points for testing PDE residual after training
    anchors=points_wf    # Ensure collocation points are placed exactly at WF locations
)

model = dde.Model(data, net)

# ============================================================================
# 5. Train the Model
# ============================================================================
# Loss weights - IMPORTANT: These may require significant tuning!
# Order: [PDE_cont, PDE_mom_x, PDE_mom_y, PDE_k, PDE_eps, BCs...]
# Give higher weight to BCs initially?
pde_weights = [1, 1, 1, 1, 1] # Base weights for PDE residuals
# Weights for each BC in the 'bcs' list order:
bc_weights = [
    10, 10, 10, 10, # Inlet U, V, k, eps
    10,             # Outlet p'
    10, 10,         # Wall U, V (physical)
    10, 10, 10      # WF U, k, eps (simplified) - Give these decent weight
]
loss_weights = pde_weights + bc_weights

# Compile with Adam optimizer
learning_rate_adam = 1e-3
model.compile("adam", lr=learning_rate_adam, loss_weights=loss_weights)

# Checkpoint setup
model_dir = "model_rans_channel_wf"
ensure_dir(model_dir)
checkpointer_path = os.path.join(model_dir, "rans_channel_wf.ckpt")
checker = dde.callbacks.ModelCheckpoint(checkpointer_path, save_better_only=True, period=1000, verbose=1) # Save less frequently for long runs

# Adam Training
print("Starting Adam training...")
adam_iterations = 100000 # Increased iterations
start_time = time.time()
losshistory, train_state = model.train(iterations=adam_iterations, display_every=5000, callbacks=[checker])
adam_time = time.time() - start_time
print(f"Adam training ({adam_iterations} iterations) completed in {adam_time:.2f} seconds.")

# Compile with L-BFGS optimizer for fine-tuning
print("\nStarting L-BFGS training...")
model.compile("L-BFGS", loss_weights=loss_weights) # Re-using weights, could adjust
start_time = time.time()
lbfgs_iterations = 50000 # Increased iterations
# Note: L-BFGS doesn't use display_every in the same way as Adam. It runs until convergence or max iterations.
# The checker callback will still save periodically based on its internal step count.
losshistory, train_state = model.train(iterations=lbfgs_iterations, callbacks=[checker]) # Remove display_every for L-BFGS standard behavior
lbfgs_time = time.time() - start_time
print(f"L-BFGS training (max {lbfgs_iterations} iterations) completed in {lbfgs_time:.2f} seconds.")

# Save loss history plot
plots_dir = "plots"
ensure_dir(plots_dir)
dde.saveplot(losshistory, train_state, issave=True, isplot=True, output_dir=plots_dir)
print(f"Loss history plot saved in '{plots_dir}'.")


# ============================================================================
# 6. Post-processing and Visualization
# ============================================================================
print("\nStarting post-processing...")

# Define prediction grid (can reduce nx/ny if 2000x2000 is too slow/memory intensive)
nx_pred, ny_pred = 200, 100 # Reduced grid for faster post-processing
x_coords = np.linspace(0, L, nx_pred)
y_coords = np.linspace(-h, h, ny_pred)
X, Y = np.meshgrid(x_coords, y_coords)
pred_points = np.vstack((np.ravel(X), np.ravel(Y))).T

# Predict on the grid
predictions_raw = model.predict(pred_points)

# Extract and reshape variables, applying inverse transform for k, eps
u_pred = predictions_raw[:, 0].reshape(ny_pred, nx_pred)
v_pred = predictions_raw[:, 1].reshape(ny_pred, nx_pred)
p_prime_pred = predictions_raw[:, 2].reshape(ny_pred, nx_pred)
k_raw_pred = predictions_raw[:, 3].reshape(ny_pred, nx_pred)
eps_raw_pred = predictions_raw[:, 4].reshape(ny_pred, nx_pred)

# Apply exp to get positive k and eps
k_pred = np.exp(k_raw_pred) + eps_small
eps_pred = np.exp(eps_raw_pred) + eps_small

# Calculate eddy viscosity
nu_t_pred = Cmu * np.square(k_pred) / (eps_pred) # eps_pred is already positive

# --- Generate Contour Plots ---
print("Generating contour plots...")
plt.figure(figsize=(15, 10)) # Adjusted figure size

plt.subplot(2, 3, 1)
cf1 = plt.contourf(X, Y, u_pred, levels=50, cmap='viridis')
plt.colorbar(cf1, label='u (m/s)')
plt.title('Streamwise Velocity (u)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')

plt.subplot(2, 3, 2)
cf2 = plt.contourf(X, Y, v_pred, levels=50, cmap='viridis')
plt.colorbar(cf2, label='v (m/s)')
plt.title('Transverse Velocity (v)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')

plt.subplot(2, 3, 3)
cf3 = plt.contourf(X, Y, p_prime_pred, levels=50, cmap='viridis')
plt.colorbar(cf3, label='p/rho (m^2/s^2)')
plt.title('Kinematic Pressure (p\')')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')

plt.subplot(2, 3, 4)
# Use log scale for k if values vary widely, check for non-positive values first
k_plot = np.log10(np.maximum(k_pred, eps_small)) # Avoid log10(0)
cf4 = plt.contourf(X, Y, k_plot, levels=50, cmap='viridis')
plt.colorbar(cf4, label='log10(k)')
plt.title('Turbulent Kinetic Energy (log scale)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')

plt.subplot(2, 3, 5)
# Use log scale for epsilon
eps_plot = np.log10(np.maximum(eps_pred, eps_small))
cf5 = plt.contourf(X, Y, eps_plot, levels=50, cmap='viridis')
plt.colorbar(cf5, label='log10(epsilon)')
plt.title('Dissipation Rate (log scale)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')

plt.subplot(2, 3, 6)
cf6 = plt.contourf(X, Y, nu_t_pred / nu, levels=50, cmap='viridis')
plt.colorbar(cf6, label='nu_t / nu')
plt.title('Eddy Viscosity Ratio')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "field_contours.png"))
plt.close()
print("Contour plots saved.")

# --- Load OpenFOAM Data (Placeholder) ---
# !! Replace this section with actual code to load your OpenFOAM results !!
# Example: Assuming data sampled at x=L/2 is saved in a file
fvm_data_file = 'postProcessing/samplingSets/1000/midLine_y_U_p_k_epsilon_nut.xy' # Example path
has_fvm_data = False
try:
    # Example using pandas (install pandas if needed: pip install pandas)
    # import pandas as pd
    # Adjust skiprows, delimiter based on your OpenFOAM raw format output
    # fvm_df = pd.read_csv(fvm_data_file, skiprows=2, delim_whitespace=True, header=None)
    # Column mapping might be: 0:y, 1:Ux, 2:Uy, 3:Uz, 4:p', 5:k, 6:eps, 7:nut
    # y_fvm = fvm_df[0].values
    # u_fvm = fvm_df[1].values
    # k_fvm = fvm_df[5].values
    # eps_fvm = fvm_df[6].values
    # nut_fvm = fvm_df[7].values

    # --- Using Numpy loadtxt (alternative) ---
    # Adjust skiprows based on header lines in the file
    fvm_raw_data = np.loadtxt(fvm_data_file, skiprows=2) # Might need comments='%'
    # Check column order carefully based on your OpenFOAM sampleDict/controlDict Fields entry
    y_fvm = fvm_raw_data[:, 0] # Assuming y is the first column
    u_fvm = fvm_raw_data[:, 1] # Assuming U_x is the second
    # v_fvm = fvm_raw_data[:, 2] # Assuming U_y is the third (usually small)
    # p_fvm = fvm_raw_data[:, 4] # Assuming p is the fifth (check if kinematic)
    k_fvm = fvm_raw_data[:, 5] # Assuming k is the sixth
    eps_fvm = fvm_raw_data[:, 6] # Assuming epsilon is the seventh
    nut_fvm = fvm_raw_data[:, 7] # Assuming nut is the eighth
    has_fvm_data = True
    print(f"Successfully loaded FVM data from '{fvm_data_file}'.")

    # Estimate u_tau from FVM wall shear stress (if available) or from profile
    # Placeholder value - REPLACE with actual FVM u_tau
    u_tau_fvm_estimated = 0.07 # Should ideally come from FVM post-processing (e.g., yPlusRAS)

except FileNotFoundError:
    print(f"Warning: FVM data file not found at '{fvm_data_file}'. Skipping FVM comparison.")
    # Create dummy arrays so plotting code doesn't crash
    y_fvm, u_fvm, k_fvm, eps_fvm, nut_fvm = [], [], [], [], []
    u_tau_fvm_estimated = u_tau_target # Use target as fallback
except Exception as e:
    print(f"Warning: Error loading FVM data: {e}. Skipping FVM comparison.")
    y_fvm, u_fvm, k_fvm, eps_fvm, nut_fvm = [], [], [], [], []
    u_tau_fvm_estimated = u_tau_target

# --- Generate Profile Plots (Comparison at x=L/2) ---
print("Generating profile comparison plots...")
x_slice_loc = L / 2
x_slice_idx = np.argmin(np.abs(x_coords - x_slice_loc))
y_slice = y_coords

# Extract PINN slices
u_slice_pinn = u_pred[:, x_slice_idx]
k_slice_pinn = k_pred[:, x_slice_idx]
eps_slice_pinn = eps_pred[:, x_slice_idx]
nu_t_slice_pinn = nu_t_pred[:, x_slice_idx]

# Plot Linear Profiles (u, k, eps, nu_t/nu vs y) - Matches Figs 3b, 4a, 4b, 4c (shape)
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(u_slice_pinn, y_slice, 'r-', linewidth=2, label='PINN u')
if has_fvm_data: plt.plot(u_fvm, y_fvm, 'b--', linewidth=1.5, label='FVM u')
plt.xlabel('u (m/s)')
plt.ylabel('y/h') # Normalized y
plt.title(f'Velocity Profile at x={x_slice_loc:.1f} m')
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.legend()
plt.grid(True, linestyle=':')

plt.subplot(2, 2, 2)
plt.plot(k_slice_pinn, y_slice, 'r-', linewidth=2, label='PINN k')
if has_fvm_data: plt.plot(k_fvm, y_fvm, 'b--', linewidth=1.5, label='FVM k')
plt.xlabel('k (m^2/s^2)')
plt.ylabel('y/h')
plt.title(f'TKE Profile at x={x_slice_loc:.1f} m')
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.legend()
plt.grid(True, linestyle=':')

plt.subplot(2, 2, 3)
plt.plot(eps_slice_pinn, y_slice, 'r-', linewidth=2, label='PINN epsilon')
if has_fvm_data: plt.plot(eps_fvm, y_fvm, 'b--', linewidth=1.5, label='FVM epsilon')
plt.xlabel('epsilon (m^2/s^3)')
plt.ylabel('y/h')
plt.title(f'Dissipation Profile at x={x_slice_loc:.1f} m')
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.legend()
plt.grid(True, linestyle=':')
# Use log scale for x-axis if epsilon varies over orders of magnitude
# Check if data is positive before applying log scale
if np.all(eps_slice_pinn > 0) and (not has_fvm_data or np.all(eps_fvm > 0)):
     plt.semilogx()

plt.subplot(2, 2, 4)
plt.plot(nu_t_slice_pinn / nu, y_slice, 'r-', linewidth=2, label='PINN nu_t/nu')
if has_fvm_data: plt.plot(nut_fvm / nu, y_fvm, 'b--', linewidth=1.5, label='FVM nu_t/nu')
plt.xlabel('nu_t / nu')
plt.ylabel('y/h')
plt.title(f'Eddy Viscosity Ratio at x={x_slice_loc:.1f} m')
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.legend()
plt.grid(True, linestyle=':')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "profile_comparison_linear_y.png")) # Generic name
plt.close()

# --- Calculate u_tau from PINN results near the wall at x=L/2 ---
# Evaluate gradients near the top wall (y=h)
y_eval_1 = h - y_p * 1.1 # Point closer to wall than y_p
y_eval_2 = h - y_p * 0.9 # Point further from wall than y_p (closer to center)
eval_points_1 = np.array([[x_slice_loc, y_eval_1]])
eval_points_2 = np.array([[x_slice_loc, y_eval_2]])

# Predict U and nu_t at these points
pred_1_raw = model.predict(eval_points_1)[0]
pred_2_raw = model.predict(eval_points_2)[0]

u1 = pred_1_raw[0]
u2 = pred_2_raw[0]
k1_raw, eps1_raw = pred_1_raw[3], pred_1_raw[4]
k2_raw, eps2_raw = pred_2_raw[3], pred_2_raw[4]

k1 = np.exp(k1_raw) + eps_small
eps1 = np.exp(eps1_raw) + eps_small
nu_t1 = Cmu * k1**2 / eps1

k2 = np.exp(k2_raw) + eps_small
eps2 = np.exp(eps2_raw) + eps_small
nu_t2 = Cmu * k2**2 / eps2

# Average nu_t and calculate gradient
nu_t_eval = (nu_t1 + nu_t2) / 2.0
nu_eff_eval = nu + nu_t_eval
du_dy_eval = (u2 - u1) / (y_eval_2 - y_eval_1) # Gradient dy positive towards center

# Calculate wall shear stress and friction velocity
tau_w_pinn = rho * nu_eff_eval * abs(du_dy_eval) # Use abs value of gradient at wall
u_tau_pinn_estimated = np.sqrt(tau_w_pinn / rho)
print(f"\nEstimated PINN u_tau at x={x_slice_loc:.1f} m: {u_tau_pinn_estimated:.4f} m/s")
print(f"Estimated FVM u_tau at x={x_slice_loc:.1f} m: {u_tau_fvm_estimated:.4f} m/s (placeholder)")


# --- Generate Wall Unit Plots (U+, k+, eps+ vs y+) - Matches Figs 3a, 4a, 4b ---
print("Generating wall unit plots...")
plt.figure(figsize=(18, 5))

# Select data near the top wall (y > 0)
wall_indices = y_slice > 0
y_wall = y_slice[wall_indices]
u_wall_pinn = u_slice_pinn[wall_indices]
k_wall_pinn = k_slice_pinn[wall_indices]
eps_wall_pinn = eps_slice_pinn[wall_indices]

# Calculate distance from top wall
y_dist_wall = h - y_wall

# Calculate wall units for PINN
y_plus_pinn = y_dist_wall * u_tau_pinn_estimated / nu
u_plus_pinn = u_wall_pinn / u_tau_pinn_estimated
k_plus_pinn = k_wall_pinn / u_tau_pinn_estimated**2
eps_plus_pinn = eps_wall_pinn * nu / u_tau_pinn_estimated**4

# Calculate wall units for FVM (if data exists)
if has_fvm_data:
    fvm_wall_indices = y_fvm > 0
    y_dist_fvm = h - y_fvm[fvm_wall_indices]
    u_wall_fvm = u_fvm[fvm_wall_indices]
    k_wall_fvm = k_fvm[fvm_wall_indices]
    eps_wall_fvm = eps_fvm[fvm_wall_indices]

    y_plus_fvm = y_dist_fvm * u_tau_fvm_estimated / nu
    u_plus_fvm = u_wall_fvm / u_tau_fvm_estimated
    k_plus_fvm = k_wall_fvm / u_tau_fvm_estimated**2
    eps_plus_fvm = eps_wall_fvm * nu / u_tau_fvm_estimated**4

# Plot U+ vs y+ (Log-Log or Semi-log) - Fig 3a
plt.subplot(1, 3, 1)
plt.semilogx(y_plus_pinn, u_plus_pinn, 'r-', linewidth=2, label='PINN')
if has_fvm_data: plt.semilogx(y_plus_fvm, u_plus_fvm, 'b--', linewidth=1.5, label='FVM')
# Add theoretical log-law line
y_plus_theory = np.logspace(np.log10(max(1, np.min(y_plus_pinn))), np.log10(np.max(y_plus_pinn)), 100)
u_plus_loglaw = (1 / kappa) * np.log(y_plus_theory) + E # Note: E used here, not E/kappa
plt.semilogx(y_plus_theory, u_plus_loglaw, 'k:', label='Log-Law (k=0.41, E=9.8)')
plt.xlabel('$y^+$')
plt.ylabel('$U^+$')
plt.title('$U^+$ vs $y^+$ at $x=L/2$')
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.ylim(bottom=0)
plt.xlim(left=1) # Start from y+=1 for log scale


# Plot k+ vs y+ - Fig 4a (Wall units)
plt.subplot(1, 3, 2)
plt.semilogx(y_plus_pinn, k_plus_pinn, 'r-', linewidth=2, label='PINN')
if has_fvm_data: plt.semilogx(y_plus_fvm, k_plus_fvm, 'b--', linewidth=1.5, label='FVM')
# Add target line k+ = 1/sqrt(Cmu)
plt.axhline(1/np.sqrt(Cmu), color='k', linestyle=':', label=f'$k^+=1/\\sqrt{{C_\\mu}} \\approx {1/np.sqrt(Cmu):.2f}$')
# Mark the location y_p+
plt.axvline(yp_plus_target, color='g', linestyle='-.', label=f'$y_p^+ \\approx {yp_plus_target:.1f}$')
plt.xlabel('$y^+$')
plt.ylabel('$k^+$')
plt.title('$k^+$ vs $y^+$ at $x=L/2$')
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.ylim(bottom=0)
plt.xlim(left=1)

# Plot epsilon+ vs y+ - Fig 4b (Wall units)
plt.subplot(1, 3, 3)
plt.semilogx(y_plus_pinn, eps_plus_pinn, 'r-', linewidth=2, label='PINN')
if has_fvm_data: plt.semilogx(y_plus_fvm, eps_plus_fvm, 'b--', linewidth=1.5, label='FVM')
# Add target line eps+ = 1 / (kappa * y+)
eps_plus_target_theory = 1 / (kappa * y_plus_theory)
plt.semilogx(y_plus_theory, eps_plus_target_theory, 'k:', label='$\\epsilon^+ = 1/(\\kappa y^+)$')
# Mark the location y_p+
plt.axvline(yp_plus_target, color='g', linestyle='-.', label=f'$y_p^+ \\approx {yp_plus_target:.1f}$')
plt.xlabel('$y^+$')
plt.ylabel('$\\epsilon^+$')
plt.title('$\\epsilon^+$ vs $y^+$ at $x=L/2$')
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.ylim(bottom=0)
plt.xlim(left=1)


plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "profile_comparison_wall_units.png"))
plt.close()
print("Wall unit profile plots saved.")


# --- Generate Pressure Gradient Plot - Fig 3c ---
print("Generating pressure gradient plot...")
# Extract centerline pressure
center_idx = ny_pred // 2
p_prime_centerline_pinn = p_prime_pred[center_idx, :]

# Calculate gradient using numpy.gradient
dp_dx_pinn, = np.gradient(p_prime_centerline_pinn, x_coords)

# Placeholder for FVM pressure gradient (replace with actual calculation)
if has_fvm_data:
    # Need FVM pressure sampled along centerline
    # Example: Assume p_fvm_centerline exists
    # dp_dx_fvm, = np.gradient(p_fvm_centerline, x_coords_fvm) # Need FVM x coords
    dp_dx_fvm = np.full_like(dp_dx_pinn, np.mean(dp_dx_pinn)) # Dummy data
    pass
else:
    dp_dx_fvm = []

plt.figure(figsize=(8, 5))
plt.plot(x_coords, dp_dx_pinn, 'r-', linewidth=2, label='PINN')
if has_fvm_data: plt.plot(x_coords, dp_dx_fvm, 'b--', linewidth=1.5, label='FVM') # Use matching x_coords if possible
plt.xlabel('x (m)')
plt.ylabel('$dp\'/dx$')
plt.title('Streamwise Kinematic Pressure Gradient along Centerline')
plt.legend()
plt.grid(True, linestyle=':')
plt.ylim(1.2 * np.min(dp_dx_pinn), 0.8 * np.max(dp_dx_pinn)) # Adjust ylim based on data

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "pressure_gradient.png"))
plt.close()
print("Pressure gradient plot saved.")


print("\nPost-processing complete.")


