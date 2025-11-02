import pandas as pd
import numpy as np
import os
import moviepy as mp
import shutil
import tqdm
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

data = pd.read_csv('Output/Neirenberg/Sim Neirenberg.csv', header=0)

data['$\\sigma$'].unique()
# data['$\\gamma$ (PWI)'].unique()
# data['c or $r_{s}$'].unique()

mask_data = data[data['$\\sigma$'] == 279.495]
# mask_data = data[data['$\\gamma$ (PWI)'] == 2.30808]
# mask_data = data[data['c or $r_{s}$'] == 89.49495]


def nelder_mead_glafic_with_history(func, x0, y0, ftol=1e-4, nmax=10000, verbose=False):
    """
    Nelder-Mead simplex optimization matching glafic's implementation,
    with tracking of all evaluation points.
    """
    # Simplex coefficients from glafic
    ALPHA = 1.0  # reflection coefficient
    BETA = 0.5   # contraction coefficient
    GAMMA = 2.0  # expansion coefficient

    n = 2  # number of dimensions

    # Initialize simplex: 3 vertices for 2D problem
    v = np.zeros((n + 1, n))
    f = np.zeros(n + 1)

    # Track all evaluations
    eval_history = []

    # Set initial simplex
    v[0] = [x0, y0]
    dx = 0.1 * abs(x0) if x0 != 0 else 0.1
    dy = 0.1 * abs(y0) if y0 != 0 else 0.1
    v[1] = [x0 + dx, y0]
    v[2] = [x0, y0 + dy]

    # Evaluate function at initial vertices and record
    for i in range(n + 1):
        f[i] = func(v[i][0], v[i][1])
        eval_history.append((v[i][0], v[i][1], f[i]))

    # Main optimization loop
    for itr in range(1, nmax + 1):
        # Handle cases where points are outside interpolation bounds (returning inf)
        f_finite_indices = np.where(np.isfinite(f))[0]
        if len(f_finite_indices) < 3:
            # Not enough valid points to form a simplex, try to shrink
            vs = np.argmin(f) # find the best point we have
            for i in range(n + 1):
                if i != vs:
                    v[i] = v[vs] + (v[i] - v[vs]) / 2.0
                    f[i] = func(v[i][0], v[i][1])
                    eval_history.append((v[i][0], v[i][1], f[i]))
            continue

        # Find indices of best, worst, and second-worst vertices among finite values
        f_temp_finite = f[f_finite_indices]
        vg_idx_finite = np.argmax(f_temp_finite)
        vs_idx_finite = np.argmin(f_temp_finite)
        
        vg = f_finite_indices[vg_idx_finite] # Index in the original f array
        vs = f_finite_indices[vs_idx_finite] # Index in the original f array

        f_temp = f.copy()
        f_temp[vg] = -np.inf
        vh = np.argmax(f_temp)

        # Calculate centroid of all points except worst
        vm = np.mean(v[[i for i in range(n + 1) if i != vg]], axis=0)

        # Reflection
        vr = vm + ALPHA * (vm - v[vg])
        fr = func(vr[0], vr[1])
        eval_history.append((vr[0], vr[1], fr))

        if np.isnan(fr): fr = np.inf

        if f[vs] <= fr < f[vh]:
            # Accept reflection
            v[vg], f[vg] = vr, fr
        elif fr < f[vs]:
            # Try expansion
            ve = vm + GAMMA * (vr - vm)
            fe = func(ve[0], ve[1])
            eval_history.append((ve[0], ve[1], fe))
            if np.isnan(fe): fe = np.inf
            
            if fe < fr:
                v[vg], f[vg] = ve, fe
            else:
                v[vg], f[vg] = vr, fr
        else:
            # Contraction
            if fr < f[vg]:
                vc = vm + BETA * (vr - vm) # Outside
            else:
                vc = vm - BETA * (vm - v[vg]) # Inside
            
            fc = func(vc[0], vc[1])
            eval_history.append((vc[0], vc[1], fc))
            if np.isnan(fc): fc = np.inf

            if fc < f[vg]:
                v[vg], f[vg] = vc, fc
            else:
                # Shrink toward best vertex
                for i in range(n + 1):
                    if i != vs:
                        v[i] = v[vs] + (v[i] - v[vs]) / 2.0
                        f[i] = func(v[i][0], v[i][1])
                        eval_history.append((v[i][0], v[i][1], f[i]))
        
        # Check convergence
        f_finite = f[np.isfinite(f)]
        if len(f_finite) < 2: continue
        max_f, min_f = np.max(f_finite), np.min(f_finite)
        
        rtol = 2.0 * abs(max_f - min_f) / (abs(max_f) + abs(min_f)) if (abs(max_f) + abs(min_f)) > 0 else 0.0
          
        if verbose:
            print(f"Iteration {itr}: f_min = {min_f:.6e}, rtol = {rtol:.6e}")
          
        if rtol < ftol:
            break
      
    vs = np.argmin(f)
    return v[vs][0], v[vs][1], f[vs], eval_history



print("\nStep 2: Extracting data from DataFrame for interpolation...")
points = mask_data[['e', '$θ_{e}$']].values

# Extract the chi2 column as a (N,) numpy array of values.
values = mask_data['chi2'].values
print(f"Extracted {len(values)} points.")


# 3. CREATE THE INTERPOLATION FUNCTION
print("\nStep 3: Creating an interpolation function from the data...")
def create_interpolated_chi2_func_with_bounds(points, values):
    """
    Creates a callable function that interpolates chi2 values
    and enforces the physical constraint e >= 0.
    """
    def chi2_interpolator_with_bounds(e, theta_e):
        # --- CONSTRAINT CHECK ---
        # If ellipticity is negative, this is a forbidden state.
        # Return infinity to heavily penalize the optimizer.
        if e < 0:
            return np.inf

        # If the constraint is met, proceed with interpolation as before.
        chi2 = griddata(points, values, (e, theta_e), method='linear', fill_value=np.inf)
        return chi2
        
    return chi2_interpolator_with_bounds

interpolated_chi2 = create_interpolated_chi2_func_with_bounds(points, values)
print("Interpolator function created successfully.")


# 4. RUN THE OPTIMIZER
print("\nStep 4: Running Nelder-Mead optimizer on the interpolated surface...")
initial_e = 0.6
initial_theta_e = 200

e_min, theta_e_min, chi2_min, history = nelder_mead_glafic_with_history(
    interpolated_chi2,
    initial_e,
    initial_theta_e,
    verbose=True
)

print("\n--- Optimization Finished ---")
print(f"Initial Guess: (e={initial_e}, θe={initial_theta_e})")
print(f"Found Minimum Chi-squared (interpolated): {chi2_min:.6f}")
print(f"At Parameters: (e={e_min:.4f}, θe={theta_e_min:.4f})")
print(f"Total Interpolator Evaluations: {len(history)}")

print("\nStep 5: Visualizing the results...")
plt.figure(figsize=(10, 8))

# Create a grid for a smooth contour plot
e_grid = np.linspace(data['e'].min(), data['e'].max(), 100)
theta_e_grid = np.linspace(data['$θ_{e}$'].min(), data['$θ_{e}$'].max(), 100)
E_grid, THETA_grid = np.meshgrid(e_grid, theta_e_grid)

# Interpolate the chi2 values onto this grid for plotting
CHI2_grid = griddata(points, values, (E_grid, THETA_grid), method='cubic')

# Plot the contour
contour = plt.contourf(E_grid, THETA_grid, CHI2_grid, levels=30, cmap='viridis')
plt.colorbar(contour, label='Interpolated Chi-Squared')

# Overlay the path of the optimizer
hist_e = [p[0] for p in history]
hist_theta_e = [p[1] for p in history]
plt.plot(hist_e, hist_theta_e, 'w-o', markersize=3, alpha=0.7, label='Optimizer Path')

# Plot key points
plt.plot(initial_e, initial_theta_e, 'ro', markersize=10, label='Initial Guess')
plt.plot(e_min, theta_e_min, 'g*', markersize=18, markeredgecolor='k', label=f'Found Minimum ({e_min:.2f}, {theta_e_min:.2f})')

plt.xlabel('Parameter e')
plt.ylabel('Parameter $θ_{e}$ (°)')
plt.title('Optimization on Interpolated Surface from DataFrame')
plt.legend()
plt.show()