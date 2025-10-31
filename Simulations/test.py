import pandas as pd
import numpy as np
import os
import moviepy as mp
import shutil
import tqdm

data = pd.read_csv('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Simulations/Output/Neirenberg/Sim Neirenberg.csv', header=0)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp # Import the multiprocessing library
import time # To time the execution

# Global variables to hold interpolation data
_INTERP_POINTS = None
_INTERP_VALUES = None

def chi2_interpolator_with_bounds(e, theta_e, phi):
    """
    Global function for interpolation. This will be called by worker processes.
    """
    # --- CONSTRAINT CHECK ---
    if e < 0:
        return np.inf

    # Proceed with 3D interpolation using global data
    chi2 = griddata(_INTERP_POINTS, _INTERP_VALUES, (e, theta_e, phi), method='linear', fill_value=np.inf)
    return chi2

def setup_interpolator(points, values):
    """
    Helper function to set the global data for the interpolator.
    """
    global _INTERP_POINTS, _INTERP_VALUES
    _INTERP_POINTS = points
    _INTERP_VALUES = values


# --- Nelder-Mead Optimizer Modified for Multiprocessing ---
def nelder_mead_glafic_with_history(pool, func, x0, y0, z0, ftol=1e-4, nmax=10000, verbose=False):
    """
    Nelder-Mead simplex optimization for 3 parameters, using a multiprocessing pool
    to parallelize function evaluations.
    """
    # Simplex coefficients
    ALPHA = 1.0
    BETA = 0.5
    GAMMA = 2.0
    n = 3

    v = np.zeros((n + 1, n))
    f = np.zeros(n + 1)
    eval_history = []

    # Set initial simplex
    v[0] = [x0, y0, z0]
    dx = 0.1 * abs(x0) if x0 != 0 else 0.1
    dy = 0.1 * abs(y0) if y0 != 0 else 0.1
    dz = 0.1 * abs(z0) if z0 != 0 else 0.1
    v[1] = [x0 + dx, y0, z0]
    v[2] = [x0, y0 + dy, z0]
    v[3] = [x0, y0, z0 + dz]

    # PARALLEL EVALUATION of the initial simplex
    f = np.array(pool.starmap(func, v))
    for i in range(n + 1):
        eval_history.append((v[i][0], v[i][1], v[i][2], f[i]))

    # Main optimization loop
    for itr in range(1, nmax + 1):
        f_finite_indices = np.where(np.isfinite(f))[0]
        if len(f_finite_indices) < n + 1:
            vs = np.argmin(f)
            shrink_points = [v[vs] + (v[i] - v[vs]) / 2.0 for i in range(n + 1) if i != vs]
            shrink_f = pool.starmap(func, shrink_points)
            
            f_idx = 0
            for i in range(n + 1):
                if i != vs:
                    v[i] = shrink_points[f_idx]
                    f[i] = shrink_f[f_idx]
                    eval_history.append((v[i][0], v[i][1], v[i][2], f[i]))
                    f_idx += 1
            continue

        vg = np.argmax(f)
        vs = np.argmin(f)
        
        f_temp = f.copy()
        f_temp[vg] = -np.inf
        vh = np.argmax(f_temp)

        vm = np.mean(v[[i for i in range(n + 1) if i != vg]], axis=0)

        # Reflection
        vr = vm + ALPHA * (vm - v[vg])
        fr = pool.starmap(func, [vr])[0] # Evaluate in parallel
        eval_history.append((vr[0], vr[1], vr[2], fr))
        if np.isnan(fr): fr = np.inf

        if f[vs] <= fr < f[vh]:
            v[vg], f[vg] = vr, fr
        elif fr < f[vs]:
            # Expansion
            ve = vm + GAMMA * (vr - vm)
            fe = pool.starmap(func, [ve])[0] # Evaluate in parallel
            eval_history.append((ve[0], ve[1], ve[2], fe))
            if np.isnan(fe): fe = np.inf
            
            if fe < fr:
                v[vg], f[vg] = ve, fe
            else:
                v[vg], f[vg] = vr, fr
        else:
            # Contraction
            vc_point = vm + BETA * (vr - vm) if fr < f[vg] else vm - BETA * (vm - v[vg])
            fc = pool.starmap(func, [vc_point])[0] # Evaluate in parallel
            eval_history.append((vc_point[0], vc_point[1], vc_point[2], fc))
            if np.isnan(fc): fc = np.inf

            if fc < f[vg]:
                v[vg], f[vg] = vc_point, fc
            else:
                # PARALLEL SHRINK
                shrink_points = [v[vs] + (v[i] - v[vs]) / 2.0 for i in range(n + 1) if i != vs]
                shrink_f = pool.starmap(func, shrink_points)
                
                f_idx = 0
                for i in range(n + 1):
                    if i != vs:
                        v[i] = shrink_points[f_idx]
                        f[i] = shrink_f[f_idx]
                        eval_history.append((v[i][0], v[i][1], v[i][2], f[i]))
                        f_idx += 1

        f_finite = f[np.isfinite(f)]
        if len(f_finite) < 2: continue
        max_f, min_f = np.max(f_finite), np.min(f_finite)
        
        rtol = 2.0 * abs(max_f - min_f) / (abs(max_f) + abs(min_f)) if (abs(max_f) + abs(min_f)) > 0 else 0.0
          
        if verbose:
            print(f"Iteration {itr}: f_min = {min_f:.6e}, rtol = {rtol:.6e}")
          
        if rtol < ftol:
            break
      
    vs = np.argmin(f)
    return v[vs][0], v[vs][1], v[vs][2], f[vs], eval_history

# --- Main Application ---
# The main script must be guarded by `if __name__ == "__main__":` for multiprocessing
if __name__ == "__main__":
    # 1. LOAD YOUR DATA
    print("Step 1: Creating sample data...")
    # Using a larger dataset to better see the performance improvement
    np.random.seed(42)
    
    # 2. PREPARE DATA FOR INTERPOLATION
    print("\nStep 2: Extracting data for interpolation...")
    points = data[['e', '$θ_{e}$', '$\\sigma$']].values
    values = data['chi2'].values
    
    # 3. SETUP THE INTERPOLATOR FOR WORKER PROCESSES
    print("\nStep 3: Setting up the interpolator function...")
    setup_interpolator(points, values)
    print("Interpolator data set globally for worker processes.")

    # 4. RUN THE OPTIMIZER
    print("\nStep 4: Running Nelder-Mead optimizer with multiprocessing...")
    initial_e = 0.55
    initial_theta_e = 195
    initial_phi = 260

    start_time = time.time()
    
    # Use a context manager to handle the pool of processes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        e_min, theta_e_min, phi_min, chi2_min, history = nelder_mead_glafic_with_history(
            pool, # Pass the pool to the optimizer
            chi2_interpolator_with_bounds,
            initial_e,
            initial_theta_e,
            initial_phi,
            verbose=True
        )
    
    end_time = time.time()

    # 5. PRINT RESULTS
    print("\n--- Optimization Finished ---")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    print(f"Initial Guess: (e={initial_e}, θe={initial_theta_e}, phi={initial_phi})")
    print(f"Found Minimum Chi-squared (interpolated): {chi2_min:.6f}")
    print(f"At Parameters: (e={e_min:.4f}, θe={theta_e_min:.4f}, phi={phi_min:.4f})")
    print(f"Total Interpolator Evaluations: {len(history)}")

    # 6. VISUALIZE THE RESULT (3D Scatter Plot)
    print("\nStep 6: Visualizing the results in 3D...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(data['e'], data['$θ_{e}$'], data['phi'], c=data['chi2'], cmap='viridis', s=20, alpha=0.5, label='Original Data Points')
    plt.colorbar(sc, ax=ax, label='Chi-Squared')
    
    hist_e = [p[0] for p in history]
    hist_theta_e = [p[1] for p in history]
    hist_phi = [p[2] for p in history]
    ax.plot(hist_e, hist_theta_e, hist_phi, 'w-o', markersize=3, alpha=0.8, label='Optimizer Path')

    ax.plot([initial_e], [initial_theta_e], [initial_phi], 'ro', markersize=10, label='Initial Guess')
    ax.plot([e_min], [theta_e_min], [phi_min], 'g*', markersize=18, markeredgecolor='k', label=f'Found Minimum')

    ax.set_xlabel('Parameter e')
    ax.set_ylabel('Parameter $θ_{e}$ (°)')
    ax.set_zlabel('Parameter phi')
    ax.set_title('3D Optimization with Multiprocessing')
    ax.legend()
    plt.show()