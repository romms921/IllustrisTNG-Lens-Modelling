#!/usr/bin/env python
import glafic
from tqdm import tqdm
import numpy as np
import psutil
import shutil
import os
import time
import requests
import json  # ### NEW ### - For handling the restart file

# ==== Config ====
m = [round(x, 4) for x in np.linspace(0.01, 0.5, 100)]
n = [round(x, 1) for x in np.linspace(0, 360, 50)]
o = [round(x, 4) for x in np.linspace(-0.5, 0.5, 50)]

ram_threshold_percent = 90
disk_check_interval = 100
critical_disk_usage_percent = 90

model_output_dir = '/Volumes/T7 Shield/Sim 6'
log_file_path = '/Users/ainsleylewis/Documents/Astronomy/Discord Bot/simulation_log.txt'

### NEW/MODIFIED ###
# Path for the restart state file. It will be created in the same directory as the log file.
restart_file_path = os.path.join(os.path.dirname(log_file_path), 'simulation_restart_state.json')

# ==== Helpers ====
def get_memory_usage():
    return psutil.virtual_memory().percent

def get_dir_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size

def get_disk_threat_assessment(directory, avg_model_size, models_remaining):
    try:
        total, used, free = shutil.disk_usage(directory)
        projected_usage = avg_model_size * models_remaining
        if projected_usage > free:
            return "HIGH"
        elif projected_usage > 0.8 * free:
            return "MEDIUM"
        else:
            return "LOW"
    except FileNotFoundError:
        return "UNKNOWN (Disk not found)"


def upload_to_replit(log_path: str, replit_url: str = "https://fd07c8f5-4e98-4ab1-92c3-e95ee5cf451d-00-h61aopcz3tjm.janeway.replit.dev/upload"):
    try:
        with open(log_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(replit_url, files=files)
            if response.status_code == 200:
                print("✅ Log file uploaded to Replit successfully.")
            else:
                print(f"⚠️ Failed to upload log file: {response.text}")
    except Exception as e:
        print(f"❌ Error during upload: {e}")

### NEW ###
def save_restart_state(path, i, j, k):
    """Saves the current loop indices to a JSON file."""
    state = {'i': i, 'j': j, 'k': k}
    with open(path, 'w') as f:
        json.dump(state, f, indent=4)
    print(f"\n✅ Simulation state saved to {path} at indices (i={i}, j={j}, k={k}).")

### NEW ###
def load_restart_state(path):
    """Loads loop indices from a JSON file. Returns (0, 0, 0) if not found or invalid."""
    if not os.path.exists(path):
        print("ℹ️ Restart file not found. Starting a fresh simulation.")
        return 0, 0, 0
    try:
        with open(path, 'r') as f:
            state = json.load(f)
            i = state.get('i', 0)
            j = state.get('j', 0)
            k = state.get('k', 0)
            print(f"✅ Restart file found. Resuming simulation from indices (i={i}, j={j}, k={k+1}).")
            # We resume from k+1, as the saved state is the last *completed* one.
            # The loop logic below handles this correctly.
            return i, j, k
    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️ Could not read restart file: {e}. Starting a fresh simulation.")
        return 0, 0, 0

# ==== Main ====
total_iterations = len(m) * len(n) * len(o)
print(f"Total iterations: {total_iterations}")

### NEW/MODIFIED ###
# Load the state from where we left off
start_i, start_j, start_k = load_restart_state(restart_file_path)

# Calculate the number of iterations already completed to set the progress bar correctly
iterations_done = start_i * len(n) * len(o) + start_j * len(o) + start_k
iteration_count = iterations_done

# Adjust for the very first run
if not (start_i == 0 and start_j == 0 and start_k == 0):
    # The saved 'k' is the last one *completed*, so we start the next loop at k+1
    start_k += 1
    iteration_count += 1

initial_size = get_dir_size(model_output_dir)

# --- Initialize tracking variables ---
last_ram_usage = get_memory_usage()
last_threat = "LOW"
try:
    total, used, free = shutil.disk_usage(model_output_dir)
    last_disk_used_percent = used / total * 100
    last_disk_free_bytes = free
except FileNotFoundError:
    print(f"⚠️ Warning: Output directory {model_output_dir} not found. Cannot assess disk usage.")
    last_disk_used_percent = 0
    last_disk_free_bytes = 0

# ==== Main Loop ====
# ### NEW/MODIFIED ### - Wrap the main loop in a try...finally block
pbar = None
try:
    with tqdm(total=total_iterations, desc="Processing", initial=iterations_done) as pbar:
        # ### NEW/MODIFIED ### - Adjust loop ranges based on loaded state
        for i in range(start_i, len(m)):
            # On the first resumed 'i', start 'j' from its saved state. For all subsequent 'i's, start 'j' from 0.
            j_start_index = start_j if i == start_i else 0
            for j in range(j_start_index, len(n)):
                # On the first resumed 'i' and 'j', start 'k' from its saved state. Otherwise, start 'k' from 0.
                k_start_index = start_k if i == start_i and j == start_j else 0
                for k in range(k_start_index, len(o)):

                    # --- This is the start of your original loop body ---
                    model_name = f'SIE_POS_SHEAR_{m[i]}_{n[j]}_{o[k]}'
                    model_path = os.path.join(model_output_dir, model_name)

                    print(f"\nProcessing Iteration = {iteration_count + 1} of {total_iterations} | Indices(i={i}, j={j}, k={k})")

                    # --- Model Generation ---
                    glafic.init(0.3, 0.7, -1.0, 0.7, model_path, 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb=0)
                    glafic.set_secondary('chi2_splane 1', verb=0)
                    glafic.set_secondary('chi2_checknimg 1', verb=0)
                    glafic.set_secondary('chi2_restart   -1', verb=0)
                    glafic.set_secondary('chi2_usemag    1', verb=0)
                    glafic.set_secondary('hvary          0', verb=0)
                    glafic.set_secondary('ran_seed -122000', verb=0)
                    glafic.startup_setnum(2, 0, 1)
                    glafic.set_lens(1, 'sie', 0.261343256161012, 1.549839e+02, 20.78, 20.78, 0.107, 23.38, 0.0, 0.0)
                    glafic.set_lens(2, 'pert', 0.261343256161012, 1.0, 20.78, 20.78, m[i], n[j], 0.0, o[k])
                    glafic.set_point(1, 1.0, 20.78, 20.78)
                    glafic.setopt_lens(1, 0, 1, 1, 1, 1, 1, 0, 0)
                    glafic.setopt_lens(2, 0, 0, 0, 0, 0, 0, 0, 0)
                    glafic.setopt_point(1, 0, 1, 1)
                    glafic.model_init(verb=0)
                    glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
                    glafic.optimize()
                    glafic.findimg()
                    glafic.writecrit(1.0)
                    glafic.writelens(1.0)
                    glafic.quit()

                    iteration_count += 1
                    pbar.update(1)
                    # --- This is the end of your original loop body ---

                    # --- Update RAM and Disk Threat Every N Iterations ---
                    if iteration_count % disk_check_interval == 0:
                        size_now = get_dir_size(model_output_dir)
                        # Avoid division by zero if starting from a resumed state
                        models_generated_this_run = iteration_count - iterations_done
                        if models_generated_this_run > 0:
                            size_diff = size_now - initial_size
                            avg_model_size = size_diff / models_generated_this_run
                        else:
                            avg_model_size = 0 # Cannot calculate average yet

                        models_remaining = total_iterations - iteration_count
                        last_ram_usage = get_memory_usage()
                        last_threat = get_disk_threat_assessment(model_output_dir, avg_model_size, models_remaining)

                        try:
                            total, used, free = shutil.disk_usage(model_output_dir)
                            disk_used_percent = 100 * used / total
                            last_disk_used_percent = disk_used_percent
                            last_disk_free_bytes = free
                        except FileNotFoundError:
                            disk_used_percent = 0

                        print(f"🧠 Disk Threat: {last_threat} | Used: {disk_used_percent:.2f}% | RAM: {last_ram_usage:.1f}%")
                        # ... (rest of your check logic is fine)

                    # ... (rest of your loop logic is fine)

                    # --- Save Simulation Progress Info ---
                    percentage_complete = (iteration_count / total_iterations) * 100
                    # Use pbar's internal stats for better ETA
                    avg_time_per_iteration = pbar.format_dict['elapsed'] / pbar.n if pbar.n > 0 else 0
                    approx_time_remaining = (total_iterations - iteration_count) * avg_time_per_iteration

                    progress_info = {
                        'current_iteration': iteration_count,
                        'total_iterations': total_iterations,
                        'percentage_complete': percentage_complete,
                        'avg_time_per_iteration': avg_time_per_iteration,
                        'approx_time_remaining': approx_time_remaining,
                        'ram_usage_percent': last_ram_usage,
                        'disk_threat_level': last_threat,
                        'disk_used_percent': last_disk_used_percent,
                        'disk_free': round(last_disk_free_bytes / (1024 ** 3), 2)
                    }

                    with open(log_file_path, 'a') as log_file: # Changed to 'a' (append) to not overwrite log on resume
                        log_file.write(f"Iteration {iteration_count}/{total_iterations}: {progress_info}\n")

                    if iteration_count % 100 == 0:
                        upload_to_replit(log_file_path)

                # ### NEW ### - Reset inner loop start indices for the next outer loop iteration
                start_k = 0
            start_j = 0

    # ### NEW/MODIFIED ### - If loop completes successfully, clean up the restart file
    print("\n🎉 Simulation completed successfully!")
    if os.path.exists(restart_file_path):
        os.remove(restart_file_path)
        print(f"🗑️ Removed restart file: {restart_file_path}")

finally:
    # ### NEW/MODIFIED ### - This block runs on normal exit, error, or Ctrl+C
    if pbar is not None and pbar.n < pbar.total:
        # The simulation did not complete. Save its state.
        # 'i', 'j', 'k' will hold the indices of the last *completed* iteration
        save_restart_state(restart_file_path, i, j, k)