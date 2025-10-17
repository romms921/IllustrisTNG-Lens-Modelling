#!/usr/bin/env python
import glafic
from tqdm import tqdm
import numpy as np
import psutil
import shutil
import os
import time
import requests
import json
import pandas as pd
import re
import sys
import csv
import multiprocessing
from itertools import product
import uuid

# Redirect stdout/stderr to devnull for cleaner output from child processes
# For debugging workers, comment these lines out.
# sys.stdout = open(os.devnull, 'w')
# sys.stderr = open(os.devnull, 'w')

# ==== Config ====
# --- Simulation Parameters ---
m = [round(x, 5) for x in np.linspace(0.001, 0.1, 10)]
n = [round(x, 5) for x in np.linspace(0, 360, 10)]
o = [round(x, 5) for x in np.linspace(0, 1, 10)]

# --- File Paths (CHANGE THESE FOR YOUR SETUP) ---
# For an external drive: '/Volumes/T7 Shield/Sim_18_Parallel_Outputs'
# For a local folder: '/Users/ainsleylewis/Documents/Sim_18_Parallel_Outputs'
model_output_dir = '/Volumes/T7 Shield/Sim 21' # <<< Local path example
base_results_path = '/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Test/'
log_file_path = '/Users/ainsleylewis/Documents/Astronomy/Discord Bot/simulation_log.txt'
restart_file_path = os.path.join(os.path.dirname(log_file_path), 'simulation_restart_state.json')

# --- Performance & Resource Management ---
# Leave one core free for system stability
NUM_PROCESSORS = 5
ram_threshold_percent = 90
disk_check_interval = 1000
CHUNK_SIZE = 100000

# Ensure the output directories exist
os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(base_results_path, exist_ok=True)


# ==== Helpers ====
# Note: Helper functions are defined globally so they can be pickled and sent to child processes.
def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

def get_memory_usage():
    return psutil.virtual_memory().percent

def get_dir_size(directory):
    """
    Recursively calculates the total size of all files in a directory.
    This function is now correctly defined.
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                # skip if it is a symbolic link
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except FileNotFoundError:
        return 0 # Return 0 if the directory doesn't exist yet
    return total_size

def get_disk_usage_info(directory, avg_model_size, models_remaining):
    """
    Safely checks disk usage. Returns a dictionary with usage info or 'N/A' on failure.
    This function is robust for both local and external drives.
    """
    try:
        disk_root = os.path.abspath(directory)
        while not os.path.ismount(disk_root):
            parent = os.path.dirname(disk_root)
            if parent == disk_root:
                break
            disk_root = parent

        total, used, free = shutil.disk_usage(disk_root)
        
        used_percent = (used / total) * 100
        free_gb = free / (1024**3)
        
        projected_usage = avg_model_size * models_remaining
        if projected_usage > free:
            threat = "HIGH"
        elif projected_usage > 0.8 * free:
            threat = "MEDIUM"
        else:
            threat = "LOW"
            
        return {
            "used_percent": f"{used_percent:.2f}%",
            "free_gb": f"{free_gb:.2f} GB",
            "threat_level": threat
        }
    except (FileNotFoundError, Exception):
        return {
            "used_percent": "N/A",
            "free_gb": "N/A",
            "threat_level": "Unavailable"
        }

def save_restart_state(path, i, j, k, chunk_number):
    """Saves the current loop indices and chunk number to a JSON file."""
    state = {'i': i, 'j': j, 'k': k, 'chunk_number': chunk_number}
    with open(path, 'w') as f:
        json.dump(state, f, indent=4)
    # Use sys.__stdout__ to ensure this message is always printed to the console
    with open(sys.__stdout__.fileno(), 'w', closefd=False) as real_stdout:
        real_stdout.write(f"\n✅ Simulation state saved to {path} at indices (i={i}, j={j}, k={k}), chunk={chunk_number}.\n")

def load_restart_state(path):
    """Loads loop indices and chunk number from a JSON file. Returns (0, 0, 0, 1) if not found or invalid."""
    if not os.path.exists(path):
        with open(sys.__stdout__.fileno(), 'w', closefd=False) as real_stdout:
            real_stdout.write("ℹ️ Restart file not found. Starting a fresh simulation.\n")
        return 0, 0, 0, 1
    try:
        with open(path, 'r') as f:
            state = json.load(f)
            i, j, k = state.get('i', 0), state.get('j', 0), state.get('k', 0)
            chunk = state.get('chunk_number', 1)
            with open(sys.__stdout__.fileno(), 'w', closefd=False) as real_stdout:
                real_stdout.write(f"✅ Restart file found. Resuming from (i={i}, j={j}, k={k}), chunk={chunk}.\n")
            return i, j, k, chunk
    except (json.JSONDecodeError, KeyError):
        with open(sys.__stdout__.fileno(), 'w', closefd=False) as real_stdout:
            real_stdout.write(f"⚠️ Could not read restart file: {e}. Starting fresh.\n")
        return 0, 0, 0, 1

def get_csv_filename(chunk_number):
    """Returns the appropriate CSV filename for the given chunk number."""
    sim_name = "Sim_21_Parallel"
    if chunk_number == 1:
        return os.path.join(base_results_path, f"{sim_name}_summary.csv")
    else:
        return os.path.join(base_results_path, f"{sim_name}_summary_{chunk_number}.csv")

def write_to_csv(df, chunk_number):
    csv_file = get_csv_filename(chunk_number)
    header = not os.path.exists(csv_file)
    df.to_csv(csv_file, mode='a', index=False, header=header)

def calculate_chunk_number(iteration_count):
    return ((iteration_count - 1) // CHUNK_SIZE) + 1

# NOTE: Paste your full rms_extract function and model_params dictionary here.
# For brevity, they are omitted from this code block.
def rms_extract(model_ver, model_path, constraint):
    # This is a placeholder for your actual function.
    # Replace this with your full, correct implementation.
    try:
        pos_rms, mag_rms, chi2 = np.random.rand(3)
        dfs_dummy = [
            pd.DataFrame(np.random.rand(1, 7), columns=model_params['SIE']),
            pd.DataFrame({'$\epsilon$': [np.random.rand()], '$θ_{m}$': [np.random.rand()]})
        ]
        return pos_rms, mag_rms, dfs_dummy, chi2
    except Exception:
        return -1, -1, None, -1
model_params = {'SIE': ['$\sigma$', 'x', 'y', 'e', '$θ_{e}$', '$r_{core}$', 'NaN']}


# ==== Worker Function for Multiprocessing ====
def run_single_model(params):
    """
    This function runs a single glafic model for a given set of parameters.
    It's designed to be executed in a separate process.
    """
    i, j, k, m_val, n_val, o_val = params
    
    # --- MODIFICATION START ---
    # Create a SHORT, UNIQUE model name using UUID to prevent buffer overflows in glafic.
    # The descriptive name was too long for the Fortran library to handle.
    model_name = f"glafic_run_{uuid.uuid4()}"
    model_path = os.path.join(model_output_dir, model_name)
    # --- MODIFICATION END ---

    try:
        # The actual glafic call uses the short, safe model_path
        glafic.init(0.3, 0.7, -1.0, 0.7, model_path, -3.0, -3.0, 3.0, 3.0, 0.01, 0.01, 1, verb=0)
        glafic.set_secondary('chi2_splane 1', verb=0)
        glafic.set_secondary('chi2_checknimg 0', verb=0)
        glafic.set_secondary('chi2_restart   -1', verb=0)
        glafic.set_secondary('chi2_usemag    1', verb=0)
        glafic.set_secondary('hvary          0', verb=0)
        glafic.set_secondary('ran_seed -122000', verb=0)
        glafic.startup_setnum(2, 0, 1)
        glafic.set_lens(1, 'sie', 0.261343256161012, 1.563051e+02, 0.0, 0.0, 2.168966e-01, -1.398259e+00,  0.0, 0.0)
        glafic.set_lens(2, 'pert', 0.261343256161012, 1.0, 0.0, 0.0, m_val, n_val, 0.0, o_val)
        glafic.set_point(1, 1.0, 0.0, 0.0)
        glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 0, 0)
        glafic.setopt_lens(2, 0, 0, 0, 0, 1, 1, 0, 1)
        glafic.setopt_point(1, 0, 1, 1)
        glafic.model_init(verb=0)
        glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/System 2/pos_point.dat')
        glafic.optimize()
        glafic.findimg()
        glafic.quit()

        # The model name passed to rms_extract needs to match what glafic used
        macro_columns = model_params["SIE"]
        pos_rms, mag_rms, dfs, chi2 = rms_extract(model_name, model_output_dir, 'pos')
        if pos_rms == -1: raise IOError("rms_extract failed.")

        point_file_path = model_path + '_point.dat'
        num_images = 0
        if os.path.exists(point_file_path):
            with open(point_file_path, 'r') as f:
                num_images = sum(1 for line in f if line.strip()) - 1
        
        # IMPORTANT: The results dictionary still contains the ACTUAL parameter values.
        # We have lost no information.
        result_dict = {
            'strength': m_val, 'pa': n_val, 'o_param': o_val,
            'num_images': num_images, 'pos_rms': pos_rms, 'mag_rms': mag_rms,
            't_mpole_str': dfs[1]['$\epsilon$'].iloc[0] if dfs else 0,
            't_mpole_pa': dfs[1]['$θ_{m}$'].iloc[0] if dfs else 0,
            'chi2': chi2,
            **{col: dfs[0][col].iloc[0] if dfs else 0 for col in macro_columns}
        }
    except Exception as e:
        result_dict = {'error': str(e)}
    finally:
        # Cleanup uses the same short path, ensuring the correct temp files are deleted.
        for suffix in ['_point.dat', '_optresult.dat']:
            file_to_delete = model_path + suffix
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)
                
    # The return value still includes the original indices for perfect state tracking.
    return (i, j, k, result_dict)


# ==== Main Process ====
def main():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    print(f"Starting simulation with {NUM_PROCESSORS} processes.")
    all_tasks = [(i, j, k, m[i], n[j], o[k]) for i, j, k in product(range(len(m)), range(len(n)), range(len(o)))]
    total_iterations = len(all_tasks)
    print(f"Total iterations to run: {total_iterations}")

    start_i, start_j, start_k, chunk_number = load_restart_state(restart_file_path)
    start_index = 0
    if start_i > 0 or start_j > 0 or start_k > 0:
        try:
            last_idx = [task[:3] for task in all_tasks].index((start_i, start_j, start_k))
            start_index = last_idx + 1
        except ValueError:
            print("Warning: Restart indices not found. Starting from scratch.")
    tasks_to_run = all_tasks[start_index:]
    iterations_done = start_index
    
    if not tasks_to_run:
        print("🎉 Simulation already completed!")
        if os.path.exists(restart_file_path): os.remove(restart_file_path)
        return
        
    print(f"Resuming from iteration {iterations_done}. {len(tasks_to_run)} tasks remaining.")

    start_time = time.time()
    last_state_saved_at_iteration = iterations_done
    last_i, last_j, last_k = start_i, start_j, start_k
    avg_model_size = 0 
    initial_dir_size = get_dir_size(base_results_path)

    try:
        with multiprocessing.Pool(processes=NUM_PROCESSORS) as pool:
            results_iterator = pool.imap_unordered(run_single_model, tasks_to_run)
            pbar = tqdm(results_iterator, total=len(tasks_to_run), desc="Initializing...")
            
            for i_res, j_res, k_res, result_dict in pbar:
                if 'error' in result_dict: continue
                
                last_i, last_j, last_k = i_res, j_res, k_res
                current_chunk = calculate_chunk_number(iterations_done + 1)
                write_to_csv(pd.DataFrame([result_dict]), current_chunk)
                iterations_done += 1

                if iterations_done % disk_check_interval == 0:
                    models_this_run = iterations_done - start_index
                    if models_this_run > 0:
                        current_dir_size = get_dir_size(base_results_path)
                        avg_model_size = (current_dir_size - initial_dir_size) / models_this_run
                    
                    disk_info = get_disk_usage_info(base_results_path, avg_model_size, total_iterations - iterations_done)
                    pbar.set_description(
                        f"Processing | CPU:{get_cpu_usage()}% RAM:{get_memory_usage()}% "
                        f"Disk:{disk_info['used_percent']} Free:{disk_info['free_gb']} "
                        f"Threat:{disk_info['threat_level']}"
                    )

                if iterations_done - last_state_saved_at_iteration >= 500:
                    save_restart_state(restart_file_path, last_i, last_j, last_k, current_chunk)
                    last_state_saved_at_iteration = iterations_done
    
    except KeyboardInterrupt:
        print("\n⚠️ Simulation interrupted by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred in the main process: {e}")
    finally:
        print("\nFinalizing simulation state...")
        if iterations_done == total_iterations:
            print("🎉 Simulation completed successfully!")
            if os.path.exists(restart_file_path):
                os.remove(restart_file_path)
                print("🗑️ Removed restart file.")
        else:
            save_restart_state(restart_file_path, last_i, last_j, last_k, calculate_chunk_number(iterations_done))
        
        if os.path.exists(model_output_dir):
            shutil.rmtree(model_output_dir)
            print(f"🗑️ Removed temporary output directory: {model_output_dir}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

    