#!/usr/bin/env python
import glafic
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

# For debugging workers, you can comment out these lines.
# sys.stdout = open(os.devnull, 'w')
# sys.stderr = open(os.devnull, 'w')

# ==== Config ====
# --- Simulation Parameters (3 Variables) ---
m = [round(x, 5) for x in np.linspace(0.001, 0.1, 10)]
n = [round(x, 5) for x in np.linspace(0, 360, 10)]
o = [round(x, 5) for x in np.linspace(0, 1, 10)] # Third variable re-introduced

# --- File Paths (CHANGE THESE FOR YOUR SETUP) ---
model_output_dir = '/Volumes/T7 Shield/Sim 22' # New output directory
base_results_path = '/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Test/'
log_file_path = '/Users/ainsleylewis/Documents/Astronomy/Discord Bot/simulation_log.txt'
restart_file_path = os.path.join(os.path.dirname(log_file_path), 'simulation_restart_state_3var.json') # New restart file

# --- Performance & Resource Management ---
NUM_PROCESSORS = 8
CHUNK_SIZE = 100000
CONSOLE_UPDATE_INTERVAL = 100

# Ensure the output directories exist
os.makedirs(model_output_dir, exist_ok=True)
os.makedirs(base_results_path, exist_ok=True)


# ==== Helpers ====
# NOTE: All helper functions are pasted here for completeness.
def get_cpu_usage():
    return psutil.cpu_percent(interval=None)
def get_memory_usage():
    return psutil.virtual_memory().percent
def get_dir_size(directory):
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp): total_size += os.path.getsize(fp)
    except FileNotFoundError: return 0
    return total_size
def get_disk_usage_info(directory, avg_model_size, models_remaining):
    try:
        disk_root = os.path.abspath(directory)
        while not os.path.ismount(disk_root):
            parent = os.path.dirname(disk_root)
            if parent == disk_root: break
            disk_root = parent
        total, used, free = shutil.disk_usage(disk_root)
        used_percent = (used / total) * 100
        free_gb = free / (1024**3)
        projected_usage = avg_model_size * models_remaining
        if projected_usage > free: threat = "HIGH"
        elif projected_usage > 0.8 * free: threat = "MEDIUM"
        else: threat = "LOW"
        return {"used_percent": used_percent, "free_gb": free_gb, "threat_level": threat}
    except (FileNotFoundError, Exception):
        return {"used_percent": 0, "free_gb": 0, "threat_level": "Unavailable"}

def save_restart_state(path, i, j, k, chunk_number):
    """Saves the current loop indices for a 3-variable run."""
    state = {'i': i, 'j': j, 'k': k, 'chunk_number': chunk_number}
    with open(path, 'w') as f:
        json.dump(state, f, indent=4)
    with open(sys.__stdout__.fileno(), 'w', closefd=False) as real_stdout:
        real_stdout.write(f"\n✅ Simulation state saved to {path} at indices (i={i}, j={j}, k={k}), chunk={chunk_number}.\n")

def load_restart_state(path):
    """Loads loop indices for a 3-variable run."""
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
            real_stdout.write(f"⚠️ Could not read restart file. Starting fresh.\n")
        return 0, 0, 0, 1

def get_csv_filename(chunk_number):
    sim_name = model_output_dir.strip('/').split('/')[-1]
    return os.path.join(base_results_path, f"{sim_name}_summary.csv" if chunk_number == 1 else f"{sim_name}_summary_{chunk_number}.csv")

def write_to_csv(df, chunk_number):
    csv_file = get_csv_filename(chunk_number)
    header = not os.path.exists(csv_file)
    df.to_csv(csv_file, mode='a', index=False, header=header)

def calculate_chunk_number(iteration_count):
    return ((iteration_count - 1) // CHUNK_SIZE) + 1

def rms_extract(model_ver, model_path, constraint):
    global pos_rms, mag_rms, chi2_value
    # Load the data
    with open(model_path + '/' + model_ver + '_optresult' + '.dat', 'r') as file:
        opt_result = file.readlines()

    # Find the last line with 'optimize' in it
    last_optimize_index = None
    for idx in range(len(opt_result) - 1, -1, -1):
        if 'optimize' in opt_result[idx]:
            last_optimize_index = idx
            last_optimize_line = opt_result[idx]
            break
    if last_optimize_index is None:
        raise ValueError("No line with 'optimize' found in the file.")

    # Extract everything after the last 'optimize' line
    opt_result = opt_result[last_optimize_index + 1:]

    # Count the number of lines that start with 'lens'
    lens_count = sum(1 for line in opt_result if line.startswith('lens'))

    # Initialize a dictionary to hold the lens parameters
    lens_params_dict = {}

    # Extract the lens parameters
    lens_params = []
    for line in opt_result:
        if line.startswith('lens'):
            parts = re.split(r'\s+', line.strip())
            lens_name = parts[1]
            params = [float(x) for x in parts[2:]]

            # Store the parameters in the dictionary
            lens_params_dict[lens_name] = params
            lens_params.append((lens_name, params))

    # Remove the first lens parameter
    if lens_params:
        for i in range(len(lens_params)):
            lens_name, params = lens_params[i]
            lens_params_dict[lens_name] = params[1:]
    
    # Extract the chi2 
    chi2_line = next((line for line in opt_result if 'chi^2' in line), None)
    if chi2_line is None:
        raise ValueError("No line with 'chi2' found in the file.")

    chi2_value = float(chi2_line.split('=')[-1].strip().split()[0])
    # print(f"✅ Extracted chi2 value: {chi2_value}")

    # Number of len profiles
    num_lens_profiles = len(lens_params_dict)

    # Use generic column names: param1, param2, ...
    df = pd.DataFrame()
    rows = []
    max_param_len = 0

    for lens_name, params in lens_params_dict.items():
        row = {'Lens Name': lens_name}
        for i, val in enumerate(params):
            row[f'param{i+1}'] = val
        rows.append(row)
        if len(params) > max_param_len:
            max_param_len = len(params)

    columns = ['Lens Name'] + [f'param{i+1}' for i in range(max_param_len)]
    df = pd.DataFrame(rows, columns=columns)
    
    # Load the input parameters from the Python file
    with open('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Test/POS+MAG/SIE+SHEAR/pos_point.py', 'r') as file:
        py = file.readlines()

    # Extracting the input parameters from the Python file
    set_lens_lines = [line for line in py if line.startswith('glafic.set_lens(')]
    if not set_lens_lines:
        raise ValueError("No lines starting with 'glafic.set_lens(' found in the file.")

    set_lens_params = []
    for line in set_lens_lines:
        match = re.search(r'set_lens\((.*?)\)', line)
        if match:
            params_str = match.group(1)
            params = [param.strip() for param in params_str.split(',')]
            set_lens_params.append(params)
        else:
            raise ValueError(f"No valid parameters found in line: {line.strip()}")

    # Store the parameters in a dictionary
    set_lens_dict = {}
    for params in set_lens_params:
        if len(params) < 3:
            raise ValueError(f"Not enough parameters found in line: {params}")
        lens_name = params[1].strip("'\"")  # Remove quotes from lens name
        lens_params = [float(x) for x in params[2:]]  # Skip index and lens name
        set_lens_dict[lens_name] = lens_params

    # Remove the first lens parameter
    if set_lens_dict:
        for lens_name, params in set_lens_dict.items():
            set_lens_dict[lens_name] = params[1:]  # Remove the first parameter (index)

    # Use generic column names: param1, param2, ...
    df_input = pd.DataFrame()
    rows_input = []
    max_param_len_input = 0
    for lens_name, params in set_lens_dict.items():
        row = {'Lens Name': lens_name}
        for i, val in enumerate(params):
            row[f'param{i+1}'] = val
        rows_input.append(row)
        if len(params) > max_param_len_input:
            max_param_len_input = len(params)
    columns_input = ['Lens Name'] + [f'param{i+1}' for i in range(max_param_len_input)]
    df_input = pd.DataFrame(rows_input, columns=columns_input)
    
    # Extract input flags from the Python file
    set_flag_lines = [line for line in py if line.startswith('glafic.setopt_lens(')]
    if not set_flag_lines:
        raise ValueError("No lines starting with 'glafic.setopt_lens(' found in the file.")
    set_flag_params = []
    for line in set_flag_lines:
        match = re.search(r'setopt_lens\((.*?)\)', line)
        if match:
            params_str = match.group(1)
            params = [param.strip() for param in params_str.split(',')]
            set_flag_params.append(params)
        else:
            raise ValueError(f"No valid parameters found in line: {line.strip()}")
    
    # Store the parameters in a dictionary
    set_flag_dict = {}
    for params in set_flag_params:
        if len(params) < 2:
            raise ValueError(f"Not enough parameters found in line: {params}")
        # The lens name is not present in setopt_lens, so use the lens index to map to set_lens_dict
        lens_index = params[0].strip("'\"")
        # Find the lens name corresponding to this index from set_lens_params
        lens_name = None
        for lens_params in set_lens_params:
            if lens_params[0].strip("'\"") == lens_index:
                lens_name = lens_params[1].strip("'\"")
                break
        if lens_name is None:
            raise ValueError(f"Lens name for index {lens_index} not found in set_lens_params")
        flag = ','.join(params[1:])  # Join all flag values as a string
        set_flag_dict[lens_name] = flag
   
    # Remove the first flag parameter
    if set_flag_dict:
        for lens_name, flag in set_flag_dict.items():
            flag_parts = flag.split(',')
            set_flag_dict[lens_name] = ','.join(flag_parts[1:])  # Remove the first flag parameter
    
    # Dynamically create columns: 'Lens Name', 'flag1', 'flag2', ..., based on the maximum number of flags
    df_flag = pd.DataFrame()
    rows_flag = []
    max_flag_len = 0
    
    # First, determine the maximum number of flags
    for flag in set_flag_dict.values():
        flag_parts = flag.split(',')
        if len(flag_parts) > max_flag_len:
            max_flag_len = len(flag_parts)
    for lens_name, flag in set_flag_dict.items():
        flag_parts = flag.split(',')
        row = {'Lens Name': lens_name}
        for i, val in enumerate(flag_parts):
            row[f'flag{i+1}'] = val
        rows_flag.append(row)
    columns_flag = ['Lens Name'] + [f'flag{i+1}' for i in range(max_flag_len)]  
    df_flag = pd.DataFrame(rows_flag, columns=columns_flag)
    
    # Combine all dataframes into a list of dataframes for each lens
    dfs = []
    
    for i in range(num_lens_profiles):
        lens_name = df['Lens Name'][i]
        
        # Find the model type (case-insensitive match)
        model_type = None
        for m in model_list:
            if m.lower() == lens_name.lower():
                model_type = m
                break
        if model_type is None:
            continue

        symbols = model_params[model_type][:7]
        # Row 2: input
        row_input = pd.DataFrame([df_input.iloc[i, 1:8].values], columns=symbols)
        # Row 3: output
        row_output = pd.DataFrame([df.iloc[i, 1:8].values], columns=symbols)
        # Row 4: flags
        row_flags = pd.DataFrame([df_flag.iloc[i, 1:8].values], columns=symbols)

        # Stack vertically, add a label column for row type
        lens_df = pd.concat([
            row_input.assign(Type='Input'),
            row_output.assign(Type='Output'),
            row_flags.assign(Type='Flag')
        ], ignore_index=True)
        lens_df.insert(0, 'Lens Name', lens_name)
        
        # Move 'Type' to the second column
        cols = lens_df.columns.tolist()
        cols.insert(1, cols.pop(cols.index('Type')))
        lens_df = lens_df[cols]
        dfs.append(lens_df)
    
    # Anomaly Calculation
    columnn_names = ['x', 'y', 'mag', 'pos_err', 'mag_err', '1', '2', '3']
    obs_point = pd.read_csv('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/System 2/pos+flux_point.dat', delim_whitespace=True, header=None, skiprows=1, names=columnn_names)
    out_point = pd.read_csv(model_path + '/' + model_ver + '_point.dat', delim_whitespace=True, header=None, skiprows=1, names=columnn_names)
    out_point.drop(columns=['mag_err', '1', '2', '3'], inplace=True)

    # Drop rows in obs_point where the corresponding out_point['mag'] < 1
    mask = abs(out_point['mag']) >= 1
    out_point = out_point[mask[:len(out_point)]].reset_index(drop=True)
    out_point['x_diff'] = abs(out_point['x'] - obs_point['x'])
    out_point['y_diff'] = abs(out_point['y'] - obs_point['y'])
    out_point['mag_diff'] = abs(abs(out_point['mag']) - abs(obs_point['mag']))
    out_point['pos_sq'] = np.sqrt((out_point['x_diff']**2 + out_point['y_diff']**2).astype(float))  # Plotted on graph

    # RMS
    pos_rms = np.average(out_point['pos_sq'])

    mag_rms = np.average(np.sqrt((out_point['mag_diff']**2).astype(float)))

    return pos_rms, mag_rms, dfs, chi2_value

pow_params = ['$z_{s,fid}$', 'x', 'y', 'e', '$θ_{e}$', '$r_{Ein}$', '$\gamma$ (PWI)']

# SIE
sie_params = ['$\sigma$', 'x', 'y', 'e', '$θ_{e}$', '$r_{core}$', 'NaN']

# NFW
nfw_params = ['M', 'x', 'y', 'e', '$θ_{e}$', 'c or $r_{s}$', 'NaN']

# EIN
ein_params = ['M', 'x', 'y', 'e', '$θ_{e}$', 'c or $r_{s}$', r'$\alpha_{e}$']

# SHEAR 
shear_params = ['$z_{s,fid}$', 'x', 'y', '$\gamma$', '$θ_{\gamma}$', 'NaN', '$\kappa$']

# Sersic
sersic_params = ['$M_{tot}$', 'x', 'y', 'e', '$θ_{e}$', '$r_{e}$', '$n$']

# Cored SIE
cored_sie_params = ['M', 'x', 'y', 'e', '$θ_{e}$', '$r_{core}$', 'NaN']

# Multipoles
mpole_params = ['$z_{s,fid}$', 'x', 'y', '$\epsilon$', '$θ_{m}$', 'm', 'n']

model_list = ['POW', 'SIE', 'ANFW', 'EIN', 'PERT', 'SERS', 'MPOLE']
model_params = {
    'POW': pow_params,
    'SIE': sie_params,
    'ANFW': nfw_params,
    'EIN': ein_params,
    'PERT': shear_params,
    'SERS': sersic_params,
    'MPOLE' : mpole_params
}


# ==== Worker Function for Multiprocessing ====
def run_single_model(params):
    """
    Worker function for a 3-variable model run.
    """
    i, j, k, m_val, n_val, o_val = params # Unpack 3 variables and their indices
    
    model_name = f'SIE_POS_SHEAR_{m[i]}_{n[j]}_{o[k]}'
    model_path = os.path.join(model_output_dir, model_name)
    try:
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

        columns = ['x', 'y', 'm', 'm_err']

        macro_model_params = model_name.strip().split('_')[0]
        macro_columns = model_params[macro_model_params]

        df = pd.DataFrame(columns=['strength', 'pa', 'o_param', 'num_images', 'pos_rms', 'mag_rms', 't_mpole_str', 't_mpole_pa', 'chi2'] + macro_columns)

        model_ver = model_name

        if 'POS+FLUX' in model_ver:
            constraint = 'pos_flux'
        elif 'POS' in model_ver:
            constraint = 'pos'

        pos_rms, mag_rms, dfs, chi2 = rms_extract(model_name, model_output_dir, constraint)
        if pos_rms == -1: raise IOError("rms_extract failed.")
        point_file_path = model_path + '_point.dat'; num_images = 0
        if os.path.exists(point_file_path):
            with open(point_file_path, 'r') as f: num_images = sum(1 for line in f if line.strip()) - 1
        
        # Result dictionary with 'o_param'
        result_dict = {
            'strength': m_val, 
            'pa': n_val, 
            'o_param': o_val,
            'num_images': num_images, 
            'pos_rms': pos_rms, 
            'mag_rms': mag_rms,
            't_mpole_str': dfs[1]['$\epsilon$'].iloc[0] if dfs and len(dfs) > 1 else 0,
            't_mpole_pa': dfs[1]['$θ_{m}$'].iloc[0] if dfs and len(dfs) > 1 else 0,
            'chi2': chi2,
            **{col: dfs[0][col].iloc[0] if dfs else 0 for col in macro_columns}
        }
    except Exception as e:
        result_dict = {'error': str(e)}
    finally:
        for suffix in ['_point.dat', '_optresult.dat']:
            file_to_delete = model_path + suffix
            if os.path.exists(file_to_delete): os.remove(file_to_delete)
            
    # Return 3 indices and the result
    return (i, j, k, result_dict)


# ==== Main Process ====
def main():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    print(f"Starting 3-variable simulation with {NUM_PROCESSORS} processes.")
    # Generate task list for 3 variables
    all_tasks = [(i, j, k, m[i], n[j], o[k]) for i, j, k in product(range(len(m)), range(len(n)), range(len(o)))]
    total_iterations = len(all_tasks)
    print(f"Total iterations to run: {total_iterations}")

    # Load state for 3 variables
    start_i, start_j, start_k, chunk_number = load_restart_state(restart_file_path)
    start_index = 0
    if start_i > 0 or start_j > 0 or start_k > 0:
        try:
            # Find start index based on 3 indices
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
    last_i, last_j, last_k = start_i, start_j, start_k # Track 3 indices
    avg_model_size, initial_dir_size = 0, get_dir_size(base_results_path)
    disk_info = {"used_percent": 0, "free_gb": 0, "threat_level": "Calculating..."}

    try:
        with multiprocessing.Pool(processes=NUM_PROCESSORS) as pool:
            results_iterator = pool.imap_unordered(run_single_model, tasks_to_run)
            
            # Unpack 3 indices from results
            for i_res, j_res, k_res, result_dict in results_iterator:
                if 'error' in result_dict:
                    print(f"Warning: A worker failed with error: {result_dict['error']}")
                    continue
                
                last_i, last_j, last_k = i_res, j_res, k_res
                current_chunk = calculate_chunk_number(iterations_done + 1)
                write_to_csv(pd.DataFrame([result_dict]), current_chunk)
                iterations_done += 1

                if iterations_done % CONSOLE_UPDATE_INTERVAL == 0 or iterations_done == total_iterations:
                    percentage_complete = (iterations_done / total_iterations) * 100
                    print(f"Progress: {iterations_done} / {total_iterations} ({percentage_complete:.2f}%)")

                # --- LOG FILE WRITING (EVERY ITERATION, ORIGINAL FORMAT) ---
                models_this_run = iterations_done - start_index
                
                if models_this_run > 0 and models_this_run % 100 == 0:
                    current_dir_size = get_dir_size(base_results_path)
                    avg_model_size = (current_dir_size - initial_dir_size) / models_this_run
                    disk_info = get_disk_usage_info(base_results_path, avg_model_size, total_iterations - iterations_done)
                
                percentage_complete = (iterations_done / total_iterations) * 100
                elapsed_time = time.time() - start_time
                avg_time_per_iteration = elapsed_time / models_this_run if models_this_run > 0 else 0
                approx_time_remaining = (total_iterations - iterations_done) * avg_time_per_iteration

                progress_info = {
                    'current_iteration': iterations_done,
                    'total_iterations': total_iterations,
                    'percentage_complete': percentage_complete,
                    'avg_time_per_iteration': avg_time_per_iteration,
                    'approx_time_remaining': approx_time_remaining,
                    'ram_usage_percent': get_memory_usage(),
                    'disk_threat_level': disk_info['threat_level'],
                    'cpu_usage_percent': get_cpu_usage(),
                    'disk_used_percent': disk_info['used_percent'],
                    'disk_free': round(disk_info['free_gb'], 2) if isinstance(disk_info['free_gb'], (int, float)) else disk_info['free_gb'],
                    'current_chunk': current_chunk
                }

                with open(log_file_path, 'w') as log_file:
                    log_file.write(f"Iteration {iterations_done}/{total_iterations}: {progress_info}\n")

                if iterations_done - last_state_saved_at_iteration >= 500:
                    # Save state with 3 indices
                    save_restart_state(restart_file_path, last_i, last_j, last_k, current_chunk)
                    last_state_saved_at_iteration = iterations_done
    
    except KeyboardInterrupt:
        print("\n⚠️ Simulation interrupted by user.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nFinalizing simulation state...")
        if iterations_done == total_iterations:
            print("🎉 Simulation completed successfully!")
            if os.path.exists(restart_file_path):
                os.remove(restart_file_path)
                print("🗑️ Removed restart file.")
        else:
            # Save final state with 3 indices
            save_restart_state(restart_file_path, last_i, last_j, last_k, calculate_chunk_number(iterations_done))
        
        if os.path.exists(model_output_dir):
            shutil.rmtree(model_output_dir)
            print(f"🗑️ Removed temporary output directory: {model_output_dir}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()