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
import pandas as pd
import re

# ==== Config ====
m = [round(x, 4) for x in np.linspace(0.01, 0.5, 100)]
n = [round(x, 1) for x in np.linspace(0, 360, 100)]
o = [round(x, 4) for x in np.linspace(-0.5, 0.5, 100)]

ram_threshold_percent = 90
disk_check_interval = 100
critical_disk_usage_percent = 90

model_output_dir = '/Volumes/T7 Shield/Sim 6'
log_file_path = '/Users/ainsleylewis/Documents/Astronomy/Discord Bot/simulation_log.txt'


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
    
# Define the lens corresponding parameters (order preserved)
# POW
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

# Define function to make both tables
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
    print(f"✅ Extracted chi2 value: {chi2_value}")

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
    with open('Test/POS+MAG/SIE+SHEAR/pos_point.py', 'r') as file:
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
    obs_point = pd.read_csv('obs_point/obs_point_(POS+FLUX).dat', delim_whitespace=True, header=None, skiprows=1, names=columnn_names)
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

                    columns = ['x', 'y', 'm', 'm_err']

                    df = pd.DataFrame(columns=['strength', 'pa', 'kappa', 'num_images', 'pos_rms', 'mag_rms', 't_shear_str', 't_shear_pa', 't_shear_kappa', 'sie_vel_disp', 'sie_pa', 'sie_ell', 'chi2'])

                    model_ver = model_name
                    model_path_0 = model_output_dir

                    if 'POS+FLUX' in model_ver:
                        constraint = 'pos_flux'
                    elif 'POS' in model_ver:
                        constraint = 'pos'

                    pos_rms, mag_rms, dfs, chi2 = rms_extract(model_ver, model_path_0, constraint)

                    file_name = model_path + '_point.dat'

                    if os.path.exists(file_name):
                        data = pd.read_csv(file_name, delim_whitespace=True, skiprows=1, header=None, names=columns)
                        num_images = len(data)
                        df = pd.concat([df, pd.DataFrame({
                            'strength': [m[i]],
                            'pa': [n[j]],
                            'kappa': [o[k]],
                            'num_images': [num_images],
                            'pos_rms': [pos_rms],
                            'mag_rms': [mag_rms],
                            't_shear_str': dfs[1]['$\gamma$'][1],
                            't_shear_pa': dfs[1]['$θ_{\gamma}$'][1],
                            't_shear_kappa': dfs[1]['$\kappa$'][1],
                            'sie_vel_disp': dfs[0]['$\sigma$'][1],
                            'sie_pa': dfs[0]['$θ_{e}$'][1],
                            'sie_ell': dfs[0]['e'][1],
                            'chi2': chi2

                        })], ignore_index=True)
                        if data.empty:
                            print(f"File {file_name} is empty.")
                        else:
                            print(f"File {file_name} exists and is not empty.")
                            if not os.path.exists('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Test/' + model_output_dir.split('/')[-1] + '_summary.csv'):
                                df.to_csv('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Test/' + model_output_dir.split('/')[-1] + '_summary.csv', index=False)
                            else:
                                old_df = pd.read_csv('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Test/' + model_output_dir.split('/')[-1] + '_summary.csv')
                                df = pd.concat([old_df, df], ignore_index=True)
                                df.to_csv('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Test/' + model_output_dir.split('/')[-1] + '_summary.csv', index=False)
                                
                            # Delete generated files to save space
                            # Define Files 
                            crit_file = model_path + '_crit.dat'
                            lens_file = model_path + '_lens.fits'
                            point_file = model_path + '_point.dat'
                            opt_file = model_path + '_optresult.dat'

                            # Delete Files
                            if os.path.exists(crit_file):
                                os.remove(crit_file)
                            if os.path.exists(lens_file):
                                os.remove(lens_file)
                            if os.path.exists(point_file):
                                os.remove(point_file)
                            if os.path.exists(opt_file):
                                os.remove(opt_file)
                    else:
                        print(f"File {file_name} does not exist.")

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