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
import sys
import csv
import subprocess

# === Num of CPUs ===
available_cpus = psutil.cpu_count(logical=False)
print(f"Available CPUs: {available_cpus}")

num_cpus = 4

# ==== Params ====
m = [round(x, 5) for x in np.linspace(0.001, 0.1, 1)]
n = [round(x, 5) for x in np.linspace(0, 360, 100)]
CHUNK_SIZE = 10

# ==== Paths ====
base_path = "/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Simulations"
storage_path = "/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Simulations"
obs_file = "obs_point.dat"
base_py_file = "input.py"
log_file_path = '/Users/ainsleylewis/Documents/Astronomy/Discord Bot/simulation_log.txt'
restart_file_path = 'simulation_restart_state.json'

# ==== Function Definitions ====

def create_processor_dirs(num_cpus):
    for i in range(num_cpus):
        dir_name = storage_path + '/' + f"processor_{i}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")
        else:
            print(f"Directory already exists: {dir_name}")

def rms_extract(model_ver, model_path, cpu_id):
    global pos_rms, mag_rms, chi2_value
    # Load the data
    with open(model_path + '/' + model_ver + '_optresult' + '.dat', 'r') as file:
        opt_result = file.readlines()

    # Find the last line with 'optimize' in it
    last_optimize_index = None
    for idx in range(len(opt_result) - 1, -1, -1):
        if 'optimize' in opt_result[idx]:
            last_optimize_index = idx
            break
    if last_optimize_index is None:
        raise ValueError("No line with 'optimize' found in the file.")

    # Extract everything after the last 'optimize' line
    opt_result = opt_result[last_optimize_index + 1:]

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
    with open(storage_path + '/processor_' + cpu_id + '/glafic.py', 'r') as file:
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
    obs_point = pd.read_csv(base_path + '/' + obs_file, delim_whitespace=True, header=None, skiprows=1, names=columnn_names)
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

def run_glafic_for_processor(proc_id):
    """Run all models assigned to a given processor."""
    values = f"{storage_path}/processor_{proc_id}/values_pairs.csv"
    glafic_script = f"{storage_path}/processor_{proc_id}/glafic.py"
    
    # Read all m values for this processor
    if not os.path.exists(values):
        print(f"File not found: {values}")
        return
    with open(values, "r") as f:
        m_values = [float(line.strip(",")[0]) for line in f if line.strip()]
        n_values = [float(line.strip().split(",")[1]) for line in f if line.strip()]

    # Read the original GLAFIC script template once
    with open(glafic_script, "w") as f:
        base_script = f.read()

        # Replace placeholders with actual values
        for m_value, n_value in zip(m_values, n_values):
            model_path = f"{storage_path}/processor_{proc_id}/model_m_{m_value:.5g}_n_{n_value:.5g}"
            modified_script = base_script.replace("m", f"{m_value:.12g}").replace("n", f"{n_value:.12g}").replace("path", f"'{model_path}'")
            f.write(modified_script + "\n\n")  # Separate different runs

            # Run GLAFIC using the modified script directly via Python -c
            subprocess.run(["python3", "-c", modified_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            

    print(f"Processor {proc_id}: Completed all models.")

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

# ==== Setup for MultiProcessing ====
os.chdir(base_path)
create_processor_dirs(num_cpus)

with open(base_py_file, 'r') as file:
    base_content = file.read()

for cpu in range(num_cpus):
    dir_name = storage_path + '/' + f"processor_{cpu}"
    py_file_path = os.path.join(dir_name, f'glafic.py')
    with open(py_file_path, 'w') as file:
        file.write(base_content)

# ==== Number of Iterations ====
total_iterations = len(m) * len(n)
print(f"Total iterations: {total_iterations}")

# Distribute models acorss CPUs
models_per_cpu = total_iterations // num_cpus

processor_models = {proc: list(range(proc, models_per_cpu * num_cpus, num_cpus)) for proc in range(num_cpus)}

# Create a single CSV file with the (m,n) pairs for each processor
for proc in range(num_cpus):
    indices = processor_models.get(proc, [])
    pairs = []
    for idx in indices:
        im = idx // len(n)       # index into m
        in_ = idx % len(n)      # index into n
        if im < len(m) and in_ < len(n):
            pairs.append((m[im], n[in_]))

    out_dir = os.path.join(storage_path, f'processor_{proc}')
    pairs_path = os.path.join(out_dir, 'values_pairs.csv')

    if pairs:
        arr = np.array(pairs, dtype=float)
        np.savetxt(pairs_path, arr, delimiter=',', header='m,n', fmt='%.12g', comments='')
    else:
        # create CSV with header only if no assignments
        with open(pairs_path, 'w') as f:
            f.write('m,n\n')

    print(f'Wrote {len(pairs)} (m,n) pairs to {out_dir}')
