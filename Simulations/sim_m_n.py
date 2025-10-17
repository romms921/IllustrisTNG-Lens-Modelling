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
os.chdir(base_path)
storage_path = "/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Simulations"
base_py_file = "input.py"

# ==== Function Definitions ====

def create_processor_dirs(num_cpus):
    for i in range(num_cpus):
        dir_name = storage_path + '/' + f"processor_{i}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")
        else:
            print(f"Directory already exists: {dir_name}")


# ==== Setup for MultiProcessing ====
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
