#!/usr/bin/env python
import glafic
from tqdm import tqdm
import numpy as np
import requests
import os

m = [round(x, 4) for x in np.linspace(0.01, 0.5, 100)]
n = [round(x, 1) for x in np.linspace(0, 360, 10)]
o = [round(x, 4) for x in np.linspace(-0.5, 0.5, 100)]


# Calculate total iterations for the progress bar
total_iterations = len(m) * len(n) * len(o)
print(f"Total iterations: {total_iterations}")

# Create progress bar
with tqdm(total=total_iterations, desc="Processing") as pbar:
    for i in range(len(m)):
        for j in range(len(n)):
            for k in range(len(o)):
                # Current Iteration Number 
                print(f"Processing Iteration = {i * len(n) * len(o) + j * len(o) + k + 1} of {total_iterations}")

                glafic.init(0.3, 0.7, -1.0, 0.7, f'/Volumes/LinuxTechTi/Astro/Sim 3/SIE_POS_SHEAR_{m[i]}_{n[j]}_{o[k]}', 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb = 0)

                glafic.set_secondary('chi2_splane 1', verb = 0)
                glafic.set_secondary('chi2_checknimg 1', verb = 0)
                glafic.set_secondary('chi2_restart   -1', verb = 0)
                glafic.set_secondary('chi2_usemag    1', verb = 0)
                glafic.set_secondary('hvary          0', verb = 0)
                glafic.set_secondary('ran_seed -122000', verb = 0)

                glafic.startup_setnum(2, 0, 1)
                glafic.set_lens(1, 'sie', 0.261343256161012, 1.549839e+02, 20.78, 20.78, 0.107, 23.38,  0.000000e+00,  0.000000e+00)
                glafic.set_lens(2, 'pert', 0.261343256161012, 1.0, 20.78, 20.78, m[i], n[j], 0.0, o[k])
                glafic.set_point(1, 1.0, 20.78, 20.78)

                glafic.setopt_lens(1, 0, 1, 1, 1, 1, 1, 0, 0)
                glafic.setopt_lens(2, 0, 0, 0, 0, 0, 0, 0, 0)
                glafic.setopt_point(1, 0, 1, 1)

                glafic.model_init(verb = 0)

                glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
                # glafic.parprior('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Test/Simulation/priorfile.dat')
                glafic.optimize()
                glafic.findimg()
                glafic.writecrit(1.0)
                glafic.writelens(1.0)

                glafic.quit()
                
                # Update progress bar
                pbar.update(1)

                current_iteration = i * len(n) * len(o) + j * len(o) + k + 1
                precentage_complete = (current_iteration / total_iterations) * 100
                avg_time_per_iteration = pbar.format_dict['elapsed'] / current_iteration if current_iteration > 0 else 0
                approx_time_remaining = (total_iterations - current_iteration) * avg_time_per_iteration

                # Store in a dictionary
                progress_info = {
                    'current_iteration': current_iteration,
                    'total_iterations': total_iterations,
                    'percentage_complete': precentage_complete,
                    'avg_time_per_iteration': avg_time_per_iteration,
                    'approx_time_remaining': approx_time_remaining
                }

                # Save to simulation log file
                with open('/Users/ainsleylewis/Documents/Astronomy/Discord Bot/simulation_log.txt', 'w') as log_file:
                    log_file.write(f"Iteration {current_iteration}/{total_iterations}: {progress_info}\n")


                # if current_iteration % 1000 == 0:
                #     # Get directory size in GB
                #     dir_path = '/Volumes/LinuxTechTi/Astro/Sim 3'
                #     dir_size = sum(os.path.getsize(os.path.join(dirpath,filename)) for dirpath, _, filenames in os.walk(dir_path) for filename in filenames)
                #     dir_size_gb = dir_size / (1024 * 1024 * 1024)  # Convert bytes to GB
                    
                #     requests.post("https://ntfy.sh/romms_lensingsim",
                #         data=f"Completed iteration {current_iteration}. Directory size: {dir_size_gb:.2f} GB".encode(encoding='utf-8'))