#!/usr/bin/env python
import glafic
from tqdm import tqdm
import numpy as np
import psutil
import shutil
import os
import time
import requests

# ==== Config ====
m = [round(x, 4) for x in np.linspace(0.01, 0.5, 50)]
n = [round(x, 1) for x in np.linspace(0, 360, 10)]
o = [round(x, 4) for x in np.linspace(-0.5, 0.5, 50)]

ram_threshold_percent = 90
disk_check_interval = 10
critical_disk_usage_percent = 90

model_output_dir = '/Volumes/Astro/Sim 5'
log_file_path = '/Users/ainsleylewis/Documents/Astronomy/Discord Bot/simulation_log.txt'

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
    total, used, free = shutil.disk_usage(directory)
    projected_usage = avg_model_size * models_remaining
    if projected_usage > free:
        return "HIGH"
    elif projected_usage > 0.8 * free:
        return "MEDIUM"
    else:
        return "LOW"

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

# ==== Main ====
total_iterations = len(m) * len(n) * len(o)
print(f"Total iterations: {total_iterations}")
model_sizes = []
iteration_count = 0
initial_size = get_dir_size(model_output_dir)

last_ram_usage = get_memory_usage()
last_threat = "LOW"
initial_size = get_dir_size(model_output_dir)
last_disk_used_percent = shutil.disk_usage(model_output_dir).used / shutil.disk_usage(model_output_dir).total * 100
total, used, free = shutil.disk_usage(model_output_dir)
last_disk_free_bytes = free


# ==== Main Loop ====
with tqdm(total=total_iterations, desc="Processing") as pbar:
    for i in range(len(m)):
        for j in range(len(n)):
            for k in range(len(o)):
                iteration_count += 1
                model_name = f'SIE_POS_SHEAR_{m[i]}_{n[j]}_{o[k]}'
                model_path = os.path.join(model_output_dir, model_name)

                print(f"Processing Iteration = {iteration_count} of {total_iterations}")

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

                # --- Update RAM and Disk Threat Every N Iterations ---
                if iteration_count % disk_check_interval == 0:
                    size_now = get_dir_size(model_output_dir)
                    size_diff = size_now - initial_size
                    avg_model_size = size_diff / max(1, iteration_count)
                    models_remaining = total_iterations - iteration_count

                    last_ram_usage = get_memory_usage()
                    last_threat = get_disk_threat_assessment(model_output_dir, avg_model_size, models_remaining)

                    total, used, free = shutil.disk_usage(model_output_dir)
                    disk_used_percent = 100 * used / total

                    last_disk_used_percent = disk_used_percent  # just after calculating it
                    last_disk_free_bytes = free  # after calling shutil.disk_usage

                    print(f"🧠 Disk Threat: {last_threat} | Used: {disk_used_percent:.2f}% | RAM: {last_ram_usage:.1f}%")

                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f"[{iteration_count}] Threat: {last_threat} | RAM: {last_ram_usage:.1f}% | Disk: {disk_used_percent:.2f}%\n")

                    if iteration_count == disk_check_interval and last_threat == "HIGH":
                        print("⚠️ Initial disk overflow risk high. Continuing cautiously.")

                    if last_threat == "HIGH" and disk_used_percent >= critical_disk_usage_percent:
                        print("🛑 Critical disk usage reached. Stopping further model generation.")
                        exit()

                    # Check RAM after model finishes
                    if last_ram_usage > ram_threshold_percent:
                        print("⚠️ High RAM usage detected. Pausing...")
                        time.sleep(30)
                        while get_memory_usage() > ram_threshold_percent:
                            print("🔁 Still high. Waiting...")
                            time.sleep(30)

                # --- Save Simulation Progress Info ---
                current_iteration = iteration_count
                percentage_complete = (current_iteration / total_iterations) * 100
                avg_time_per_iteration = pbar.format_dict['elapsed'] / current_iteration if current_iteration > 0 else 0
                approx_time_remaining = (total_iterations - current_iteration) * avg_time_per_iteration

                progress_info = {
                    'current_iteration': current_iteration,
                    'total_iterations': total_iterations,
                    'percentage_complete': percentage_complete,
                    'avg_time_per_iteration': avg_time_per_iteration,
                    'approx_time_remaining': approx_time_remaining,
                    'ram_usage_percent': last_ram_usage,
                    'disk_threat_level': last_threat,
                    'disk_used_percent': last_disk_used_percent,
                    'disk_free': round(last_disk_free_bytes / (1024 ** 3), 2)
                }

                with open(log_file_path, 'w') as log_file:
                    log_file.write(f"Iteration {current_iteration}/{total_iterations}: {progress_info}\n")

                if current_iteration % 100 == 0:
                    upload_to_replit("/Users/ainsleylewis/Documents/Astronomy/Discord Bot/simulation_log.txt")

                pbar.update(1)
