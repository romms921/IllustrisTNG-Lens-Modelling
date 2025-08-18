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
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Header, Footer, Static, ProgressBar, Log
from textual.drivers.web_driver import WebDriver

# ==== Your Original Config and Helper Functions ====
# (I've included them here for completeness, no changes needed to this part)
m = [round(x, 5) for x in np.linspace(0.01, 0.5, 50)]
n = [round(x, 4) for x in np.linspace(0, 360, 10)]
o = [round(x, 5) for x in np.linspace(-0.5, 0.5, 10)]

ram_threshold_percent = 90
disk_check_interval = 100
critical_disk_usage_percent = 90
CHUNK_SIZE = 1000

model_output_dir = '/Volumes/T7 Shield/Sim 7'
log_file_path = '/Users/ainsleylewis/Documents/Astronomy/Discord Bot/simulation_log.txt'

restart_file_path = os.path.join(os.path.dirname(os.path.abspath(log_file_path)), 'simulation_restart_state.json')


def get_memory_usage():
    return psutil.virtual_memory().percent

def get_dir_size(directory):
    total_size = 0
    if not os.path.exists(directory):
        return total_size
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


def upload_to_replit(log_path: str, replit_url: str = "https://fd07c8f5-4e98-4ab1-92c3-e95ee5cf45d1-00-h61aopcz3tjm.janeway.replit.dev/upload"):
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

def save_restart_state(path, i, j, k, chunk_number):
    state = {'i': i, 'j': j, 'k': k, 'chunk_number': chunk_number}
    with open(path, 'w') as f:
        json.dump(state, f, indent=4)

def load_restart_state(path):
    if not os.path.exists(path):
        return 0, 0, 0, 1
    try:
        with open(path, 'r') as f:
            state = json.load(f)
            i = state.get('i', 0)
            j = state.get('j', 0)
            k = state.get('k', 0)
            chunk_number = state.get('chunk_number', 1)
            return i, j, k, chunk_number
    except (json.JSONDecodeError, KeyError):
        return 0, 0, 0, 1

def get_csv_filename(chunk_number):
    base_path = '/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Test/'
    os.makedirs(base_path, exist_ok=True)
    sim_name = os.path.basename(model_output_dir)
    if chunk_number == 1:
        return f"{base_path}{sim_name}_summary.csv"
    else:
        return f"{base_path}{sim_name}_summary_{chunk_number}.csv"

def save_to_csv(df, chunk_number):
    csv_file = get_csv_filename(chunk_number)
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)

def calculate_chunk_number(iteration_count):
    return ((iteration_count - 1) // CHUNK_SIZE) + 1

pow_params = ['$z_{s,fid}$', 'x', 'y', 'e', '$θ_{e}$', '$r_{Ein}$', '$\gamma$ (PWI)']
sie_params = ['$\sigma$', 'x', 'y', 'e', '$θ_{e}$', '$r_{core}$', 'NaN']
nfw_params = ['M', 'x', 'y', 'e', '$θ_{e}$', 'c or $r_{s}$', 'NaN']
ein_params = ['M', 'x', 'y', 'e', '$θ_{e}$', 'c or $r_{s}$', r'$\alpha_{e}$']
shear_params = ['$z_{s,fid}$', 'x', 'y', '$\gamma$', '$θ_{\gamma}$', 'NaN', '$\kappa$']
sersic_params = ['$M_{tot}$', 'x', 'y', 'e', '$θ_{e}$', '$r_{e}$', '$n$']
cored_sie_params = ['M', 'x', 'y', 'e', '$θ_{e}$', '$r_{core}$', 'NaN']
mpole_params = ['$z_{s,fid}$', 'x', 'y', '$\epsilon$', '$θ_{m}$', 'm', 'n']

model_list = ['POW', 'SIE', 'ANFW', 'EIN', 'PERT', 'SERS', 'MPOLE']
model_params = {
    'POW': pow_params, 'SIE': sie_params, 'ANFW': nfw_params, 'EIN': ein_params,
    'PERT': shear_params, 'SERS': sersic_params, 'MPOLE': mpole_params
}

def rms_extract(model_ver, model_path, constraint):
    opt_file_path = os.path.join(model_path, f'{model_ver}_optresult.dat')
    if not os.path.exists(opt_file_path):
        return 0, 0, [], 0
    with open(opt_file_path, 'r') as file:
        opt_result = file.readlines()

    last_optimize_index = -1
    for idx, line in reversed(list(enumerate(opt_result))):
        if 'optimize' in line:
            last_optimize_index = idx
            break
    if last_optimize_index == -1: return 0, 0, [], 0
    opt_result = opt_result[last_optimize_index + 1:]

    lens_params_dict = {}
    for line in opt_result:
        if line.startswith('lens'):
            parts = re.split(r'\s+', line.strip())
            lens_name, params = parts[1], [float(x) for x in parts[2:]]
            lens_params_dict[lens_name] = params[1:]

    chi2_line = next((line for line in opt_result if 'chi^2' in line), None)
    chi2_value = float(chi2_line.split('=')[-1].strip().split()[0]) if chi2_line else 0

    num_lens_profiles = len(lens_params_dict)
    rows = []
    max_param_len = 0
    for lens_name, params in lens_params_dict.items():
        row = {'Lens Name': lens_name}
        for i, val in enumerate(params):
            row[f'param{i+1}'] = val
        rows.append(row)
        max_param_len = max(max_param_len, len(params))
    columns = ['Lens Name'] + [f'param{i+1}' for i in range(max_param_len)]
    df = pd.DataFrame(rows, columns=columns)

    dfs = []
    for i in range(num_lens_profiles):
        lens_name = df['Lens Name'].iloc[i]
        model_type = next((m for m in model_list if m.lower() == lens_name.lower()), None)
        if model_type:
            symbols = model_params[model_type][:7]
            row_output = pd.DataFrame([df.iloc[i, 1:8].values], columns=symbols)
            lens_df = pd.concat([row_output.assign(Type='Output')], ignore_index=True)
            lens_df.insert(0, 'Lens Name', lens_name)
            cols = lens_df.columns.tolist()
            cols.insert(1, cols.pop(cols.index('Type')))
            lens_df = lens_df[cols]
            dfs.append(lens_df)

    obs_point_path = 'obs_point/obs_point_(POS+FLUX).dat'
    out_point_path = os.path.join(model_path, f'{model_ver}_point.dat')
    if not os.path.exists(obs_point_path) or not os.path.exists(out_point_path):
        return 0, 0, dfs, chi2_value

    column_names = ['x', 'y', 'mag', 'pos_err', 'mag_err', '1', '2', '3']
    obs_point = pd.read_csv(obs_point_path, delim_whitespace=True, header=None, skiprows=1, names=column_names)
    out_point = pd.read_csv(out_point_path, delim_whitespace=True, header=None, skiprows=1, names=column_names)
    out_point.drop(columns=['mag_err', '1', '2', '3'], inplace=True)
    mask = abs(out_point['mag']) >= 1
    out_point = out_point[mask[:len(out_point)]].reset_index(drop=True)
    out_point['x_diff'] = abs(out_point['x'] - obs_point['x'])
    out_point['y_diff'] = abs(out_point['y'] - obs_point['y'])
    out_point['mag_diff'] = abs(abs(out_point['mag']) - abs(obs_point['mag']))
    out_point['pos_sq'] = np.sqrt((out_point['x_diff']**2 + out_point['y_diff']**2).astype(float))
    pos_rms = np.average(out_point['pos_sq'])
    mag_rms = np.average(np.sqrt((out_point['mag_diff']**2).astype(float)))
    return pos_rms, mag_rms, dfs, chi2_value

# ==== Textual GUI Application ====

class SimulationStatus(Static):
    """A widget to display simulation statistics."""
    def on_mount(self) -> None:
        self.update_stats({}) # Render initial empty state

    def update_stats(self, new_stats: dict) -> None:
        self.stats = new_stats
        self.update(self.render())

    def render(self) -> str:
        # Provide default values to prevent errors before the first iteration
        m_val = self.stats.get('m', 'N/A')
        n_val = self.stats.get('n', 'N/A')
        o_val = self.stats.get('o', 'N/A')
        pos_rms = self.stats.get('pos_rms', 0.0)
        mag_rms = self.stats.get('mag_rms', 0.0)
        chi2 = self.stats.get('chi2', 0.0)

        return (
            f"[bold #888888]Current Parameters[/]\n"
            f"  m: [bold cyan]{m_val}[/]\n"
            f"  n: [bold cyan]{n_val}[/]\n"
            f"  o: [bold cyan]{o_val}[/]\n\n"
            f"[bold #888888]Latest Results[/]\n"
            f"  Position RMS : [bold magenta]{pos_rms:.4f}[/]\n"
            f"  Magnitude RMS: [bold magenta]{mag_rms:.4f}[/]\n"
            f"  Chi²         : [bold magenta]{chi2:.3f}[/]"
        )

class SystemMonitor(Static):
    """A widget to display system monitoring information."""
    def on_mount(self) -> None:
        self.update_clocks()
        self.set_interval(2, self.update_clocks)

    def get_usage_style(self, percent: float) -> str:
        if percent < 70: return "green"
        if percent < 90: return "yellow"
        return "red"

    def update_clocks(self) -> None:
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        cpu_style = self.get_usage_style(cpu_percent)
        ram_style = self.get_usage_style(ram_percent)
        
        cpu_bar = "█" * int(cpu_percent / 4) + " " * (25 - int(cpu_percent / 4))

        self.update(
            f"Wall Clock : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"CPU Usage  : [bold {cpu_style}][{cpu_bar}] {cpu_percent: >3}%[/]\n"
            f"RAM Usage  : [bold {ram_style}]{ram_percent: >3}%[/]"
        )

class SimulationApp(App):
    CSS_PATH = "style.css"
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode"), ("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        total_iters = len(m) * len(n) * len(o)
        yield Header()
        with Container(id="main_container"):
            with Vertical(id="left_panel"):
                yield Static("📊 Simulation Progress", classes="header")
                yield ProgressBar(total=total_iters, show_eta=True, id="progress")
                yield Static(f"0 / {total_iters}", id="iteration_counter")
                yield SimulationStatus()
            with Vertical(id="right_panel"):
                yield Static("⚙️ System Monitor", classes="header")
                yield SystemMonitor()
                yield Static("📜 Live Log", classes="header")
                yield Log(id="log", max_lines=100, highlight=True)
        yield Footer()
        
    def on_mount(self) -> None:
        self.run_worker(self.run_simulation, thread=True)

    def run_simulation(self):
        log = self.query_one(Log)
        total_iterations = len(m) * len(n) * len(o)
        log.write("[bold cyan]Simulation starting...[/]")
        
        start_i, start_j, start_k, chunk_number = load_restart_state(restart_file_path)
        log.write(f"[cyan]Resuming from indices (i={start_i}, j={start_j}, k={start_k})[/]")
        
        iterations_done = start_i * len(n) * len(o) + start_j * len(o) + start_k
        iteration_count = iterations_done

        if iterations_done > 0: start_k += 1
        os.makedirs(model_output_dir, exist_ok=True)
        self.query_one(ProgressBar).progress = iteration_count

        try:
            for i in range(start_i, len(m)):
                j_start = start_j if i == start_i else 0
                for j in range(j_start, len(n)):
                    k_start = start_k if i == start_i and j == j_start else 0
                    for k in range(k_start, len(o)):
                        model_name = f'POW_POS_SHEAR_{m[i]}_{n[j]}_{o[k]}'
                        log.write(f"Running iteration [bold yellow]{iteration_count+1}[/]...")
                        
                        # --- Model Generation ---
                        # ** FIX HERE: Pass the DIRECTORY to glafic.init **
                        glafic.init(0.3, 0.7, -1.0, 0.7, model_output_dir, 20.0, 20.0, 21.56, 21.56, 0.01, 0.01, 1, verb=0)
                        glafic.set_secondary('chi2_splane 1', verb=0)
                        glafic.set_secondary('chi2_checknimg 1', verb=0)
                        glafic.set_secondary('chi2_restart   -1', verb=0)
                        glafic.set_secondary('chi2_usemag    1', verb=0)
                        glafic.set_secondary('hvary          0', verb=0)
                        glafic.set_secondary('ran_seed -122000', verb=0)
                        glafic.startup_setnum(2, 0, 1)
                        glafic.set_lens(1, 'pow', 0.261343256161012, 1.0, 20.78, 20.78, 0.107, 23.38, 0.46, 2.1)
                        glafic.set_lens(2, 'pert', 0.261343256161012, 1.0, 20.78, 20.78, m[i], n[j], 0.0, o[k])
                        glafic.set_point(1, 1.0, 20.78, 20.78)
                        glafic.setopt_lens(1, 0, 0, 1, 1, 1, 1, 1, 1)
                        glafic.setopt_lens(2, 0, 0, 0, 0, 1, 1, 0, 1)
                        glafic.setopt_point(1, 0, 1, 1)
                        glafic.model_init(verb=0)
                        glafic.readobs_point('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/obs_point/obs_point_(POS).dat')
                        glafic.parprior('/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Test/Simulation/priorfile.dat')
                        glafic.optimize()
                        glafic.findimg()
                        glafic.writecrit(1.0)
                        glafic.writelens(1.0)
                        glafic.quit()
                        
                        # --- Data Extraction ---
                        # ** FIX HERE: Pass the correct directory and model name **
                        pos_rms, mag_rms, dfs, chi2 = rms_extract(model_name, model_output_dir, 'pos')
                        
                        self.call_from_thread(self.update_ui, {
                            'm': m[i], 'n': n[j], 'o': o[k], 
                            'pos_rms': pos_rms, 'mag_rms': mag_rms, 'chi2': chi2,
                            'iteration': iteration_count + 1, 'total': total_iterations
                        })

                        file_name = model_path + '_point.dat'

                        # Calculate current chunk number based on iteration count
                        current_chunk = calculate_chunk_number(iteration_count + 1)

                        if os.path.exists(file_name):
                            data = pd.read_csv(file_name, delim_whitespace=True, skiprows=1, header=None, names=columns)
                            num_images = len(data)
                            
                            # Create the result dataframe
                            result_df = pd.DataFrame({
                                'strength': [m[i]],
                                'pa': [n[j]],
                                'kappa': [o[k]],
                                'num_images': [num_images],
                                'pos_rms': [pos_rms],
                                'mag_rms': [mag_rms],
                                't_shear_str': [dfs[1]['$\gamma$'][1]],
                                't_shear_pa': [dfs[1]['$θ_{\gamma}$'][1]],
                                't_shear_kappa': [dfs[1]['$\kappa$'][1]],
                                'chi2': [chi2],
                                **{col: [dfs[0][col][1]] for col in macro_columns}
                            })
                            
                            if data.empty:
                                print(f"File {file_name} is empty.")
                                # Override with zeros for empty data
                                result_df = pd.DataFrame({
                                    'strength': [m[i]],
                                    'pa': [n[j]],
                                    'kappa': [o[k]],
                                    'num_images': [0],
                                    'pos_rms': [0],
                                    'mag_rms': [0], 
                                    't_shear_str': [0],
                                    't_shear_pa': [0],
                                    't_shear_kappa': [0],
                                    'chi2': [0],
                                    **{col: [0] for col in macro_columns}
                                })
                            else:
                                print(f"File {file_name} exists and is not empty.")
                                
                            # Save to the appropriate CSV chunk
                            save_to_csv(result_df, current_chunk)
                            
                            # Delete generated files to save space (only if data is not empty)
                            # Define Files 
                            print(f"Deleting files for model: {model_name}")
                            crit_file = model_path + '_crit.dat'  
                            lens_file = model_path + '_lens.fits'
                            point_file = model_path + '_point.dat'
                            opt_file = model_path + '_optresult.dat'

                            # Delete Files
                            for file_to_delete in [crit_file, lens_file, point_file, opt_file]:
                                if os.path.exists(file_to_delete):
                                    os.remove(file_to_delete)
                                        
                        else:
                            print(f"File {file_name} does not exist.")
                        
                        iteration_count += 1
                        if iteration_count % 100 == 0:
                            save_restart_state(restart_file_path, i, j, k, chunk_number)
                            log.write("[bold green]Checkpoint saved.[/]")

                    start_k = 0
                start_j = 0
        except Exception as e:
            log.write(f"[bold red]FATAL ERROR: {e}[/]")
            if 'i' in locals():
                save_restart_state(restart_file_path, i, j, k, chunk_number)
                log.write("[bold yellow]Saved final state before exiting.[/]")

        log.write("\n[bold green]🎉 Simulation completed successfully![/]")
        if os.path.exists(restart_file_path):
            os.remove(restart_file_path)
            log.write(f"[green]🗑️ Removed restart file.[/]")

    def update_ui(self, stats: dict):
        self.query_one(ProgressBar).progress = stats['iteration']
        self.query_one("#iteration_counter").update(f"{stats['iteration']} / {stats['total']}")
        self.query_one(SimulationStatus).update_stats(stats)


if __name__ == "__main__":
    app = SimulationApp()
    app.run()