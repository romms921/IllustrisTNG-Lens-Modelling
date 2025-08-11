import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['figure.dpi'] = 500
plt.rcParams['text.color'] = 'w'
plt.rcParams['axes.labelcolor'] = 'k'
plt.rcParams['xtick.color'] = 'w'
plt.rcParams['ytick.color'] = 'w'
plt.rcParams['axes.edgecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'k'
plt.rcParams['axes.facecolor'] = 'k'
from scipy.interpolate import griddata
import pandas as pd
import re
from astropy.io import fits
import os
import seaborn as sns
import plotly.graph_objects as go
from scipy.interpolate import griddata

from matplotlib.lines import Line2D
from manim import *
# from manim import config
# config.ffmpeg_executable = "/opt/homebrew/bin/ffmpeg"  # Adjust path as needed

os.chdir("/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling")
df = pd.read_csv('Test/Simulation/sim_3.csv')

class SurfaceMorphAnimation(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=45 * DEGREES, theta=-30 * DEGREES, zoom=0.6)

        # --- DATA PREPARATION (Do this only ONCE) ---
        x = df['t_shear_str'].values
        y = df['t_shear_kappa'].values
        z = df['pos_rms'].values
        
        mask = ~np.isnan(z) & ~np.isnan(x) & ~np.isnan(y)
        x, y, z = x[mask], y[mask], z[mask]
        
        # Define ranges
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        # Create the axes first to define the coordinate system
        axes = ThreeDAxes(
            x_range=[x_min, x_max],
            y_range=[y_min, y_max],
            z_range=[0, z.max() if z.max() > 0 else 1],
            x_length=7, y_length=7, z_length=3
        )
        self.add(axes)

        # --- OPTIMIZATION: PRE-CALCULATE THE FINAL SURFACE ---
        # Instead of calling griddata repeatedly, call it once to get all final z-values.
        u_vals = np.linspace(x_min, x_max, 50)
        v_vals = np.linspace(y_min, y_max, 50)
        final_z_grid = griddata((x, y), z, (u_vals[None, :], v_vals[:, None]), method='cubic')
        final_z_grid = np.nan_to_num(final_z_grid) # Replace any edge-case NaNs with 0

        # --- CREATE START AND END SURFACES ---

        # 1. The Starting Surface (flat)
        start_surface = Surface(
            lambda u, v: axes.c2p(u, v, 0),
            u_range=[x_min, x_max],
            v_range=[y_min, y_max],
            resolution=(10, 10),
            fill_opacity=1,
        )

        start_surface.set_color(GREEN)  # Color the starting surface

        # 2. The Final Surface (morphed)
        end_surface = Surface(
            lambda u, v: axes.c2p(u, v, griddata((x,y), z, (u,v), method='cubic', fill_value=0)),
            u_range=[x_min, x_max],
            v_range=[y_min, y_max],
            resolution=(10, 10),
            fill_opacity=1,
            checkerboard_colors=False
        )

        # Color the final surface based on its Z-values
        end_surface.set_color(BLUE)

        # Add the starting surface to the scene
        self.add(start_surface)
        self.wait(0.5)

        # --- ANIMATE EFFICIENTLY USING TRANSFORM ---
        # Transform will handle the point-by-point interpolation from start to end.
        # It will also interpolate the colors smoothly.
        self.play(
            Transform(start_surface, end_surface),
            run_time=5
        )
        self.wait()
        
        # Add labels
        labels = axes.get_axis_labels(
            x_label="Shear Strength",
            y_label="Shear Kappa",
            z_label="Position RMS"
        )
        self.play(Write(labels))

        # Rotate the view
        self.begin_ambient_camera_rotation(rate=0.4)
        self.wait(20)
        self.stop_ambient_camera_rotation()