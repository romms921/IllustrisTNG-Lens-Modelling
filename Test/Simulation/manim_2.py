from manim import *
import pandas as pd
import numpy as np

class LensAnimation(Scene):
    def construct(self):
        # Constants and setup
        const_x = 28
        base_folder = '/Users/ainsleylewis/Documents/Astronomy/IllustrisTNG Lens Modelling/Test/Shear Shift/'
        shear_str = [0.0, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020,
                     0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.030,
                     0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040]

        # Create axes
        axes = Axes(
            x_range=[200, 1400],
            y_range=[200, 1400],
            axis_config={"include_tip": False},
        ).scale(0.5)  # Scale to fit screen

        # Create title
        title = Text("Shear = 0.0").to_edge(UP)
        self.add(axes, title)

        for i, shear in enumerate(shear_str):
            # Load and process data (same as your original code)
            columns = ['x_crit', 'y_crit', 'x_caustic', 'y_caustic', 'x_crit_0', 'y_crit_0', 'x_caustic_0', 'y_caustic_0']
            crit_curve = pd.read_csv(f'{base_folder}SIE_POS_EXTEND_{shear}_crit.dat', 
                                   delim_whitespace=True, header=None, skiprows=0, names=columns)
            source_points = pd.read_csv(f'{base_folder}SIE_POS_EXTEND_{shear}_point.dat', 
                                      delim_whitespace=True, header=None)
            
            # Process data (same transformations as your original code)
            image_indices = source_points[source_points[1] == 1.0].index
            sources = source_points.iloc[image_indices]
            sources.columns = ['num_images', 'z', 'x', 'y']
            
            out_indices = ~source_points.index.isin(image_indices)
            out_df = source_points[out_indices]
            out_df.columns = ['x', 'y', 'mag', 'pos_err']

            # Transform coordinates
            for df in [sources, out_df, crit_curve]:
                for col in df.columns:
                    if col.startswith(('x', 'y')):
                        df[col] = (df[col] - 20)/0.001

            # Create Manim objects
            source_dots = VGroup(*[Dot(axes.c2p(x, y), color=BLUE, radius=0.05) 
                                 for x, y in zip(sources['x'], sources['y'])])
            
            output_dots = VGroup(*[Dot(axes.c2p(x, y), color=RED, radius=0.03) 
                                 for x, y in zip(out_df['x'], out_df['y'])])
            
            crit_curve_dots = VGroup(*[Dot(axes.c2p(x, y), color=WHITE, radius=0.01) 
                                     for x, y in zip(crit_curve['x_crit'], crit_curve['y_crit'])])
            
            caustic_dots = VGroup(*[Dot(axes.c2p(x, y), color=WHITE, radius=0.01) 
                                  for x, y in zip(crit_curve['x_caustic'], crit_curve['y_caustic'])])

            new_title = Text(f"Shear = {shear}").to_edge(UP)

            # Animation
            if i == 0:
                self.play(
                    Create(source_dots),
                    Create(output_dots),
                    Create(crit_curve_dots),
                    Create(caustic_dots)
                )
            else:
                self.play(
                    Transform(old_source_dots, source_dots),
                    Transform(old_output_dots, output_dots),
                    Transform(old_crit_dots, crit_curve_dots),
                    Transform(old_caustic_dots, caustic_dots),
                    Transform(title, new_title)
                )

            # Store current dots for next transformation
            old_source_dots = source_dots
            old_output_dots = output_dots
            old_crit_dots = crit_curve_dots
            old_caustic_dots = caustic_dots
            
            # Wait a bit before next frame
            self.wait(0.5)

