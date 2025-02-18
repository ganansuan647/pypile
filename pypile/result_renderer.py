import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

class ResultRenderer:
    def __init__(self, analysis_results):
        self.analysis_results = analysis_results

    def render_deformation_cloud_map(self):
        # Render deformation cloud map using PyVista
        grid = pv.UniformGrid()
        grid.dimensions = self.analysis_results["dimensions"]
        grid.origin = self.analysis_results["origin"]
        grid.spacing = self.analysis_results["spacing"]
        grid.point_data["deformation"] = self.analysis_results["deformation"]
        plotter = pv.Plotter()
        plotter.add_mesh(grid, scalars="deformation", cmap="viridis")
        plotter.show()

    def render_section_view(self, section_plane):
        # Render section view using Matplotlib
        deformation = self.analysis_results["deformation"]
        section_data = deformation[section_plane]
        plt.imshow(section_data, cmap="viridis")
        plt.colorbar(label="Deformation")
        plt.title("Section View")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def render_vector_field(self):
        # Render vector field for soil pressure distribution using PyVista
        grid = pv.UniformGrid()
        grid.dimensions = self.analysis_results["dimensions"]
        grid.origin = self.analysis_results["origin"]
        grid.spacing = self.analysis_results["spacing"]
        grid.point_data["pressure"] = self.analysis_results["pressure"]
        plotter = pv.Plotter()
        plotter.add_mesh(grid, scalars="pressure", cmap="plasma")
        plotter.show()
