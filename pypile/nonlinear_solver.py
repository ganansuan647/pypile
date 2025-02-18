import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from pypile.result_renderer import ResultRenderer

class NonlinearSolver:
    def __init__(self, pile_model):
        self.pile_model = pile_model
        self.stiffness_matrix = None
        self.load_vector = None
        self.displacement_vector = None

    def assemble_stiffness_matrix(self):
        # Assemble the system stiffness matrix
        num_nodes = len(self.pile_model["nodes"])
        self.stiffness_matrix = lil_matrix((num_nodes, num_nodes))
        for element in self.pile_model["elements"]:
            element_stiffness = self._calculate_element_stiffness(element)
            self._add_element_stiffness_to_global_matrix(element, element_stiffness)

    def _calculate_element_stiffness(self, element):
        # Calculate the stiffness matrix for a single element
        length = element["length"]
        area = element["area"]
        youngs_modulus = element["youngs_modulus"]
        stiffness = (youngs_modulus * area) / length
        return np.array([[stiffness, -stiffness], [-stiffness, stiffness]])

    def _add_element_stiffness_to_global_matrix(self, element, element_stiffness):
        # Add the element stiffness matrix to the global stiffness matrix
        node1, node2 = element["nodes"]
        self.stiffness_matrix[node1, node1] += element_stiffness[0, 0]
        self.stiffness_matrix[node1, node2] += element_stiffness[0, 1]
        self.stiffness_matrix[node2, node1] += element_stiffness[1, 0]
        self.stiffness_matrix[node2, node2] += element_stiffness[1, 1]

    def solve_equations_only(self):
        """只进行方程求解，不包含渲染"""
        self.apply_boundary_conditions()
        self.displacement_vector = spsolve(self.stiffness_matrix.tocsc(), self.load_vector)
        print("计算结果:", self.displacement_vector)

    def solve_control_equations(self):
        """完整的求解过程，包含渲染"""
        self.apply_boundary_conditions()
        self.solve_equations_only()
        self.pass_results_to_renderer()

    def apply_boundary_conditions(self):
        # Apply boundary conditions using Lagrange multipliers
        for boundary_condition in self.pile_model["boundary_conditions"]:
            node = boundary_condition["node"]
            value = boundary_condition["value"]
            self.stiffness_matrix[node, :] = 0
            self.stiffness_matrix[node, node] = 1
            self.load_vector[node] = value

    def export_results(self, file_path):
        # Export the results to a JSON file
        import json
        results = {
            "displacement_vector": self.displacement_vector.tolist()
        }
        with open(file_path, 'w') as file:
            json.dump(results, file, indent=4)

    def get_result_renderer(self, analysis_results):
        return ResultRenderer(analysis_results)

    def pass_results_to_renderer(self):
        analysis_results = {
            "dimensions": (10, 10, 10),
            "origin": (0, 0, 0),
            "spacing": (1, 1, 1),
            "deformation": np.random.rand(10, 10, 10),
            "pressure": np.random.rand(10, 10, 10)
        }
        renderer = self.get_result_renderer(analysis_results)
        try:
            renderer.render_deformation_cloud_map()
        except ImportError as err:
            print(f"Rendering skipped due to import error: {err}")
        renderer.render_section_view((5, 5))
        renderer.render_vector_field()
