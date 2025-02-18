import __init__ # Required to import the module

import unittest
import numpy as np
from scipy.sparse import lil_matrix
from pypile.nonlinear_solver import NonlinearSolver
import os

class TestNonlinearSolver(unittest.TestCase):

    def setUp(self):
        self.pile_model = {
            "nodes": [0, 1, 2],
            "elements": [
                {"nodes": [0, 1], "length": 5, "area": 1.2, "youngs_modulus": 2e6},
                {"nodes": [1, 2], "length": 10, "area": 1.2, "youngs_modulus": 2e6}
            ],
            "boundary_conditions": [
                {"node": 0, "value": 0},
                {"node": 2, "value": 0}
            ]
        }
        self.solver = NonlinearSolver(self.pile_model)

    def test_assemble_stiffness_matrix(self):
        self.solver.assemble_stiffness_matrix()
        self.assertIsInstance(self.solver.stiffness_matrix, lil_matrix)
        self.assertEqual(self.solver.stiffness_matrix.shape, (3, 3))

    def test_solve_control_equations(self):
        self.solver.assemble_stiffness_matrix()
        self.solver.load_vector = np.array([0, 1000, 0])
        
        # 分离渲染逻辑
        self.solver.solve_equations_only()  # 新方法，只进行求解
        
        self.assertIsNotNone(self.solver.displacement_vector)
        self.assertEqual(len(self.solver.displacement_vector), 3)

    # 添加单独的渲染测试
    def test_rendering(self):
        # 只在需要测试渲染时运行
        if os.environ.get('TEST_RENDERING'):
            self.solver.assemble_stiffness_matrix()
            self.solver.load_vector = np.array([0, 1000, 0])
            self.solver.solve_control_equations()  # 包含渲染

    def test_apply_boundary_conditions(self):
        self.solver.assemble_stiffness_matrix()
        self.solver.load_vector = np.array([0, 1000, 0])
        self.solver.apply_boundary_conditions()
        self.assertEqual(self.solver.stiffness_matrix[0, 0], 1)
        self.assertEqual(self.solver.stiffness_matrix[2, 2], 1)
        self.assertEqual(self.solver.load_vector[0], 0)
        self.assertEqual(self.solver.load_vector[2], 0)

if __name__ == '__main__':
    unittest.main()
