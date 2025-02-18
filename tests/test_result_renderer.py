import unittest
import numpy as np
from pypile.result_renderer import ResultRenderer

class TestResultRenderer(unittest.TestCase):

    def setUp(self):
        self.analysis_results = {
            "dimensions": (10, 10, 10),
            "origin": (0, 0, 0),
            "spacing": (1, 1, 1),
            "deformation": np.random.rand(10, 10, 10),
            "pressure": np.random.rand(10, 10, 10)
        }
        self.renderer = ResultRenderer(self.analysis_results)

    def test_render_deformation_cloud_map(self):
        try:
            self.renderer.render_deformation_cloud_map()
        except Exception as e:
            self.fail(f"render_deformation_cloud_map raised an exception: {e}")

    def test_render_section_view(self):
        try:
            self.renderer.render_section_view((5, 5))
        except Exception as e:
            self.fail(f"render_section_view raised an exception: {e}")

    def test_render_vector_field(self):
        try:
            self.renderer.render_vector_field()
        except Exception as e:
            self.fail(f"render_vector_field raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
