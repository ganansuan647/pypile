import __init__ # Required to import the module

import unittest
from pypile.pile_builder import PileBuilder
from pypile.config_parser import ConfigParser

class TestPileBuilder(unittest.TestCase):

    def setUp(self):
        self.config_parser = ConfigParser()
        self.config_parser.parse('tests/test_files/config.yaml')
        self.config_data = self.config_parser.get_data()
        self.pile_builder = PileBuilder(self.config_data)

    def test_initialize_pile_properties(self):
        self.pile_builder.initialize_pile_properties()
        self.assertIsNotNone(self.pile_builder.pile_model)
        self.assertIn("pile_groups", self.pile_builder.pile_model)
        self.assertIn("load_cases", self.pile_builder.pile_model)

    def test_validate_pile_properties(self):
        self.pile_builder.initialize_pile_properties()
        self.pile_builder.validate_pile_properties()
        # If no exception is raised, the test passes
        self.assertTrue(True)

    def test_build_pile_segments(self):
        self.pile_builder.initialize_pile_properties()
        self.pile_builder.build_pile_segments()
        for pile_group in self.pile_builder.pile_model["pile_groups"]:
            self.assertIn("segments", pile_group)
            self.assertGreater(len(pile_group["segments"]), 0)

if __name__ == '__main__':
    unittest.main()
