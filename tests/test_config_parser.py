import unittest
from pypile.config_parser import ConfigParser

class TestConfigParser(unittest.TestCase):

    def test_yaml_parsing(self):
        config_parser = ConfigParser()
        config_parser.parse('tests/test_files/config.yaml')
        data = config_parser.get_data()
        print("YAML Parsed Data:", data)
        self.assertIsNotNone(data)

    def test_json_parsing(self):
        config_parser = ConfigParser()
        config_parser.parse('tests/test_files/config.json')
        data = config_parser.get_data()
        print("JSON Parsed Data:", data)
        self.assertIsNotNone(data)

    def test_text_parsing(self):
        config_parser = ConfigParser()
        config_parser.parse('tests/test_files/config.txt')
        data = config_parser.get_data()
        print("Text Parsed Data:", data)
        self.assertIsNotNone(data)

if __name__ == '__main__':
    unittest.main()
