import json
import yaml
from pypile.parsers.yaml_parser import YamlParser
from pypile.parsers.json_parser import JsonParser
from pypile.parsers.text_parser import TextParser

class ConfigParser:
    def __init__(self):
        self.data = None

    def parse(self, file_path):
        file_format = self._determine_file_format(file_path)
        if file_format == 'yaml':
            parser = YamlParser()
        elif file_format == 'json':
            parser = JsonParser()
        elif file_format == 'text':
            parser = TextParser()
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        self.data = parser.parse(file_path)

    def _determine_file_format(self, file_path):
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            return 'yaml'
        elif file_path.endswith('.json'):
            return 'json'
        elif file_path.endswith('.txt'):
            return 'text'
        else:
            raise ValueError("Unsupported file format")

    def get_data(self):
        return self.data

    def validate_data(self):
        # Implement validation logic here
        pass
