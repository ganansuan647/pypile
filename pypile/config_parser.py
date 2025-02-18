import json
import yaml
from pypile.result_renderer import ResultRenderer

class ConfigParser:
    def __init__(self):
        self.data = None
    
    def parse(self, file_path):
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                self.data = yaml.safe_load(f)
            elif file_path.endswith('.json'):
                self.data = json.load(f)
            else:
                self.data = f.read()
    
    def get_data(self):
        return self.data

    def get_result_renderer(self, analysis_results):
        return ResultRenderer(analysis_results)
