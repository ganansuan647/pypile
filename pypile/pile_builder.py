from pypile.result_renderer import ResultRenderer

class PileBuilder:
    def __init__(self, config_data):
        self.config_data = config_data
        self.pile_model = None

    def initialize_pile_properties(self):
        # Initialize pile properties from config data
        self.pile_model = {
            "pile_groups": [],
            "load_cases": []
        }
        for pile_group in self.config_data.get("pile_groups", []):
            self.pile_model["pile_groups"].append(self._initialize_pile_group(pile_group))
        for load_case in self.config_data.get("load_cases", []):
            self.pile_model["load_cases"].append(self._initialize_load_case(load_case))

    def _initialize_pile_group(self, pile_group):
        # Initialize a single pile group using get to avoid KeyError
        return {
            "type": pile_group.get("type"),
            "coordinates": pile_group.get("coordinates"),
            "properties": pile_group.get("properties", {})
        }

    def _initialize_load_case(self, load_case):
        # Initialize a single load case using get to avoid KeyError
        return {
            "node": load_case.get("node"),
            "forces": load_case.get("forces")
        }

    def validate_pile_properties(self):
        # Validate pile properties
        if not self.pile_model:
            raise ValueError("Pile model is not initialized.")
        for pile_group in self.pile_model["pile_groups"]:
            if "type" not in pile_group or "coordinates" not in pile_group or "properties" not in pile_group:
                raise ValueError("Invalid pile group configuration.")
        for load_case in self.pile_model["load_cases"]:
            if "node" not in load_case or "forces" not in load_case:
                raise ValueError("Invalid load case configuration.")

    def build_pile_segments(self):
        # Build pile segments based on configuration
        for pile_group in self.pile_model["pile_groups"]:
            pile_group["segments"] = []
            for segment in pile_group["properties"].get("segments", []):
                pile_group["segments"].append(self._build_segment(segment))

    def _build_segment(self, segment):
        # Build a single pile segment using get to avoid KeyError
        return {
            "depth": segment.get("depth"),
            "soil_type": segment.get("soil_type")
        }

    def export_pile_model(self, file_path):
        # Export the pile model to a JSON file
        import json
        with open(file_path, 'w') as file:
            json.dump(self.pile_model, file, indent=4)

    def get_result_renderer(self, analysis_results):
        return ResultRenderer(analysis_results)
