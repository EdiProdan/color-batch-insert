"""
Author: Edi Prodan
Thesis: Intelligent System for Semantic Structuring of Multimodal Data
"""
import yaml
from src.pipeline.controller import PipelineController

if __name__ == "__main__":
    # Load configuration from YAML file
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    print(config)
    controller = PipelineController(config)

    controller.run_pipeline()
    #input_file = "data/input/text/0001_Made_in_China_2025.txt"
    #results = controller.process_text_file(input_file)
    #print("Processing complete. Results:")
    #print(results)
