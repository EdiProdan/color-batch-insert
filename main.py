"""
Author: Edi Prodan
Thesis: Intelligent System for Semantic Structuring of Multimodal Data
"""
import yaml
from src.pipeline.controller import PipelineController

if __name__ == "__main__":

    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    print(config)
    controller = PipelineController(config)

    controller.run_pipeline()
