import json
import os
from typing import Dict, List, Optional


class DataIntegrationLayer:

    def __init__(self, config):
        self.config = config
        self.output_dir = config["data"]["output"]

    def merge_relationships(self, text_relationships = None, image_relationships = None):

        if text_relationships is None:
            text_relationships = {"LINKS_TO": []}
        if image_relationships is None:
            image_relationships = {"LINKS_TO": []}

        text_links = text_relationships.get("LINKS_TO", [])
        image_links = image_relationships.get("LINKS_TO", [])

        all_relationships = text_links + image_links

        seen = set()
        deduplicated = []

        for rel in all_relationships:
            key = (rel["from"], rel["to"])
            if key not in seen:
                seen.add(key)
                deduplicated.append(rel)

        return {"LINKS_TO": deduplicated}

    def save_results(self, relationships: Dict):

        os.makedirs(self.output_dir, exist_ok=True)

        output_path = os.path.join(self.output_dir, "s3.json")
        with open(output_path, 'w') as f:
            json.dump(relationships, f, indent=2)

        print(f"Results saved to: {output_path}")
