import os
import json
from pathlib import Path

from src.extraction.image_classifier import ImageClassifier


class ImagePipeline:

    def __init__(self, config):
        self.image_input_dir = config["data"]["input"]["images_dir"]
        self.output_dir = config["data"]["output"]
        self.image_classifier = ImageClassifier(config)
        self.processed_images = []

    def process(self):
        print("\nStarting image processing pipeline...")
        image_data = []

        image_files = []
        if os.path.exists(self.image_input_dir):
            for file in os.listdir(self.image_input_dir):
                file_path = os.path.join(self.image_input_dir, file)
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(file_path)

        for image_path in image_files:
            try:
                filename = os.path.basename(image_path)
                print(f"\nProcessing file: {filename}")

                image_title = self._extract_title_from_filename(filename)

                entities = self.image_classifier.classify_image(image_path)
                print(f"Found {len(entities)} entities")
                print(f"Entities: {entities}")

                image_data.append({"title": image_title, "entities": list(entities)})
            except Exception as e:
                print(f"Error processing image {filename}: {e}")

        print(f"\nBuilding entity relationships...")
        return self.image_classifier.build_relationships(image_data)

    def _extract_title_from_filename(self, filename):
        name_without_ext = Path(filename).stem
        parts = name_without_ext.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            parts = parts[:-1]
        title = ' '.join(parts)
        return title if title else name_without_ext
