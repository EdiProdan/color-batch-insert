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

        image_data = []

        # Get all image files from directory
        image_files = []
        if os.path.exists(self.image_input_dir):
            for file in os.listdir(self.image_input_dir):
                file_path = os.path.join(self.image_input_dir, file)
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(file_path)

        # Process each image file
        for image_path in image_files:
            try:
                filename = os.path.basename(image_path)
                print(f"\nProcessing file: {filename}")

                # Extract title from filename (remove number suffix and replace _ with space)
                image_title = self._extract_title_from_filename(filename)

                # Extract entities from image
                entities = self.image_classifier.classify_image(image_path)
                print(f"Found {len(entities)} entities")
                print(f"Entities: {entities}")

                image_data.append({"title": image_title, "entities": list(entities)})
                print(image_data)
            except Exception as e:
                print(f"Error processing image {filename}: {e}")

        exit(0)
        print(f"\nBuilding entity relationships...")
        relationships = self.image_classifier.build_relationships(image_data)

        print(f"Built entity relationships: {len(relationships.get('LINKS_TO', []))}")
        self._save_extraction_results(relationships)

    def _extract_title_from_filename(self, filename):
        """Extract title from filename: BBC_News_0.jpg -> BBC News"""
        # Remove file extension
        name_without_ext = Path(filename).stem

        # Split by underscore
        parts = name_without_ext.split('_')

        # Remove last part if it's a number
        if len(parts) > 1 and parts[-1].isdigit():
            parts = parts[:-1]

        # Join with spaces
        title = ' '.join(parts)
        return title if title else name_without_ext

    def _save_extraction_results(self, relationships):

        with open(f'{self.output_dir}/image_relationships.json', 'w') as f:
            json.dump(relationships, f, indent=2)

        print(f"\nResults saved to {self.output_dir}")