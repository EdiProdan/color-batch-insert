import os
import json

from src.extraction import SentenceSegmenter, TextCleaner, EntityExtractor


class TextPipeline:

    def __init__(self, config):
        self.text_input_dir = config["data"]["input"]["text_dir"]
        self.output_dir = config["data"]["output"]
        self.text_cleaner = TextCleaner()
        self.sentence_segmenter = SentenceSegmenter(config)
        self.entity_extractor = EntityExtractor(config)
        self.processed_pages = []

    def process(self):

        for file in os.listdir(self.text_input_dir):
            try:
                with open(os.path.join(self.text_input_dir, file), 'r') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading file {file}: {e}")

            page_title, clean_text = self.text_cleaner.clean_text(text)
            sentences = self.sentence_segmenter.segment(clean_text)

            entities = self.entity_extractor.extract_entities(page_title, sentences)

            page_data = {
                'file': file,
                'title': page_title,
                'sentence_count': len(sentences),
                'entities': entities
            }
            self.processed_pages.append(page_data)

            print(f"Processed file: {file}")

        print("\nBuilding entity relationships...")
        relationships = self.entity_extractor.build_relationships()

        self._save_extraction_results(relationships)

    def _save_extraction_results(self, relationships):
        with open(f'{self.output_dir}/scenario_3/processed_pages', 'w') as f:
            json.dump(self.processed_pages, f, indent=2)

        with open(f'{self.output_dir}/scenario_3/relationships.json', 'w') as f:
            json.dump(relationships, f, indent=2)

        print(f"\nResults saved to {self.output_dir}")
