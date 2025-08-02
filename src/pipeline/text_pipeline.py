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
        print("\nStarting text processing pipeline...")
        page_data = []
        counter = 0
        directory_length = len(os.listdir(self.text_input_dir))
        for file in os.listdir(self.text_input_dir):
            try:
                with open(os.path.join(self.text_input_dir, file), 'r') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading file {file}: {e}")

            page_title, clean_text = self.text_cleaner.clean_text(text)
            sentences = self.sentence_segmenter.segment(clean_text)

            entities = self.entity_extractor.extract_entities(sentences)
            page_data.append({"title": page_title, "entities": entities})
            print(f"{counter}/{directory_length}")
            counter += 1

        print("\nBuilding entity relationships...")
        return self.entity_extractor.build_relationships(page_data)
