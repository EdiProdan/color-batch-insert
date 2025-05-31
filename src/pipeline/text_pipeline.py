import os

from src.extraction import SentenceSegmenter, TextCleaner


class TextPipeline:

    def __init__(self, config):
        self.text_input_dir = config["data"]["input"]["text_dir"]
        self.text_cleaner = TextCleaner()
        self.sentence_segmenter = SentenceSegmenter(config)

    def process(self):
        for file in os.listdir(self.text_input_dir):
            try:
                with open(os.path.join(self.text_input_dir, file), 'r') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading file {file}: {e}")

            page_title, clean_text = self.text_cleaner.clean_text(text)
            sentences = self.sentence_segmenter.segment(clean_text)

            print(f"File {file}: {len(sentences)} sentences")
            for i, sentence in enumerate(sentences):
                print(f"Sentence {i + 1}: {sentence}")
