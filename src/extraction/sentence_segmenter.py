import spacy


class SentenceSegmenter:
    def __init__(self, config):
        self.nlp = spacy.load(config["spacy"]["model"])

    def segment(self, text):

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        filtered_sentences = [s for s in sentences if len(s) >= 15]

        return filtered_sentences
