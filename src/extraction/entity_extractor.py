import spacy
import re
from collections import defaultdict


class EntityExtractor:

    def __init__(self, config):
        self.config = config
        try:
            self.nlp = spacy.load(config["spacy"]["model"])
        except OSError:
            print("Please install spacy model: python -m spacy download en_core_web_md")
            self.nlp = None

    def extract_entities(self, sentences):
        entities = set()
        discard_lower = {"issn", "isbn", "doi"}

        for sentence in sentences:
            doc = self.nlp(sentence)

            for ent in doc.ents:
                text = ent.text.strip()
                if (ent.label_ in {"ORG", "GPE", "LOC", "PRODUCT", "LAW", "EVENT", "WORK_OF_ART", "PERSON", "NORP"} and
                        len(text) > 3):

                    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)
                    if text.lower() not in discard_lower and not re.search(r'https?://|\b\d{4,}\b', text):
                        entities.add(text)

        return {ent for ent in entities if 3 < len(ent) <= 100}

    def build_relationships(self, page_data):
        relationships = []
        for page in page_data:
            for entity in page['entities']:
                relationships.append({"from": page['title'], "to": entity})
        return {"LINKS_TO": relationships}

    def classify_entities(self, entities):

        categories = {
            "identifier": ["ISBN", "DOI", "barcode"],
            "publication": ["magazine", "journal", "newspaper"],
            "organization": ["UNESCO", "ISO", "library"],
            "technology": ["website", "database", "microform"],
            "process": ["cataloging", "retrieval", "assignment"]
        }

        for text in entities:
            doc = self.nlp(text)
            scores = defaultdict(float)

            for category, terms in categories.items():
                category_doc = self.nlp(" ".join(terms))
                scores[category] = doc.similarity(category_doc)
