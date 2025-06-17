import spacy
import re
from collections import defaultdict, Counter
import json


class EntityExtractor:
    """
    A general-purpose entity extractor for building knowledge graphs from diverse text sources.
    Optimized for extracting meaningful entities from news articles, biographical texts, etc.
    """

    def __init__(self, config):
        # Load spaCy model
        self.config = config
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except:
            print("Please install spacy model: python -m spacy download en_core_web_md")
            self.nlp = None

    def extract_entities(self, sentences):
        entities = set()
        generic_lemmas = {"thing", "stuff", "example", "reason", "issue", "use", "area", "part", "way", "number"}
        discard_lower = {"issn", "isbn", "doi"}

        for sentence in sentences:
            doc = self.nlp(sentence)

            # 1. High-confidence NER
            for ent in doc.ents:
                text = ent.text.strip()
                if (ent.label_ in {"ORG", "GPE", "LOC", "PRODUCT", "LAW", "EVENT", "WORK_OF_ART"} and
                        len(text) > 3 and
                        not any(c.isdigit() for c in text)):

                    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)
                    if text.lower() not in discard_lower and not re.search(r'https?://|\b\d{4,}\b', text):
                        entities.add(text)

            # 2. Noun chunks (fallback if not already extracted)
            for chunk in doc.noun_chunks:
                text = chunk.text.strip()
                root = chunk.root

                if (len(text) > 3 and
                        root.pos_ in {"NOUN", "PROPN"} and
                        not root.is_stop and
                        root.lemma_ not in generic_lemmas and
                        not any(tok.like_num for tok in chunk) and
                        not (chunk[0].lower_ == "no" and len(chunk) > 1)):

                    text = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)

                    # Semantic heuristics
                    if (text.lower() not in discard_lower and
                            not re.search(r'https?://|\b\d{4,}\b', text) and
                            len(text.split()) <= 4 and
                            not text.lower().startswith(("many ", "some ", "few ", "this ", "that ")) and
                            root.lemma_ not in generic_lemmas and
                            all(tok.pos_ not in {"DET", "PRON", "CCONJ", "SCONJ"} for tok in chunk)):

                        # Avoid duplication with higher-confidence NER
                        if text not in entities:
                            entities.add(text)

        return {
            ent for ent in entities
            if 3 < len(ent) <= 100 and not ent.lower().startswith(("a ", "an ", "the "))
        }

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
            print(text)
            doc = self.nlp(text)
            scores = defaultdict(float)

            # Compare against each category's reference terms
            for category, terms in categories.items():
                category_doc = self.nlp(" ".join(terms))
                scores[category] = doc.similarity(category_doc)

            # Return top scoring category
            print(text, max(scores.items(), key=lambda x: x[1]))
