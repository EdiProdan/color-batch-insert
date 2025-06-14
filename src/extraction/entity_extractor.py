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
        for sentence in sentences:

            doc = self.nlp(sentence)

            # 1. Enhanced NER filtering
            for ent in doc.ents:
                if ent.label_ in {"ORG", "GPE", "LOC", "PRODUCT", "LAW", "EVENT", "WORK_OF_ART"} and len(ent.text) > 3:
                    if not any(c.isdigit() for c in ent.text):  # Exclude pure numbers
                        entities.add(ent.text.strip())

            # 2. Smart noun chunk filtering
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.strip()
                # Conditions for exclusion
                if (len(chunk_text) > 3 and
                        chunk.root.pos_ in {"NOUN", "PROPN"} and
                        not chunk.root.is_stop and
                        not any(tok.like_num for tok in chunk) and
                        chunk.root.lemma_ not in {"thing", "something", "example", "reason", "use"}):

                    # 3. Contextual filtering (avoid phrases like "no need")
                    if not (chunk[0].lower_ == "no" and len(chunk) > 1):
                        entities.add(chunk_text)

        # 4. Post-processing filters
        filtered_entities = {
            ent for ent in entities
            if not (ent.startswith(('a ', 'an ', 'the ')) or  # Remove articles
                    ent.lower() in {"issn", "isbn", "doi"})  # Keep important acronyms
        }

        return filtered_entities


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
