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

    def _extract_key_concepts(self, doc, min_frequency=2):
        """
        Extract important concepts based on frequency and importance.
        These are noun phrases that might not be recognized as named entities.
        """
        # Count noun chunks
        noun_chunks = []
        chunk_freq = Counter()

        for chunk in doc.noun_chunks:
            # Clean and normalize the chunk
            chunk_text = chunk.text.strip().lower()

            # Filter criteria:
            # - Not too short (at least 2 characters)
            # - Not too long (max 5 words)
            # - Not just determiners or pronouns
            word_count = len(chunk_text.split())
            if 1 <= word_count <= 5 and len(chunk_text) > 2:
                if not all(token.pos_ in ['DET', 'PRON'] for token in chunk):
                    chunk_freq[chunk.text] += 1
                    noun_chunks.append({
                        'text': chunk.text,
                        'root': chunk.root.text,
                        'start': chunk.start_char,
                        'end': chunk.end_char
                    })

        # Return concepts that appear multiple times or are particularly important
        important_concepts = []
        for chunk_text, freq in chunk_freq.items():
            if freq >= min_frequency:
                # Find the first occurrence for position info
                for chunk in noun_chunks:
                    if chunk['text'] == chunk_text:
                        chunk['frequency'] = freq
                        important_concepts.append(chunk)
                        break

        return important_concepts

    def _find_entity_relationships(self, doc, window_size=100):
        """
        Find relationships between entities based on proximity in text.
        Entities appearing close together likely have some relationship.
        """
        relationships = []
        entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

        # Find entities that appear within 'window_size' characters of each other
        for i, (text1, label1, start1, end1) in enumerate(entities):
            for j, (text2, label2, start2, end2) in enumerate(entities[i + 1:], i + 1):
                distance = abs(start2 - end1)
                if distance <= window_size:
                    relationships.append({
                        'entity1': text1,
                        'entity2': text2,
                        'distance': distance,
                        'labels': f"{label1}-{label2}"
                    })

        return relationships

    def _create_graph_structure(self, doc_title, entities, entity_relationships):
        """Create the graph structure with links_to relationships."""
        graph = {
            'document': {
                'title': doc_title,
                'entity_summary': {
                    'people': len(entities['people']),
                    'organizations': len(entities['organizations']),
                    'locations': len(entities['locations']),
                    'dates': len(entities['dates']),
                    'events': len(entities['events']),
                    'key_concepts': len(entities['concepts'])
                }
            },
            'links_to': [],
            'entity_relationships': entity_relationships[:20]  # Top 20 closest relationships
        }

        # Create links_to relationships for all meaningful entities

        # Add people (most important for biographical texts)
        for person in entities['people']:
            graph['links_to'].append({
                'from': doc_title,
                'to': person['text'],
                'type': 'mentions_person',
                'entity_label': person['label']
            })

        # Add organizations
        for org in entities['organizations']:
            graph['links_to'].append({
                'from': doc_title,
                'to': org['text'],
                'type': 'mentions_organization',
                'entity_label': org['label']
            })

        # Add locations
        for loc in entities['locations']:
            graph['links_to'].append({
                'from': doc_title,
                'to': loc['text'],
                'type': 'mentions_location',
                'entity_label': loc['label']
            })

        # Add important events
        for event in entities['events']:
            graph['links_to'].append({
                'from': doc_title,
                'to': event['text'],
                'type': 'mentions_event',
                'entity_label': event['label']
            })

        # Add high-frequency concepts
        for concept in entities['concepts']:
            if concept['frequency'] >= 3:  # Only very frequent concepts
                graph['links_to'].append({
                    'from': doc_title,
                    'to': concept['text'],
                    'type': 'discusses_concept',
                    'frequency': concept['frequency']
                })

        # Store detailed entities for analysis
        graph['detailed_entities'] = entities

        return graph

    def process_multiple_files(self, file_paths):
        """Process multiple files and aggregate results."""
        all_results = {}

        for file_path in file_paths:
            doc_title = file_path.split('/')[-1].split('.')[0]
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                results = self.extract_entities(text, doc_title)
                all_results[doc_title] = results
            except Exception as e:
                all_results[doc_title] = {"error": str(e)}

        return all_results

    def find_cross_document_entities(self, all_results):
        """Find entities that appear across multiple documents."""
        entity_documents = defaultdict(list)

        for doc_title, results in all_results.items():
            if 'links_to' in results:
                for link in results['links_to']:
                    entity = link['to']
                    entity_documents[entity].append(doc_title)

        # Find entities mentioned in multiple documents
        cross_document_entities = {
            entity: docs
            for entity, docs in entity_documents.items()
            if len(docs) > 1
        }

        return cross_document_entities

    def print_summary(self, results):
        """Print a user-friendly summary of extracted entities."""
        if 'error' in results:
            print(f"Error: {results['error']}")
            return

        print(f"\n{'=' * 60}")
        print(f"DOCUMENT: {results['document']['title']}")
        print(f"{'=' * 60}\n")

        print("ENTITY SUMMARY:")
        for entity_type, count in results['document']['entity_summary'].items():
            if count > 0:
                print(f"  • {entity_type.replace('_', ' ').title()}: {count}")

        print(f"\nTOTAL RELATIONSHIPS: {len(results['links_to'])}")

        # Show most important entities by category
        if results['links_to']:
            print("\nKEY ENTITIES FOUND:")

            # Group by type
            by_type = defaultdict(list)
            for link in results['links_to']:
                by_type[link['type']].append(link['to'])

            for link_type, entities in by_type.items():
                if entities:
                    print(f"\n  {link_type.replace('_', ' ').upper()}:")
                    for entity in list(set(entities))[:5]:  # Top 5 unique
                        print(f"    → {entity}")

        # Show entity relationships
        if results['entity_relationships']:
            print("\nCLOSELY RELATED ENTITIES (appearing near each other):")
            for rel in results['entity_relationships'][:5]:
                print(f"  • {rel['entity1']} ↔ {rel['entity2']} (distance: {rel['distance']} chars)")

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
