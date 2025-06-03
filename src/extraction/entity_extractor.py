import spacy


class EntityExtractor:
    def __init__(self, config):
        self.nlp = spacy.load(config["spacy"]["model"])

        self.similarity_threshold = 0.65
        self.hub_threshold_multiplier = 0.82
        self.min_hub_frequency = 3

        self.all_entities = {}
        self.page_entities = {}
        self.hub_entities = set()

    def extract_entities(self, page_title, sentences):

        # Step 1: Get potential entities using spaCy
        potential_entities = self._get_noun_phrases(sentences)

        # Step 2: Score entities using spaCy similarity to page title
        key_entities = self._score_and_select_entities(
            page_title, potential_entities, max_entities=5
        )

        # Step 3: Store entities for relationship building
        self._store_entities(page_title, key_entities)

        return key_entities

    def _get_noun_phrases(self, sentences):

        potential_entities = {}

        for sentence in sentences:
            doc = self.nlp(sentence)

            for chunk in doc.noun_chunks:
                text = chunk.text.strip().lower()
                if len(text) > 3 and text.replace(' ', '').isalpha():
                    potential_entities[text] = chunk

            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]:
                    text = ent.text.strip().lower()
                    potential_entities[text] = ent

        return potential_entities

    def _score_and_select_entities(self, page_title, potential_entities, max_entities=5):

        if not potential_entities:
            return []

        title_doc = self.nlp(page_title)

        scored_entities = []

        for entity_text, entity_span in potential_entities.items():
            try:
                similarity = title_doc.similarity(entity_span)

                title_bonus = 0.3 if entity_text in page_title.lower() else 0

                # Word frequency penalty (common words get lower scores)
                word_penalty = self._calculate_word_frequency_penalty(entity_text)

                final_score = similarity + title_bonus - word_penalty

                scored_entities.append({
                    'name': entity_text,
                    'similarity': final_score
                })
            except:
                continue

        scored_entities.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_entities[:max_entities]

    def _calculate_word_frequency_penalty(self, text):
        """Penalize very common words"""
        common_words = {
            'people', 'time', 'way', 'day', 'man', 'world', 'life', 'hand',
            'part', 'child', 'place', 'work', 'week', 'case', 'point', 'company'
        }

        words = text.split()
        penalty = sum(0.1 for word in words if word in common_words)
        return min(penalty, 0.3)

    def _store_entities(self, page_title, entities):
        page_entity_names = []

        for entity in entities:
            entity_name = entity['name']

            if len(entity_name.split()) > 5:
                continue
            if len(entity_name) < 3:
                continue

            entity_doc = self.nlp(entity_name)

            if entity_doc.has_vector and entity_doc.vector_norm > 0:
                page_entity_names.append(entity_name)

                if entity_name not in self.all_entities:
                    self.all_entities[entity_name] = {
                        'pages': [],
                        'doc': entity_doc
                    }
                self.all_entities[entity_name]['pages'].append(page_title)

    def build_relationships(self):

        self._identify_hub_entities()

        relationships = {'links_to': self._build_cross_page_links()}

        return relationships

    def _identify_hub_entities(self):

        entity_frequencies = {}

        for entity_name, entity_data in self.all_entities.items():
            frequency = len(entity_data['pages'])
            entity_frequencies[entity_name] = frequency

            if frequency >= self.min_hub_frequency:
                self.hub_entities.add(entity_name)

        print(f"Identified {len(self.hub_entities)} hub entities from {len(self.all_entities)} total entities")

        top_hubs = sorted(
            [(name, entity_frequencies[name]) for name in self.hub_entities],
            key=lambda x: x[1], reverse=True
        )[:10]

        print("Top hub entities:")
        for name, freq in top_hubs:
            print(f"  {name}: {freq} pages")

    def _build_cross_page_links(self):

        links = []
        entity_names = list(self.all_entities.keys())

        print(f"Building relationships between {len(entity_names)} entities...")

        relationship_count = 0
        hub_relationship_count = 0

        for i, entity1 in enumerate(entity_names):
            entity1_data = self.all_entities[entity1]

            for entity2 in entity_names[i + 1:]:
                entity2_data = self.all_entities[entity2]

                if set(entity1_data['pages']) & set(entity2_data['pages']):
                    continue

                try:
                    similarity = entity1_data['doc'].similarity(entity2_data['doc'])

                    is_hub_relationship = (entity1 in self.hub_entities or
                                           entity2 in self.hub_entities)

                    threshold = self.similarity_threshold
                    if is_hub_relationship:
                        threshold *= self.hub_threshold_multiplier

                    if similarity >= threshold:
                        links.append({
                            'from': entity1,
                            'to': entity2,
                            'similarity': float(similarity),
                            'involves_hub': is_hub_relationship,
                            'pages_1': entity1_data['pages'],
                            'pages_2': entity2_data['pages']
                        })

                        relationship_count += 1
                        if is_hub_relationship:
                            hub_relationship_count += 1

                except:
                    continue

        print(f"Created {relationship_count} total relationships")
        if relationship_count == 0:
            print("No relationships found. Check entity extraction settings.")
            return []

        return links
