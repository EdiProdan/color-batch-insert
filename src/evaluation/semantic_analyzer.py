"""
Semantic Analyzer for Knowledge Graph Construction
Classifies entities and predicts conflicts based on semantic patterns
"""
import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter


class SemanticAnalyzer:
    """
    Analyzes entities from Wikipedia-based knowledge graphs to:
    1. Classify entity types based on patterns and context
    2. Predict conflict probability between entity pairs
    3. Provide semantic grouping recommendations
    """

    def __init__(self):
        # Define type patterns based on Wikipedia entities
        self.type_patterns = {
            'identifier': {
                'exact': ['isbn', 'issn', 'imdb', 'doi', 'pmid', 'oclc'],
                'contains': ['identifier', 'id', 'code', 'number'],
                'priority': 10  # Highest priority
            },
            'geographic': {
                'exact': ['united states', 'united kingdom', 'germany', 'france',
                          'china', 'japan', 'india', 'brazil', 'canada', 'australia',
                          'london', 'paris', 'new york', 'tokyo', 'berlin', 'boston',
                          'philadelphia', 'chicago', 'los angeles', 'madrid', 'rome'],
                'contains': ['city', 'country', 'state', 'province', 'region',
                            'continent', 'ocean', 'river', 'mountain', 'island',
                            'massachusetts', 'california', 'texas', 'florida', 'ohio'],
                'suffixes': ['burg', 'stadt', 'shire', 'land', 'ville', 'ton', 'ford'],
                'priority': 8
            },
            'person': {
                'exact': [],
                'contains': ['author', 'writer', 'director', 'actor', 'actress',
                            'scientist', 'professor', 'president', 'minister',
                            'artist', 'musician', 'singer', 'player', 'einstein',
                            'newton', 'curie', 'shakespeare', 'beethoven'],
                'patterns': ['born', 'died'],  # From page context
                'suffixes': [],  # Names don't have consistent suffixes
                'priority': 7
            },
            'organization': {
                'exact': ['united nations', 'european union', 'nato', 'who'],
                'contains': ['university', 'college', 'institute', 'company',
                            'corporation', 'agency', 'department', 'ministry',
                            'police', 'military', 'army', 'navy', 'force'],
                'suffixes': ['inc', 'corp', 'ltd', 'gmbh', 'org'],
                'priority': 6
            },
            'creative_work': {
                'exact': [],
                'contains': ['film', 'movie', 'book', 'novel', 'album', 'song',
                            'series', 'show', 'documentary', 'magazine', 'journal'],
                'patterns': ['published', 'released', 'premiered'],
                'priority': 5
            },
            'event': {
                'exact': [],
                'contains': ['war', 'battle', 'revolution', 'election', 'olympics',
                            'conference', 'summit', 'festival', 'ceremony',
                            'attack', 'disaster', 'crisis', 'pandemic'],
                'patterns': ['occurred', 'happened', 'took place'],
                'priority': 4
            },
            'concept': {
                'exact': [],
                'contains': ['theory', 'law', 'principle', 'philosophy', 'religion',
                            'language', 'culture', 'science', 'technology', 'system',
                            'mechanics', 'physics', 'chemistry', 'biology', 'mathematics'],
                'priority': 3
            },
            'temporal': {
                'exact': [],
                'contains': ['century', 'decade', 'year', 'month', 'period', 'era'],
                'patterns': ['january', 'february', 'monday', 'tuesday'],
                'priority': 2
            }
        }

        # Known high-conflict entities from your research

        # Initialize conflict learning
        self.observed_conflicts = defaultdict(int)
        self.type_pair_conflicts = defaultdict(int)
        self.type_pair_attempts = defaultdict(int)

    def classify_entity(self, entity_name: str, page_title: str = None,
                       page_content: str = None) -> Tuple[str, float]:
        """
        Classify an entity based on its name and context.
        Returns: (entity_type, confidence_score)
        """
        entity_lower = entity_name.lower().strip()

        # Special handling for known database hubs first

        # Check each type by priority
        best_type = 'general'
        best_score = 0.0

        for entity_type, patterns in self.type_patterns.items():
            score = 0.0

            # Exact match check
            if entity_lower in patterns.get('exact', []):
                score = 1.0

            # Contains check
            for pattern in patterns.get('contains', []):
                if pattern in entity_lower:
                    score = max(score, 0.8)
                    break

            # Suffix check
            for suffix in patterns.get('suffixes', []):
                if entity_lower.endswith(suffix):
                    score = max(score, 0.7)
                    break

            # Context from page title
            if page_title:
                title_lower = page_title.lower()
                for pattern in patterns.get('contains', []):
                    if pattern in title_lower:
                        score = max(score, 0.6)
                        break

                # Check patterns in title
                for pattern in patterns.get('patterns', []):
                    if pattern in title_lower:
                        score = max(score, 0.5)
                        break

            # Apply priority weighting
            weighted_score = score * (patterns['priority'] / 10.0)

            if weighted_score > best_score:
                best_score = weighted_score
                best_type = entity_type

        # Special handling for known entities

        return best_type, best_score

    def predict_conflict_probability(self, entity1: str, entity2: str,
                                    entity1_type: str = None, entity2_type: str = None,
                                    entity1_degree: int = 0, entity2_degree: int = 0,
                                    involves_hub: bool = False) -> float:
        """
        Predict the probability of conflict when processing these entities together.
        """
        # Get types if not provided
        if not entity1_type:
            entity1_type, _ = self.classify_entity(entity1)
        if not entity2_type:
            entity2_type, _ = self.classify_entity(entity2)

        # Check if either is a known hub
        entity1_lower = entity1.lower().strip()
        entity2_lower = entity2.lower().strip()


        # Type-based conflict rules
        conflict_matrix = {
            ('identifier', 'identifier'): 0.8,
            ('identifier', 'any'): 0.7,
            ('geographic', 'geographic'): 0.3,
            ('person', 'person'): 0.4,
            ('organization', 'organization'): 0.4,
            ('creative_work', 'creative_work'): 0.3,
            ('event', 'event'): 0.2,
            ('concept', 'concept'): 0.2,
            ('temporal', 'temporal'): 0.1,
            ('different_types'): 0.15  # Default for different types
        }

        # Look up conflict probability
        type_pair = (entity1_type, entity2_type)
        reverse_pair = (entity2_type, entity1_type)

        if type_pair in conflict_matrix:
            base_prob = conflict_matrix[type_pair]
        elif reverse_pair in conflict_matrix:
            base_prob = conflict_matrix[reverse_pair]
        elif entity1_type == entity2_type:
            base_prob = 0.25  # Same type default
        else:
            base_prob = conflict_matrix['different_types']

        # Adjust for degree (hub detection)
        if entity1_degree > 100 or entity2_degree > 100:
            base_prob = min(base_prob + 0.3, 0.95)
        elif entity1_degree > 50 or entity2_degree > 50:
            base_prob = min(base_prob + 0.15, 0.95)

        # Explicit hub flag
        if involves_hub:
            base_prob = max(base_prob, 0.6)

        # Learn from observed conflicts (if available)
        type_key = f"{min(entity1_type, entity2_type)}-{max(entity1_type, entity2_type)}"
        if self.type_pair_attempts[type_key] > 10:  # Enough observations
            observed_rate = self.type_pair_conflicts[type_key] / self.type_pair_attempts[type_key]
            # Blend predicted and observed (weighted average)
            base_prob = 0.7 * base_prob + 0.3 * observed_rate

        return min(max(base_prob, 0.0), 1.0)  # Ensure between 0 and 1

    def enrich_relationships(self, relationships: List[Dict],
                           processed_pages: List[Dict] = None) -> List[Dict]:
        """
        Add semantic information to relationships.
        """
        # Build entity -> page mapping if pages provided
        entity_to_page = {}
        if processed_pages:
            for page in processed_pages:
                page_title = page.get('title', '')
                for entity in page.get('entities', []):
                    entity_to_page[entity['name']] = page_title

        enriched = []
        type_distribution = Counter()
        type_pair_distribution = Counter()

        for rel in relationships:
            # Get entity types
            from_name = rel['from']
            to_name = rel['to']

            from_page = entity_to_page.get(from_name, '')
            to_page = entity_to_page.get(to_name, '')

            from_type, from_conf = self.classify_entity(from_name, from_page)
            to_type, to_conf = self.classify_entity(to_name, to_page)

            # Count for statistics
            type_distribution[from_type] += 1
            type_distribution[to_type] += 1
            type_pair = f"{from_type}-{to_type}"
            type_pair_distribution[type_pair] += 1

            # Create enriched relationship
            enriched_rel = rel.copy()
            enriched_rel['from_type'] = from_type
            enriched_rel['to_type'] = to_type
            enriched_rel['from_type_confidence'] = from_conf
            enriched_rel['to_type_confidence'] = to_conf
            enriched_rel['type_pair'] = type_pair

            # Predict conflict
            enriched_rel['predicted_conflict'] = self.predict_conflict_probability(
                from_name, to_name,
                from_type, to_type,
                entity1_degree=len(rel.get('pages_1', [])),
                entity2_degree=len(rel.get('pages_2', [])),
                involves_hub=rel.get('involves_hub', False)
            )

            enriched.append(enriched_rel)

        return enriched, type_distribution, type_pair_distribution

    def update_conflict_observations(self, entity1_type: str, entity2_type: str,
                                   had_conflict: bool):
        """
        Update conflict statistics based on observed behavior.
        """
        type_key = f"{min(entity1_type, entity2_type)}-{max(entity1_type, entity2_type)}"
        self.type_pair_attempts[type_key] += 1
        if had_conflict:
            self.type_pair_conflicts[type_key] += 1

    def analyze_hub_types(self, relationships: List[Dict],
                         processed_pages: List[Dict] = None) -> Dict:
        """
        Distinguish between topical hubs (frequent in pages) and
        structural hubs (frequent in database connections).
        """
        # Count entity frequencies in relationships (database connections)
        db_frequency = Counter()
        for rel in relationships:
            db_frequency[rel['from']] += 1
            db_frequency[rel['to']] += 1

        # Count entity frequencies in pages (topical mentions)
        page_frequency = Counter()
        if processed_pages:
            for page in processed_pages:
                for entity in page.get('entities', []):
                    page_frequency[entity['name']] += 1

        # Classify hubs
        hub_analysis = {
            'structural_hubs': [],  # High DB frequency, low page frequency
            'topical_hubs': [],     # High page frequency
            'universal_hubs': [],   # High in both
            'regular_entities': []  # Low in both
        }

        # Get all unique entities
        all_entities = set(db_frequency.keys()) | set(page_frequency.keys())

        for entity in all_entities:
            db_freq = db_frequency.get(entity, 0)
            page_freq = page_frequency.get(entity, 0)

            # Classify based on frequencies (adjust thresholds as needed)
            if db_freq > 50 and page_freq < 10:
                hub_analysis['structural_hubs'].append({
                    'entity': entity,
                    'db_frequency': db_freq,
                    'page_frequency': page_freq,
                    'type': self.classify_entity(entity)[0]
                })
            elif page_freq > 20 and db_freq < 10:
                hub_analysis['topical_hubs'].append({
                    'entity': entity,
                    'db_frequency': db_freq,
                    'page_frequency': page_freq,
                    'type': self.classify_entity(entity)[0]
                })
            elif db_freq > 50 and page_freq > 20:
                hub_analysis['universal_hubs'].append({
                    'entity': entity,
                    'db_frequency': db_freq,
                    'page_frequency': page_freq,
                    'type': self.classify_entity(entity)[0]
                })

        # Sort by frequency
        for hub_type in ['structural_hubs', 'topical_hubs', 'universal_hubs']:
            hub_analysis[hub_type].sort(
                key=lambda x: x['db_frequency'] + x['page_frequency'],
                reverse=True
            )

        return hub_analysis


    def get_semantic_statistics(self, enriched_relationships):
        """
        Calculate statistics about the semantic distribution.
        """
        stats = {
            'total_relationships': len(enriched_relationships),
            'type_distribution': Counter(),
            'type_pair_distribution': Counter(),
            'conflict_predictions': {
                'high': 0,  # > 0.7
                'medium': 0,  # 0.3 - 0.7
                'low': 0  # < 0.3
            },
            'average_conflict_probability': 0.0
        }

        total_conflict_prob = 0.0

        for rel in enriched_relationships:
            # Type distribution
            stats['type_distribution'][rel['from_type']] += 1
            stats['type_distribution'][rel['to_type']] += 1
            stats['type_pair_distribution'][rel['type_pair']] += 1

            # Conflict prediction distribution
            conf_prob = rel['predicted_conflict']
            total_conflict_prob += conf_prob

            if conf_prob > 0.7:
                stats['conflict_predictions']['high'] += 1
            elif conf_prob >= 0.3:
                stats['conflict_predictions']['medium'] += 1
            else:
                stats['conflict_predictions']['low'] += 1

        stats['average_conflict_probability'] = total_conflict_prob / len(enriched_relationships)

        return stats
