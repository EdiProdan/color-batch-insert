from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import time


class SemanticAnalyzer:
    """
    Simple semantic analyzer for knowledge graph relationship processing.

    Provides situational awareness by analyzing:
    1. Current database state (existing entity load)
    2. Incoming relationship patterns (semantic types, frequency)
    3. Conflict risk assessment (overlap between busy entities and incoming data)

    This preprocessing enables intelligent batching decisions.
    """

    def __init__(self, driver):
        self.driver = driver

        # Define semantic type patterns based on Wikipedia entities
        self.type_patterns = {
            'identifier': {
                'exact': ['isbn', 'issn', 'imdb', 'doi', 'pmid', 'oclc'],
                'contains': ['identifier', 'id', 'code', 'number'],
            },
            'geographic': {
                'exact': ['united states', 'united kingdom', 'germany', 'france',
                          'china', 'japan', 'india', 'brazil', 'canada', 'australia',
                          'london', 'paris', 'new york', 'tokyo', 'berlin', 'boston',
                          'philadelphia', 'chicago', 'los angeles', 'madrid', 'rome', 'virginia', 'washington'],
                'contains': ['city', 'country', 'state', 'province', 'region',
                             'continent', 'ocean', 'river', 'mountain', 'island',
                             'massachusetts', 'california', 'texas', 'florida', 'ohio'],
                'suffixes': ['burg', 'stadt', 'shire', 'land', 'ville', 'ton', 'ford'],
            },
            'person': {
                'exact': [],
                'contains': ['author', 'writer', 'director', 'actor', 'actress',
                             'scientist', 'professor', 'president', 'minister',
                             'artist', 'musician', 'singer', 'player', 'einstein',
                             'newton', 'curie', 'shakespeare', 'beethoven'],
                'patterns': ['born', 'died'],  # From page context
                'suffixes': [],  # Names don't have consistent suffixes
            },
            'organization': {
                'exact': ['united nations', 'european union', 'nato', 'who'],
                'contains': ['university', 'college', 'institute', 'company',
                             'corporation', 'agency', 'department', 'ministry',
                             'police', 'military', 'army', 'navy', 'force'],
                'suffixes': ['inc', 'corp', 'ltd', 'gmbh', 'org'],
            },
            'creative_work': {
                'exact': [],
                'contains': ['film', 'movie', 'book', 'novel', 'album', 'song',
                             'series', 'show', 'documentary', 'magazine', 'journal'],
                'patterns': ['published', 'released', 'premiered'],
            },
            'event': {
                'exact': [],
                'contains': ['war', 'battle', 'revolution', 'election', 'olympics',
                             'conference', 'summit', 'festival', 'ceremony',
                             'attack', 'disaster', 'crisis', 'pandemic'],
                'patterns': ['occurred', 'happened', 'took place'],
            },
            'concept': {
                'exact': [],
                'contains': ['theory', 'law', 'principle', 'philosophy', 'religion',
                             'language', 'culture', 'science', 'technology', 'system',
                             'mechanics', 'physics', 'chemistry', 'biology', 'mathematics', 'climate', 'environment'],
            },
            'temporal': {
                'exact': [],
                'contains': ['century', 'decade', 'year', 'month', 'period', 'era'],
                'patterns': ['january', 'february', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                             'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'
                             ],
            },
            'media': {
                'exact': ['the new york times', 'the guardian', 'bbc', 'cnn'],
                'contains': ['news', 'media', 'press', 'journal', 'magazine']
            },
            'academic': {
                'contains': ['professor', 'research', 'study', 'journal'],
                'suffixes': ['phd', 'md', 'research']
            },

        }

    def analyze_preprocessing_context(self, relationships: List[Dict]) -> Dict:
        print("🔍 Starting semantic preprocessing analysis...")
        analysis_start = time.time()

        # Part 1: Analyze current database state
        print("  📊 Analyzing current database state...")
        database_state = self._analyze_database_state()

        # Part 2: Analyze incoming relationships
        print("  📋 Analyzing incoming relationships...")
        incoming_analysis = self._analyze_incoming_relationships(relationships)

        # Part 3: Assess conflict risks
        print("  ⚠️  Assessing conflict risks...")
        conflict_assessment = self._assess_conflict_risks(database_state, incoming_analysis)

        analysis_time = time.time() - analysis_start
        print(f"  ✅ Preprocessing completed in {analysis_time:.2f} seconds")

        return {
            'database_state': database_state,
            'incoming_analysis': incoming_analysis,
            'conflict_assessment': conflict_assessment,
            'preprocessing_time': analysis_time
        }

    def _analyze_database_state(self) -> Dict:
        """Analyze current database to understand existing entity load."""
        try:
            with self.driver.session() as session:
                # Get entity degree distribution
                result = session.run("""
                    MATCH (n:Entity)
                    WHERE n.title IS NOT NULL
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, count(r) as degree
                    RETURN n.title as entity_title, degree
                    ORDER BY degree DESC
                    LIMIT 20
                """)

                high_degree_entities = []
                for record in result:
                    entity_title = record['entity_title']
                    if entity_title is not None:  # Additional safety check
                        high_degree_entities.append({
                            'name': entity_title,  # Keep as 'name' for consistency in the rest of the code
                            'degree': record['degree']
                        })

                # Get basic database stats - also filter null titles
                stats_result = session.run("""
                    MATCH (n:Entity)
                    WHERE n.title IS NOT NULL
                    OPTIONAL MATCH (n)-[r:LINKS_TO]-()
                    RETURN count(DISTINCT n) as total_entities, count(r) as total_relationships
                """)

                stats = stats_result.single()

                return {
                    'high_degree_entities': high_degree_entities,
                    'total_entities': stats['total_entities'] if stats else 0,
                    'total_relationships': stats['total_relationships'] if stats else 0,
                    'top_entities': [e['name'] for e in high_degree_entities[:10] if e['name'] is not None]
                }

        except Exception as e:
            print(f"    Warning: Database analysis failed: {e}")
            return {
                'high_degree_entities': [],
                'total_entities': 0,
                'total_relationships': 0,
                'top_entities': []
            }

    def _analyze_incoming_relationships(self, relationships: List[Dict]) -> Dict:
        """Analyze semantic patterns in incoming relationship data."""

        all_entities = set()
        entity_frequency = Counter()

        for rel in relationships:
            # Add null safety for relationship data
            from_entity = rel.get('from', '').strip() if rel.get('from') else ''
            to_entity = rel.get('to', '').strip() if rel.get('to') else ''

            # Skip empty or null entities
            if from_entity:
                from_entity_lower = from_entity.lower()
                all_entities.add(from_entity_lower)
                entity_frequency[from_entity_lower] += 1

            if to_entity:
                to_entity_lower = to_entity.lower()
                all_entities.add(to_entity_lower)
                entity_frequency[to_entity_lower] += 1

        # Classify entities by semantic type
        entity_classifications = {}
        type_counts = Counter()

        for entity in all_entities:
            semantic_type = self._classify_entity(entity)
            entity_classifications[entity] = semantic_type
            type_counts[semantic_type] += 1

        # Find frequent entities in incoming data (potential internal conflicts)
        frequent_entities = [(entity, count) for entity, count in entity_frequency.most_common(10)]

        return {
            'total_unique_entities': len(all_entities),
            'entity_classifications': entity_classifications,
            'type_distribution': dict(type_counts),
            'frequent_incoming_entities': frequent_entities,
            'classification_coverage': (len(all_entities) - type_counts['unknown']) / len(
                all_entities) if all_entities else 0
        }

    def _classify_entity(self, entity_name: str) -> str:

        entity_lower = entity_name.lower().strip()

        for semantic_type, patterns in self.type_patterns.items():
            if 'exact' in patterns and entity_lower in patterns['exact']:
                return semantic_type

            if 'contains' in patterns:
                for pattern in patterns['contains']:
                    if pattern in entity_lower:
                        return semantic_type

            if 'suffixes' in patterns:
                for suffix in patterns['suffixes']:
                    if entity_lower.endswith(suffix):
                        return semantic_type

        return 'unknown'

    def _assess_conflict_risks(self, database_state: Dict, incoming_analysis: Dict) -> Dict:

        database_busy_entities = set()
        for entity_info in database_state['high_degree_entities']:
            if entity_info.get('name') is not None:
                database_busy_entities.add(entity_info['name'].lower())

        incoming_entities = set(incoming_analysis['entity_classifications'].keys())

        high_risk_entities = database_busy_entities.intersection(incoming_entities)

        # Assess semantic type conflict patterns
        type_conflict_risks = self._calculate_type_conflict_risks(incoming_analysis['type_distribution'])

        # Entity-level risk assessment
        entity_risks = {}
        for entity, classification in incoming_analysis['entity_classifications'].items():
            risk_score = 0.0

            # Higher risk if entity is already busy in database
            if entity in database_busy_entities:
                risk_score += 0.6

            # Higher risk for identifier types (often hubs)
            if classification == 'identifier':
                risk_score += 0.3

            # Higher risk for frequent entities in incoming data
            frequent_entities = dict(incoming_analysis['frequent_incoming_entities'])
            if entity in frequent_entities and frequent_entities[entity] > 5:
                risk_score += 0.2

            entity_risks[entity] = min(risk_score, 1.0)  # Cap at 1.0

        return {
            'high_risk_entities': list(high_risk_entities),
            'type_conflict_risks': type_conflict_risks,
            'entity_risk_scores': entity_risks,
            'risk_summary': {
                'total_high_risk': len([e for e, risk in entity_risks.items() if risk > 0.5]),
                'overlap_with_database': len(high_risk_entities)
            }
        }

    def _calculate_type_conflict_risks(self, type_distribution: Dict) -> Dict:
        """Calculate conflict risk for each semantic type based on simple heuristics."""
        type_risks = {}

        # Simple heuristic model for conflict prediction
        risk_weights = {
            'identifier': 0.8,  # High risk - often hubs
            'geographic': 0.4,  # Medium risk - some popular places
            'organization': 0.5,  # Medium risk - universities can be hubs
            'person': 0.3,  # Lower risk - usually not major hubs
            'concept': 0.4,  # Medium risk - some popular concepts
            'unknown': 0.2  # Low risk - hard to predict
        }

        for semantic_type, count in type_distribution.items():
            base_risk = risk_weights.get(semantic_type, 0.2)
            # Increase risk if many entities of this type
            volume_multiplier = min(1.0 + (count / 100), 2.0)  # Cap multiplier
            type_risks[semantic_type] = min(base_risk * volume_multiplier, 1.0)

        return type_risks

    def print_analysis_summary(self, analysis_context: Dict):
        """Print a human-readable summary of the analysis for debugging/research."""
        db_state = analysis_context['database_state']
        incoming = analysis_context['incoming_analysis']
        conflicts = analysis_context['conflict_assessment']

        print(f"\n📈 SEMANTIC ANALYSIS SUMMARY")
        print(f"{'=' * 50}")

        print(f"\n🏗️  Database State:")
        print(f"  Total entities: {db_state['total_entities']}")
        print(f"  Total relationships: {db_state['total_relationships']}")
        print(f"  Top busy entities: {', '.join(db_state['top_entities'][:5])}")

        print(f"\n📦 Incoming Data:")
        print(f"  Unique entities to process: {incoming['total_unique_entities']}")
        print(f"  Classification coverage: {incoming['classification_coverage']:.1%}")
        print(f"  Type distribution: {incoming['type_distribution']}")

        print(f"\n⚠️  Conflict Assessment:")
        print(f"  High-risk entities: {conflicts['risk_summary']['total_high_risk']}")
        print(f"  Database overlap: {conflicts['risk_summary']['overlap_with_database']}")
        print(
            f"  Riskiest types: {sorted(conflicts['type_conflict_risks'].items(), key=lambda x: x[1], reverse=True)[:3]}")

        print(f"\n⏱️  Preprocessing time: {analysis_context['preprocessing_time']:.2f}s")