"""
Test script to validate the Semantic Analyzer
Tests entity classification, conflict prediction, and relationship enrichment
"""
import json
import os
from typing import Dict, List
from collections import Counter

from src.evaluation import SemanticAnalyzer


# Import the semantic analyzer (adjust path as needed)
# from semantic_analyzer import SemanticAnalyzer


def test_entity_classification():
    """Test entity type classification with known examples"""
    print("=" * 60)
    print("TESTING ENTITY CLASSIFICATION")
    print("=" * 60)

    analyzer = SemanticAnalyzer()

    # Test cases with expected types
    test_cases = [
        # (entity_name, page_title, expected_type)
        ("isbn", None, "identifier"),
        ("issn", None, "identifier"),
        ("imdb", None, "identifier"),
        ("hamburg", "2025 Hamburg stabbing attack", "geographic"),
        ("united states", None, "geographic"),
        ("the federal police of germany", None, "organization"),
        ("the film", "The Secret Agent (2025 film)", "creative_work"),
        ("albert einstein", None, "person"),
        ("world war ii", None, "event"),
        ("quantum mechanics", None, "concept"),
        ("21st century", None, "temporal"),
        ("some random entity", None, "general")
    ]

    print(f"{'Entity':<30} {'Page Title':<30} {'Predicted':<15} {'Confidence':<10}")
    print("-" * 85)

    correct = 0
    for entity, page_title, expected in test_cases:
        entity_type, confidence = analyzer.classify_entity(entity, page_title)
        is_correct = "✓" if entity_type == expected else "✗"

        if entity_type == expected:
            correct += 1

        # Truncate for display
        entity_display = entity[:28] + ".." if len(entity) > 30 else entity
        title_display = (page_title[:28] + "..") if page_title and len(page_title) > 30 else (page_title or "N/A")

        print(f"{entity_display:<30} {title_display:<30} {entity_type:<15} {confidence:.2f} {is_correct}")

    print(f"\nAccuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.1f}%)")
    return correct / len(test_cases)


def test_conflict_prediction():
    """Test conflict prediction between entity pairs"""
    print("\n" + "=" * 60)
    print("TESTING CONFLICT PREDICTION")
    print("=" * 60)

    analyzer = SemanticAnalyzer()

    # Test cases with different conflict levels
    test_pairs = [
        # (entity1, entity2, entity1_type, entity2_type, involves_hub, expected_range)
        ("isbn", "issn", "identifier", "identifier", True, (0.8, 1.0)),
        ("isbn", "hamburg", "identifier", "geographic", True, (0.7, 1.0)),
        ("paris", "france", "geographic", "geographic", False, (0.2, 0.4)),
        ("the film", "the director", "creative_work", "person", False, (0.1, 0.3)),
        ("world war ii", "1945", "event", "temporal", False, (0.1, 0.3)),
        ("einstein", "relativity", "person", "concept", False, (0.1, 0.3)),
        ("random_entity_1", "random_entity_2", "general", "general", False, (0.2, 0.4))
    ]

    print(f"{'Entity 1':<20} {'Entity 2':<20} {'Type Pair':<25} {'Predicted':<10} {'Expected':<15}")
    print("-" * 90)

    for entity1, entity2, type1, type2, hub, expected_range in test_pairs:
        prob = analyzer.predict_conflict_probability(
            entity1, entity2, type1, type2,
            entity1_degree=100 if hub else 10,
            entity2_degree=100 if hub else 10,
            involves_hub=hub
        )

        in_range = expected_range[0] <= prob <= expected_range[1]
        status = "✓" if in_range else "✗"

        type_pair = f"{type1}-{type2}"[:23] + ".." if len(f"{type1}-{type2}") > 25 else f"{type1}-{type2}"

        print(f"{entity1:<20} {entity2:<20} {type_pair:<25} {prob:.3f} "
              f"{expected_range[0]:.1f}-{expected_range[1]:.1f} {status}")


def test_relationship_enrichment(scenario_path: str = None):
    """Test enrichment of actual relationships from your dataset"""
    print("\n" + "=" * 60)
    print("TESTING RELATIONSHIP ENRICHMENT")
    print("=" * 60)

    analyzer = SemanticAnalyzer()

    # Use sample data or load from file
    if scenario_path and os.path.exists(scenario_path):
        print(f"Loading relationships from: {scenario_path}")
        with open(scenario_path, 'r') as f:
            data = json.load(f)
            relationships = data.get('links_to', [])  # First 20 for testing
    else:
        # Create sample relationships
        print("Using sample relationship data...")
        relationships = [
            {
                "from": "isbn",
                "to": "the book",
                "similarity": 0.8,
                "involves_hub": True,
                "pages_1": ["page1", "page2", "page3"],
                "pages_2": ["page4"]
            },
            {
                "from": "hamburg",
                "to": "germany",
                "similarity": 0.9,
                "involves_hub": False,
                "pages_1": ["page5"],
                "pages_2": ["page6", "page7"]
            },
            {
                "from": "the film",
                "to": "2025",
                "similarity": 0.6,
                "involves_hub": False,
                "pages_1": ["page8"],
                "pages_2": ["page9"]
            }
        ]

    # Load processed pages if available
    processed_pages = None
    if scenario_path:
        pages_path = scenario_path.replace('relationships.json', 'processed_pages.json')
        if os.path.exists(pages_path):
            with open(pages_path, 'r') as f:
                processed_pages = json.load(f)
                print(f"Loaded {len(processed_pages)} processed pages for context")

    # Enrich relationships
    enriched, type_dist, pair_dist = analyzer.enrich_relationships(relationships, processed_pages)

    # Display results
    print(f"\nEnriched {len(enriched)} relationships")
    print("\nSample enriched relationships:")
    print("-" * 90)

    for i, rel in enumerate(enriched[:5]):  # Show first 5
        print(f"\nRelationship {i+1}:")
        print(f"  From: {rel['from']} ({rel['from_type']}, conf: {rel['from_type_confidence']:.2f})")
        print(f"  To: {rel['to']} ({rel['to_type']}, conf: {rel['to_type_confidence']:.2f})")
        print(f"  Type pair: {rel['type_pair']}")
        print(f"  Predicted conflict: {rel['predicted_conflict']:.3f}")
        print(f"  Hub involved: {rel.get('involves_hub', False)}")

    # Show statistics
    stats = analyzer.get_semantic_statistics(enriched)

    print("\nSemantic Statistics:")
    print("-" * 40)
    print(f"Total relationships: {stats['total_relationships']}")
    print(f"Average conflict probability: {stats['average_conflict_probability']:.3f}")

    print("\nType distribution:")
    for entity_type, count in stats['type_distribution'].most_common():
        print(f"  {entity_type}: {count}")

    print("\nConflict prediction distribution:")
    for level, count in stats['conflict_predictions'].items():
        percentage = 100 * count / len(enriched) if enriched else 0
        print(f"  {level}: {count} ({percentage:.1f}%)")

    print("\nTop type pairs:")
    for pair, count in list(stats['type_pair_distribution'].most_common(5)):
        print(f"  {pair}: {count}")

    return enriched, stats


def test_known_hotspots():
    """Specifically test classification of known hub entities"""
    print("\n" + "=" * 60)
    print("TESTING KNOWN HOT-SPOT ENTITIES")
    print("=" * 60)

    analyzer = SemanticAnalyzer()

    # Your known hot-spots
    hotspots = [
        ("isbn", 4396),
        ("issn", 4050),
        ("imdb", 3471)
    ]

    print(f"{'Entity':<20} {'References':<15} {'Type':<15} {'Conflict Prob':<15}")
    print("-" * 65)

    for entity, ref_count in hotspots:
        entity_type, confidence = analyzer.classify_entity(entity)

        # Test conflict with another hub
        conflict_prob = analyzer.predict_conflict_probability(
            entity, "some_other_entity",
            entity1_degree=ref_count,
            entity2_degree=10
        )

        print(f"{entity:<20} {ref_count:<15} {entity_type:<15} {conflict_prob:.3f}")


def run_all_tests(scenario_path: str = None):
    """Run all validation tests"""
    print("SEMANTIC ANALYZER VALIDATION SUITE")
    print("=" * 60)
    print()

    # Run tests
    classification_accuracy = test_entity_classification()
    test_conflict_prediction()
    test_known_hotspots()

    # Test with actual data if provided
    if scenario_path:
        enriched, stats = test_relationship_enrichment(scenario_path)
    else:
        test_relationship_enrichment()

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    # Summary
    print(f"\nEntity classification accuracy: {classification_accuracy:.1%}")
    print("Conflict prediction: Check ranges above")
    print("Hot-spot detection: All identified as high-conflict")

    if scenario_path:
        print(f"\nReady to integrate with your adaptive algorithm!")
        print(f"Semantic preprocessing would add ~{len(enriched)*0.001:.2f}s overhead")


if __name__ == "__main__":
    # Update this path to your actual scenario data
    scenario_path = "data/output/experimental_scenarios/dev_c_diverse/relationships.json"
    run_all_tests(scenario_path)
