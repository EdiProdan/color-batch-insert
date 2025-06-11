"""
Test Script for SemanticAnalyzer
Tests the semantic preprocessing capabilities with real thesis data
"""
import yaml
import json
import os
from neo4j import GraphDatabase
from src.evaluation.semantic_analyzer import SemanticAnalyzer


def load_config():
    """Load configuration from config.yaml"""
    with open("config.yaml", "r") as config_file:
        return yaml.safe_load(config_file)


def load_test_relationships(config, scenario="dev_a_precision"):
    """Load relationships from a test scenario"""
    #scenario_path = f"{config['data']['output']}{scenario}/relationships.json"
    scenario_path = f"data/output/scenario_1/relationships.json"

    if not os.path.exists(scenario_path):
        print(f"⚠️  Scenario file not found: {scenario_path}")
        print("Available scenarios in data/output/experimental_scenarios/:")
        base_path = config['data']['output']
        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                if os.path.isdir(os.path.join(base_path, item)):
                    print(f"  - {item}")
        return []

    with open(scenario_path, 'r') as f:
        data = json.load(f)

    return data['links_to']


def test_semantic_analyzer():
    """Main test function - demonstrates SemanticAnalyzer capabilities"""
    print("🧪 SEMANTIC ANALYZER TEST")
    print("=" * 60)

    # Load configuration
    print("📁 Loading configuration...")
    config = load_config()

    # Connect to Neo4j
    print("🔌 Connecting to Neo4j database...")
    try:
        driver = GraphDatabase.driver(
            config['database']['neo4j']['uri'],
            auth=(
                config['database']['neo4j']['user'],
                config['database']['neo4j']['password']
            )
        )
        print("  ✅ Database connection successful")
    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        print("  💡 Make sure Neo4j is running and credentials are correct")
        return

    # Load test relationships
    print("\n📊 Loading test relationships...")
    relationships = load_test_relationships(config)

    if not relationships:
        print("  ❌ No relationships loaded. Check your data files.")
        return

    print(f"  ✅ Loaded {len(relationships)} relationships for testing")

    # Create and test SemanticAnalyzer
    print("\n🔍 Creating SemanticAnalyzer...")
    analyzer = SemanticAnalyzer(driver)



    try:
        # Run the main analysis
        analysis_context = analyzer.analyze_preprocessing_context(relationships)

        # Print detailed results
        analyzer.print_analysis_summary(analysis_context)

        # Additional detailed inspection for research validation
        print(f"\n🔬 DETAILED RESEARCH INSIGHTS")
        print("=" * 50)

        # Show some example classifications
        incoming = analysis_context['incoming_analysis']
        print(f"\n📋 Sample Entity Classifications:")
        classification_examples = {}
        for entity, cls in list(incoming['entity_classifications'].items())[:10]:
            if cls not in classification_examples:
                classification_examples[cls] = []
            if len(classification_examples[cls]) < 3:
                classification_examples[cls].append(entity)

        for semantic_type, examples in classification_examples.items():
            print(f"  {semantic_type}: {', '.join(examples)}")

        # Show high-risk entities with details
        conflicts = analysis_context['conflict_assessment']
        print(f"\n⚠️  High-Risk Entity Analysis:")
        high_risk_entities = [(entity, risk) for entity, risk in
                             conflicts['entity_risk_scores'].items() if risk > 0.5]
        high_risk_entities.sort(key=lambda x: x[1], reverse=True)

        for entity, risk in high_risk_entities[:5]:
            classification = incoming['entity_classifications'].get(entity, 'unknown')
            print(f"  {entity} (risk: {risk:.2f}, type: {classification})")

        # Research validation metrics
        print(f"\n📈 Research Validation Metrics:")
        print(f"  Classification coverage: {incoming['classification_coverage']:.1%}")
        print(f"  Processing overhead: {analysis_context['preprocessing_time']:.2f}s")
        print(f"  Risk entities identified: {conflicts['risk_summary']['total_high_risk']}")
        print(f"  Database overlap detected: {conflicts['risk_summary']['overlap_with_database']}")

        # Test with full dataset (if reasonable size)
        if len(relationships) <= 1000:
            print(f"\n🚀 Testing with full dataset ({len(relationships)} relationships)...")
            full_analysis = analyzer.analyze_preprocessing_context(relationships)
            print(f"  Full dataset preprocessing time: {full_analysis['preprocessing_time']:.2f}s")
            print(f"  Full dataset entities: {full_analysis['incoming_analysis']['total_unique_entities']}")
        else:
            print(f"\n💡 Full dataset has {len(relationships)} relationships.")
            print(f"   For full testing, run: analyzer.analyze_preprocessing_context(relationships)")

        print(f"\n✅ SemanticAnalyzer test completed successfully!")
        print(f"\n💡 Research Insights:")
        print(f"   - Your semantic patterns captured {incoming['classification_coverage']:.1%} of entities")
        print(f"   - Preprocessing overhead is {analysis_context['preprocessing_time']:.2f}s for {len(relationships)} relationships")
        print(f"   - {conflicts['risk_summary']['total_high_risk']} high-risk entities identified for careful processing")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        driver.close()
        print("\n🔌 Database connection closed")


if __name__ == "__main__":

    test_semantic_analyzer()