"""
Evaluation Framework for Knowledge Graph Insertion Algorithms
Provides comprehensive performance measurement and statistical validation
"""
import time
import json
import os
import statistics
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime
import psutil
import threading

from neo4j import GraphDatabase


@dataclass
class PerformanceMetrics:
    """Minimal metrics for rigorous adaptive algorithm research"""

    # Research identification (essential)
    algorithm_name: str
    scenario: str
    run_number: int

    # Core performance story (your main research claims)
    total_time: float
    throughput: float
    success_rate: float

    # Intelligence investment (your innovation)
    processing_overhead_time: float
    actual_conflicts: int
    retry_count: int

    # Adaptation evidence (what makes you different)
    adaptation_events: int  # How many times algorithm adjusted
    final_parallelism: int  # What it settled on

    # Resource efficiency (scalability story)
    memory_peak: float
    cpu_avg: float

    # Analysis depth (optional but valuable)
    batch_processing_times: List[float]  # Shows learning patterns

class ConflictAnalyzer:
    """
    Analyzes relationship batches to predict conflicts before database insertion
    This is a key component of your adaptive algorithm research
    """

    def __init__(self, hub_threshold: int = 5):
        """
        Initialize conflict analyzer

        Args:
            hub_threshold: Number of relationships that makes an entity a "hub" (conflict-prone)
        """
        self.hub_threshold = hub_threshold
        self.conflict_history = defaultdict(int)  # Track historical conflicts per entity

    def detect_conflicts_in_batch(self, batch):
        """
        Analyze a batch to predict potential conflicts

        This method implements the core logic your adaptive algorithm will use
        to intelligently reorganize batches before insertion.

        Returns:
            Dictionary containing conflict analysis results
        """
        # Count how many times each entity appears in this batch
        entity_frequency = Counter()

        for relationship in batch:
            entity_frequency[relationship['from']] += 1
            entity_frequency[relationship['to']] += 1

        # Identify potential conflict entities
        conflict_entities = {
            entity: count for entity, count in entity_frequency.items()
            if count >= self.hub_threshold
        }

        # Identify hub entities (high-frequency entities prone to conflicts)
        hub_entities = {
            entity: count for entity, count in entity_frequency.items()
            if count >= self.hub_threshold
        }

        # Calculate conflict probability based on entity overlap
        total_predicted_conflicts = sum(
            max(0, count - 1) for count in entity_frequency.values()
            if count > 1
        )

        # Identify relationships involving conflict-prone entities
        conflicting_relationships = []
        for i, rel in enumerate(batch):
            if (rel['from'] in conflict_entities or
                    rel['to'] in conflict_entities):
                conflicting_relationships.append(i)

        return {
            'total_predicted_conflicts': total_predicted_conflicts,
            'conflict_entities': conflict_entities,
            'hub_entities': hub_entities,
            'conflicting_relationship_indices': conflicting_relationships,
            'entity_frequency_distribution': dict(entity_frequency),
            'max_entity_frequency': max(entity_frequency.values()) if entity_frequency else 0,
            'conflict_hotspots': [
                entity for entity, count in entity_frequency.most_common(10)
                if count >= self.hub_threshold
            ]
        }

    def analyze_cross_batch_conflicts(self, batches):
        """
        Analyze conflicts between different batches (for parallel processing)

        This helps predict which batches might conflict when processed simultaneously
        """
        batch_entities = []

        # Extract entities for each batch
        for batch in batches:
            entities = set()
            for rel in batch:
                entities.add(rel['from'])
                entities.add(rel['to'])
            batch_entities.append(entities)

        # Find overlapping entities between batches
        cross_batch_conflicts = []
        for i in range(len(batch_entities)):
            for j in range(i + 1, len(batch_entities)):
                overlap = batch_entities[i] & batch_entities[j]
                if overlap:
                    cross_batch_conflicts.append({
                        'batch_pair': (i, j),
                        'conflicting_entities': list(overlap),
                        'conflict_count': len(overlap)
                    })

        return {
            'total_cross_batch_conflicts': len(cross_batch_conflicts),
            'conflict_details': cross_batch_conflicts,
            'most_problematic_entities': self._find_most_problematic_entities(cross_batch_conflicts)
        }

    def _find_most_problematic_entities(self, conflicts):
        """Find entities that appear in the most cross-batch conflicts"""
        entity_conflict_count = Counter()

        for conflict in conflicts:
            for entity in conflict['conflicting_entities']:
                entity_conflict_count[entity] += 1

        return entity_conflict_count.most_common(10)

    def update_conflict_history(self, entity: str, actual_conflicts: int):
        """Update historical conflict data for better future predictions"""
        self.conflict_history[entity] += actual_conflicts


class ResourceMonitor:
    """Monitors system resources during algorithm execution"""

    def __init__(self):
        self.monitor_thread = None
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []

    def start_monitoring(self):
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []

        def monitor():
            while self.monitoring:
                self.cpu_samples.append(psutil.cpu_percent())
                self.memory_samples.append(psutil.virtual_memory().used / 1024 / 1024)  # MB
                time.sleep(0.5)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring and return average metrics"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)

        return {
            'cpu_avg': statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            'memory_peak': max(self.memory_samples) if self.memory_samples else 0
        }


class AlgorithmBase(ABC):

    def __init__(self, config, driver):
        self.config = config
        self.driver = driver
        self.name = self.config.get('name', 'Unknown Algorithm')

    @abstractmethod
    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        """Insert relationships and return performance metrics"""
        pass

    def clear_database(self):
        """Clear only experimental data, preserving original dataset"""
        with self.driver.session() as session:
            # Delete only nodes and relationships created during experiments
            session.run("""
                MATCH (n:Entity)
                WHERE n.experiment_id IS NOT NULL
                DETACH DELETE n
            """)

            # Also clean up any relationships that might reference experiment data
            session.run("""
                MATCH ()-[r:LINKS_TO]->()
                WHERE r.experiment_id IS NOT NULL
                DELETE r
            """)

        print("Cleared experimental data, preserved original dataset")

    def get_database_stats(self):
        """Get current database statistics"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                OPTIONAL MATCH (n)-[r]-()
                RETURN COUNT(DISTINCT n) as nodes, COUNT(r) as relationships
            """)
            return result.single()


class EvaluationFramework:

    def __init__(self, config):
        self.config = config
        self.driver = GraphDatabase.driver(
            config['database']['neo4j']['uri'],
            auth=(
                config['database']['neo4j']['user'],
                config['database']['neo4j']['password']
            )
        )
        self.algorithms = {}
        self.results = []

    def register_algorithm(self, algorithm_class, algorithm_config):

        algorithm = algorithm_class(algorithm_config, self.driver)
        self.algorithms[algorithm.name] = algorithm
        print(f"Registered algorithm: {algorithm.name}")

    def load_scenario_data(self, scenario_name: str) -> List[Dict]:

        scenario_path = f"{self.config['data']['output']}{scenario_name}/relationships.json"

        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario data not found: {scenario_path}")

        with open(scenario_path, 'r') as f:
            data = json.load(f)

        return data['links_to']

    def run_evaluation(self):
        """
        Execute comprehensive evaluation across all algorithms and scenarios
        This is your main research methodology execution
        """
        print("Starting comprehensive algorithm evaluation...")
        print(f"Algorithms: {list(self.algorithms.keys())}")
        print(f"Scenarios: {self.config['evaluation']['scenarios']}")
        print(f"Runs per algorithm: {self.config['evaluation']['runs_per_algorithm']}")

        for scenario in self.config['evaluation']['scenarios']:
            print(f"\n{'=' * 60}")
            print(f"EVALUATING SCENARIO: {scenario}")
            print(f"{'=' * 60}")

            try:
                relationships = self.load_scenario_data(scenario)
                print(f"Loaded {len(relationships)} relationships for {scenario}")

                for algorithm_name, algorithm in self.algorithms.items():
                    print(f"\nTesting {algorithm_name}...")

                    for run in range(self.config['evaluation']['runs_per_algorithm']):
                        print(f"  Run {run + 1}/{self.config['evaluation']['runs_per_algorithm']}")

                        # Clear database for clean test
                        algorithm.clear_database()

                        # Run algorithm and collect metrics
                        metrics = algorithm.insert_relationships(relationships)
                        metrics.scenario = scenario
                        metrics.run_number = run + 1

                        self.results.append(metrics)

                        # Brief result summary
                        print(f"    Time: {metrics.total_time:.2f}s, "
                              f"Throughput: {metrics.throughput:.1f} rel/s")

            except Exception as e:
                print(f"Error processing scenario {scenario}: {e}")
                continue

        self.save_results()

    def save_results(self):
        """Save raw results to JSON for detailed analysis"""
        output_dir = self.config['evaluation']['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        # Convert dataclasses to dicts for JSON serialization
        results_dict = [asdict(result) for result in self.results]

        results_file = f"{output_dir}/raw_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    def close(self):
        """Clean up resources"""
        self.driver.close()


def create_evaluation_framework(config_path: str = "config.yaml") -> EvaluationFramework:
    """Factory function to create configured evaluation framework"""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return EvaluationFramework(config)


