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
        # Delete experimental nodes (those with isBase=False)
        with self.driver.session() as session:
            result = session.run("""
                        MATCH (n:Entity)
                        WHERE n.isBase = false
                        DETACH DELETE n
                        RETURN count(n) as deleted_count
                    """)

            deleted_count = result.single()["deleted_count"]
            print(f"Cleared {deleted_count} experimental nodes and their relationships")
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

        scenario_path = f"{self.config['data']['output']}/relationships.json"

        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario data not found: {scenario_path}")

        with open(scenario_path, 'r') as f:
            data = json.load(f)

        return data['LINKS_TO']

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

        results_file = f"{output_dir}raw_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
