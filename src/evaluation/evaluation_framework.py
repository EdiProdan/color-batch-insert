import json
import os
from dataclasses import asdict
from typing import List, Dict
from datetime import datetime
from neo4j import GraphDatabase
import yaml


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

        scenario_path = f"{self.config['data']['output']}/relationships_bbc_200_connected.json"

        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario data not found: {scenario_path}")

        with open(scenario_path, 'r') as f:
            data = json.load(f)

        return data['LINKS_TO']

    def run_evaluation(self):
        print("Starting comprehensive algorithm evaluation...")
        print(f"Algorithms: {list(self.algorithms.keys())}")
        print(f"Scenarios: {self.config['evaluation']['scenarios']}")
        print(f"Runs per algorithm: {self.config['evaluation']['runs_per_algorithm']}")

        for scenario in self.config['evaluation']['scenarios']:
            print(f"\n{'=' * 60}")
            print(f"EVALUATING SCENARIO: {scenario}")
            print(f"{'=' * 60}")

            relationships = self.load_scenario_data(scenario)
            print(f"Loaded {len(relationships)} relationships for {scenario}")

            for algorithm_name, algorithm in self.algorithms.items():
                print(f"\nTesting {algorithm_name}...")

                for run in range(self.config['evaluation']['runs_per_algorithm']):
                    print(f"  Run {run + 1}/{self.config['evaluation']['runs_per_algorithm']}")

                    algorithm.clear_database()

                    metrics = algorithm.insert_relationships(relationships)
                    metrics.scenario = scenario
                    metrics.run_number = run + 1

                    self.results.append(metrics)

                    print(f"    Time: {metrics.total_time:.2f}s, "
                          f"Throughput: {metrics.throughput:.1f} rel/s")

        self.save_results()

    def save_results(self):
        output_dir = self.config['evaluation']['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        results_dict = [asdict(result) for result in self.results]

        results_file = f"{output_dir}raw_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    def close(self):
        self.driver.close()


def create_evaluation_framework(config_path: str = "config.yaml") -> EvaluationFramework:

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return EvaluationFramework(config)
