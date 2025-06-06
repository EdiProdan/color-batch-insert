from collections import Counter
from datetime import datetime
import time
import json
import tempfile
import os

from src.evaluation.framework import PerformanceMetrics, ConflictAnalyzer, ResourceMonitor, AlgorithmBase


class ApocAlgorithm(AlgorithmBase):
    """
    APOC-based insertion algorithm using apoc.periodic.iterate
    This provides a clean baseline comparison for your adaptive algorithm research
    """

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.batch_sizes = config.get('batch_sizes', [100])  # APOC batch sizes to test
        self.timeout = config.get('timeout', 2500)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.parallel_processing = config.get('parallel', True)  # Enable APOC parallel processing
        self.conflict_analyzer = ConflictAnalyzer(
            hub_threshold=config.get('hub_threshold', 5)
        )

    def insert_relationships(self, relationships):
        """
        Insert relationships using APOC with comprehensive performance tracking
        """
        print(f"\nExecuting {self.name}")
        print(f"Processing {len(relationships)} relationships using APOC")

        best_metrics = None
        best_time = float('inf')

        # Test different batch sizes to find optimal performance
        for batch_size in self.batch_sizes:
            print(f"  Testing APOC batch size: {batch_size}")

            # Clear database for clean test
            self.clear_database()

            metrics = self._run_apoc_insertion(relationships, batch_size)

            if metrics.total_time < best_time:
                best_time = metrics.total_time
                best_metrics = metrics

        return best_metrics

    def _run_apoc_insertion(self, relationships, batch_size):
        """Execute APOC insertion with detailed performance monitoring"""

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        start_time = time.time()
        experiment_id = f"apoc_exp_{int(time.time() * 1000)}"

        # Initialize tracking variables
        total_predicted_conflicts = 0
        actual_conflicts = 0
        retry_count = 0
        conflict_resolution_time = 0
        all_hotspot_entities = []

        # Analyze conflicts beforehand (this helps with research comparison)
        print("    Analyzing potential conflicts...")
        batches = [relationships[i:i + batch_size]
                   for i in range(0, len(relationships), batch_size)]

        for batch in batches:
            conflict_analysis = self.conflict_analyzer.detect_conflicts_in_batch(batch)
            total_predicted_conflicts += conflict_analysis['total_predicted_conflicts']
            all_hotspot_entities.extend(conflict_analysis['conflict_hotspots'])

        print(f"    Predicted total conflicts: {total_predicted_conflicts}")

        # Execute APOC insertion
        insertion_start = time.time()

        try:
            apoc_result = self._execute_apoc_batch_insertion(
                relationships, batch_size, experiment_id
            )

            # Extract conflict information from APOC results
            actual_conflicts = apoc_result.get('failedOperations', 0)
            retry_count = apoc_result.get('retries', 0)

            print(f"    APOC Results:")
            print(f"      Total operations: {apoc_result.get('total', 0)}")
            print(f"      Committed: {apoc_result.get('committedOperations', 0)}")
            print(f"      Failed: {apoc_result.get('failedOperations', 0)}")
            print(f"      Batches processed: {apoc_result.get('batches', 0)}")
            print(f"      Retries: {apoc_result.get('retries', 0)}")

        except Exception as e:
            print(f"    APOC insertion failed: {e}")
            actual_conflicts = len(relationships)  # Assume all failed

        conflict_resolution_time = time.time() - insertion_start
        total_time = time.time() - start_time

        # Stop monitoring and collect resource metrics
        resource_metrics = monitor.stop_monitoring()

        # Calculate performance metrics
        successful_operations = len(relationships) - actual_conflicts
        throughput = successful_operations / total_time if total_time > 0 else 0
        success_rate = (successful_operations / len(relationships)) * 100

        # Calculate conflict prediction accuracy
        if total_predicted_conflicts > 0:
            prediction_accuracy = min(100.0, (actual_conflicts / total_predicted_conflicts) * 100)
        else:
            prediction_accuracy = 100.0 if actual_conflicts == 0 else 0.0

        # Get top hotspot entities
        hotspot_counter = Counter(all_hotspot_entities)
        top_hotspots = [entity for entity, count in hotspot_counter.most_common(5)]

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",  # Will be set by framework
            run_number=0,  # Will be set by framework
            batch_size=batch_size,
            total_time=total_time,
            batch_processing_times=[conflict_resolution_time],  # APOC handles batching internally
            total_entities=self._count_unique_entities(relationships),
            total_relationships=len(relationships),
            predicted_conflicts=total_predicted_conflicts,
            actual_conflicts=actual_conflicts,
            conflict_prediction_accuracy=prediction_accuracy,
            conflict_resolution_time=conflict_resolution_time,
            retry_count=retry_count,
            hotspot_entities=top_hotspots,
            throughput=throughput,
            memory_peak=resource_metrics['memory_peak'],
            cpu_avg=resource_metrics['cpu_avg'],
            success_rate=success_rate,
            timestamp=datetime.now().isoformat()
        )

    def _execute_apoc_batch_insertion(self, relationships, batch_size, experiment_id):
        """
        Execute the actual APOC insertion with proper parameter handling

        This implementation uses APOC's parameter passing correctly
        """

        with self.driver.session() as session:
            # The key insight: we need to embed the data directly in the first statement
            # or use APOC's parameter passing mechanism correctly

            query = """
            CALL apoc.periodic.iterate(
                "UNWIND $relationships AS rel RETURN rel",
                "
                MERGE (from:Entity {name: rel.from})
                MERGE (to:Entity {name: rel.to})
                SET from.experiment_id = $experiment_id,
                    to.experiment_id = $experiment_id
                MERGE (from)-[r:LINKS_TO]->(to)
                SET r.similarity = rel.similarity,
                    r.involves_hub = rel.involves_hub,
                    r.pages_from = rel.pages_1,
                    r.pages_to = rel.pages_2,
                    r.experiment_id = $experiment_id
                ",
                {
                    batchSize: $batch_size,
                    parallel: $parallel,
                    retries: $retries,
                    params: {
                        relationships: $relationships,
                        experiment_id: $experiment_id
                    }
                }
            )
            """

            result = session.run(query, {
                'relationships': relationships,
                'experiment_id': experiment_id,
                'batch_size': batch_size,
                'parallel': self.parallel_processing,
                'retries': self.retry_attempts
            })

            return result.single().data()

    def _count_unique_entities(self, relationships):
        """Count unique entities in the relationship set"""
        entities = set()
        for rel in relationships:
            entities.add(rel['from'])
            entities.add(rel['to'])
        return len(entities)
