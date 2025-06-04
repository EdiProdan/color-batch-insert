from collections import Counter
from datetime import datetime
import time

from src.evaluation.framework import PerformanceMetrics, ConflictAnalyzer, ResourceMonitor, AlgorithmBase


class BaselineAlgorithm(AlgorithmBase):
    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.batch_sizes = config.get('batch_sizes', [250])
        self.timeout = config.get('timeout', 300)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.conflict_analyzer = ConflictAnalyzer(
            hub_threshold=config.get('hub_threshold', 5)
        )


    def insert_relationships(self, relationships):
        """
        Enhanced insertion with comprehensive conflict tracking
        """
        print(f"\nExecuting {self.name}")
        print(f"Processing {len(relationships)} relationships")

        best_metrics = None
        best_time = float('inf')

        for batch_size in self.batch_sizes:
            print(f"  Testing batch size: {batch_size}")

            self.clear_database()

            metrics = self._run_single_batch_size(relationships, batch_size)

            if metrics.total_time < best_time:
                best_time = metrics.total_time
                best_metrics = metrics

        return best_metrics


    def _run_single_batch_size(self, relationships, batch_size):
        """Run algorithm with enhanced conflict detection and metrics collection"""

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        start_time = time.time()
        batch_times = []

        # Initialize conflict tracking
        total_predicted_conflicts = 0
        total_actual_conflicts = 0
        total_retry_count = 0
        conflict_resolution_time = 0
        all_hotspot_entities = []

        successful_operations = 0
        total_operations = 0

        # Create batches
        batches = [relationships[i:i + batch_size]
                   for i in range(0, len(relationships), batch_size)]

        print(f"    Created {len(batches)} batches of size {batch_size}")

        # Analyze cross-batch conflicts (important for parallel algorithms)
        cross_batch_analysis = self.conflict_analyzer.analyze_cross_batch_conflicts(batches)
        print(f"    Cross-batch conflicts detected: {cross_batch_analysis['total_cross_batch_conflicts']}")

        # Process each batch with detailed conflict tracking
        for batch_idx, batch in enumerate(batches):
            batch_start = time.time()

            # STEP 1: Predict conflicts before insertion
            conflict_analysis = self.conflict_analyzer.detect_conflicts_in_batch(batch)
            predicted_conflicts = conflict_analysis['total_predicted_conflicts']
            total_predicted_conflicts += predicted_conflicts

            # Track hotspot entities
            all_hotspot_entities.extend(conflict_analysis['conflict_hotspots'])

            print(f"    Batch {batch_idx + 1}: Predicted {predicted_conflicts} conflicts")
            if conflict_analysis['conflict_hotspots']:
                print(f"      Hotspot entities: {conflict_analysis['conflict_hotspots'][:3]}...")

            # STEP 2: Insert batch and measure actual conflicts
            conflict_start = time.time()
            batch_conflicts, retries = 0, 0
            try:
                batch_conflicts, retries = self._insert_batch_with_conflict_tracking(batch)
                total_actual_conflicts += batch_conflicts
                total_retry_count += retries
                successful_operations += len(batch)

                # Update conflict history for learning
                for entity in conflict_analysis['conflict_entities']:
                    self.conflict_analyzer.update_conflict_history(entity, batch_conflicts)

            except Exception as e:
                print(f"    Batch {batch_idx + 1} failed: {str(e)[:100]}")

            conflict_resolution_time += time.time() - conflict_start
            total_operations += len(batch)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            print(f"    Batch {batch_idx + 1}/{len(batches)} completed. "
                  f"Actual conflicts: {batch_conflicts}, Retries: {retries}")

        total_time = time.time() - start_time

        # Stop monitoring and collect resource metrics
        resource_metrics = monitor.stop_monitoring()

        # Calculate enhanced performance metrics
        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (successful_operations / total_operations) * 100

        # Calculate conflict prediction accuracy
        if total_predicted_conflicts > 0:
            prediction_accuracy = min(100.0, (total_actual_conflicts / total_predicted_conflicts) * 100)
        else:
            prediction_accuracy = 100.0 if total_actual_conflicts == 0 else 0.0

        # Get top hotspot entities
        hotspot_counter = Counter(all_hotspot_entities)
        top_hotspots = [entity for entity, count in hotspot_counter.most_common(5)]

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",  # Will be set by framework
            run_number=0,  # Will be set by framework
            batch_size=batch_size,
            total_time=total_time,
            batch_processing_times=batch_times,
            total_entities=self._count_unique_entities(relationships),
            total_relationships=len(relationships),
            predicted_conflicts=total_predicted_conflicts,
            actual_conflicts=total_actual_conflicts,
            conflict_prediction_accuracy=prediction_accuracy,
            conflict_resolution_time=conflict_resolution_time,
            retry_count=total_retry_count,
            hotspot_entities=top_hotspots,
            throughput=throughput,
            memory_peak=resource_metrics['memory_peak'],
            cpu_avg=resource_metrics['cpu_avg'],
            success_rate=success_rate,
            timestamp=datetime.now().isoformat()
        )


    def _insert_batch_with_conflict_tracking(self, batch):
        """
        Insert batch with detailed conflict and retry tracking

        Returns:
            Tuple of (actual_conflicts, retry_count)
        """
        conflicts_detected = 0
        retry_count = 0
        experiment_id = f"exp_{int(time.time() * 1000)}"

        with self.driver.session() as session:
            for relationship in batch:
                insertion_successful = False
                attempt = 0

                while not insertion_successful and attempt < self.retry_attempts:
                    try:
                        # Track timing for conflict resolution
                        attempt_start = time.time()

                        query = """
                        MERGE (from:Entity {name: $from_name})
                        MERGE (to:Entity {name: $to_name})
                        SET from.experiment_id = $experiment_id,
                            to.experiment_id = $experiment_id
                        MERGE (from)-[r:LINKS_TO]->(to)
                        SET r.similarity = $similarity,
                            r.involves_hub = $involves_hub,
                            r.pages_from = $pages_1,
                            r.pages_to = $pages_2,
                            r.experiment_id = $experiment_id
                        """

                        session.run(query, {
                            'from_name': relationship['from'],
                            'to_name': relationship['to'],
                            'similarity': relationship['similarity'],
                            'involves_hub': relationship['involves_hub'],
                            'pages_1': relationship['pages_1'],
                            'pages_2': relationship['pages_2'],
                            'experiment_id': experiment_id
                        })

                        insertion_successful = True

                    except Exception as e:
                        attempt += 1

                        # Detect different types of conflicts
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in
                               ['lock', 'deadlock', 'timeout', 'conflict']):
                            conflicts_detected += 1

                            if attempt < self.retry_attempts:
                                retry_count += 1
                                # Wait before retry (exponential backoff)
                                time.sleep(0.1 * (2 ** attempt))
                        else:
                            # Non-conflict error, don't retry
                            break

        return conflicts_detected, retry_count


    def _count_unique_entities(self, relationships):
        """Count unique entities in the relationship set"""
        entities = set()
        for rel in relationships:
            entities.add(rel['from'])
            entities.add(rel['to'])
        return len(entities)
