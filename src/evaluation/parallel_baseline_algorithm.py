import concurrent.futures
import random
import time
from typing import List, Dict

from src.evaluation.framework import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class SimpleParallelBaseline(AlgorithmBase):
    """
    Truly simple parallel baseline for thesis research.

    This implementation strips away complexity to focus on the core research question:
    "What happens when you naively apply parallelism to database operations?"

    Key simplifications:
    - Fixed batch size, no adaptation
    - Basic round-robin distribution
    - Simple retry logic with conflict tracking
    - Simple conflict counting
    - No sophisticated error handling
    """

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.batch_size = config.get('batch_size', 100)
        self.thread_count = config.get('thread_count', 10)
        self.max_retries = config.get('max_retries', 2)  # Simple retry count
        self.name = config.get('name', 'Simplified Parallel Baseline')

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        """
        Execute simplified parallel processing
        """
        print(f"\n--- Simplified Parallel Baseline ---")
        print(f"Processing {len(relationships)} relationships")
        print(
            f"Configuration: {self.thread_count} threads, batch size {self.batch_size}, max retries {self.max_retries}")

        # Clear previous data
        self.clear_database()

        # Start monitoring
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        start_time = time.time()

        # Create batches using simple division
        batches = self._create_simple_batches(relationships)
        print(f"Created {len(batches)} batches")

        # Execute parallel processing
        results = self._execute_parallel_batches(batches)

        # Calculate final metrics
        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        # Aggregate results from all threads
        total_successful = sum(r['successful'] for r in results)
        total_conflicts = sum(r['conflicts'] for r in results)
        total_retries = sum(r['retries'] for r in results)
        batch_times = [r['time'] for r in results if r['time'] > 0]

        # Calculate derived metrics
        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (total_successful / len(relationships)) * 100

        # Simple result summary
        print(f"\nResults:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Throughput: {throughput:.1f} rel/s")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Conflicts: {total_conflicts}")
        print(f"  Retries: {total_retries}")

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",  # Set by framework
            run_number=0,  # Set by framework
            total_time=total_time,
            throughput=throughput,
            success_rate=success_rate,
            processing_overhead_time=0.0,  # No preprocessing overhead
            actual_conflicts=total_conflicts,
            retry_count=total_retries,
            adaptation_events=0,  # Static algorithm
            final_parallelism=self.thread_count,
            memory_peak=resource_metrics.get('memory_peak', 0),
            cpu_avg=resource_metrics.get('cpu_avg', 0),
            batch_processing_times=batch_times
        )

    def _create_simple_batches(self, relationships: List[Dict]) -> List[List[Dict]]:
        """
        Create batches with naive random distribution.

        This simulates a real-world scenario where a developer tries to
        parallelize without understanding the underlying data conflicts.
        The random shuffling ensures hub entities get distributed across
        threads, creating realistic conflict patterns.
        """

        # Shuffle relationships randomly to break up sequential patterns
        shuffled_relationships = relationships.copy()
        random.shuffle(shuffled_relationships)

        # Create batches from shuffled data
        batches = []
        for i in range(0, len(shuffled_relationships), self.batch_size):
            batch = shuffled_relationships[i:i + self.batch_size]
            batches.append(batch)

        print(f"  Random batching created {len(batches)} batches")
        return batches

    def _execute_parallel_batches(self, batches: List[List[Dict]]) -> List[Dict]:

        results = []

        # Use ThreadPoolExecutor for genuine parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            # Submit all batches for parallel processing
            future_to_batch = {
                executor.submit(self._process_single_batch, i, batch): i
                for i, batch in enumerate(batches)
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    result = future.result(timeout=60)  # Simple timeout
                    results.append(result)
                    print(f"  Batch {batch_id} completed: {result['successful']}/{len(batches[batch_id])} successful")
                except Exception as e:
                    print(f"  Batch {batch_id} failed: {str(e)[:50]}")
                    # Record failed batch
                    results.append({
                        'successful': 0,
                        'conflicts': 0,
                        'retries': 0,
                        'time': 0
                    })

        return results

    def _process_single_batch(self, batch_id: int, batch: List[Dict]) -> Dict:

        batch_start = time.time()

        # Use your exact _insert_batch logic WITH RETRIES
        conflicts, retries, successful = self._insert_batch_with_retry(batch, batch_id)

        processing_time = time.time() - batch_start

        print(f"    Thread {batch_id}: {successful}/{len(batch)} successful, {conflicts} conflicts, {retries} retries")

        return {
            'successful': successful,
            'conflicts': conflicts,
            'retries': retries,
            'time': processing_time
        }

    def _insert_batch_with_retry(self, batch: List[Dict], batch_idx: int) -> tuple:
        """
        Insert a batch with simple retry logic.
        IMPORTANT: We count ALL conflicts, even if retry succeeds!
        """
        total_conflicts = 0
        total_retries = 0
        successes = 0
        experiment_id = f"parallel_baseline_{batch_idx}_{int(time.time() * 1000)}"

        with self.driver.session() as session:
            for rel in batch:
                retry_count = 0
                succeeded = False

                # Try up to max_retries + 1 times (initial attempt + retries)
                while retry_count <= self.max_retries and not succeeded:
                    try:
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
                            'from_name': rel['from'],
                            'to_name': rel['to'],
                            'similarity': rel['similarity'],
                            'involves_hub': rel['involves_hub'],
                            'pages_1': rel['pages_1'],
                            'pages_2': rel['pages_2'],
                            'experiment_id': experiment_id
                        })

                        succeeded = True
                        successes += 1

                    except Exception as e:
                        # Your exact conflict detection logic
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ['lock', 'deadlock', 'timeout']):
                            # THIS IS KEY: We ALWAYS count the conflict
                            total_conflicts += 1

                            if retry_count < self.max_retries:
                                # We're going to retry
                                retry_count += 1
                                total_retries += 1

                                # Simple exponential backoff
                                wait_time = 0.1 * (2 ** retry_count)
                                time.sleep(wait_time)

                                print(
                                    f"      Conflict detected, retry {retry_count}/{self.max_retries} after {wait_time:.2f}s")
                            else:
                                # No more retries left
                                break
                        else:
                            # Non-conflict error, don't retry
                            break

        return total_conflicts, total_retries, successes