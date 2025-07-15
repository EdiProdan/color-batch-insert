import concurrent.futures
import time
from typing import List, Dict

from src.evaluation import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class NaiveParallelInsert(AlgorithmBase):

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.batch_size = config.get('batch_size')
        self.thread_count = config.get('thread_count')
        self.name = config.get('name')

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        print(f"\n--- {self.name} ---")
        print(f"Processing {len(relationships)} relationships")
        print(f"Configuration: {self.thread_count} threads, batch size {self.batch_size}")

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        start_time = time.time()

        batches = []
        for i in range(0, len(relationships), self.batch_size):
             batch = relationships[i:i + self.batch_size]
             batches.append(batch)

        results = self._execute_parallel_batches(batches)

        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        total_successful = sum(r['successful'] for r in results)
        total_conflicts = sum(r['conflicts'] for r in results)

        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (total_successful / len(relationships)) * 100

        print(f"\nResults:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Throughput: {throughput:.1f} rel/s")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Conflicts: {total_conflicts}")

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",
            run_number=0,

            total_time=total_time,
            throughput=throughput,
            success_rate=success_rate,

            processing_overhead=0.0,
            conflicts=total_conflicts,

            memory_peak=resource_metrics['memory_peak'],
            cpu_avg=resource_metrics['cpu_avg']
        )

    def _execute_parallel_batches(self, batches: List[List[Dict]]) -> List[Dict]:

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            future_to_batch = {
                executor.submit(self._process_single_batch, i, batch): i
                for i, batch in enumerate(batches)
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'successful': 0,
                        'conflicts': 0,
                        'time': 0
                    })

        return results

    def _process_single_batch(self, batch_id: int, batch: List[Dict]) -> Dict:

        batch_start = time.time()

        conflicts, successful = self._insert_batch(batch)

        processing_time = time.time() - batch_start

        print(f"    Thread {batch_id}: {successful}/{len(batch)} successful, {conflicts} conflicts")

        return {
            'successful': successful,
            'conflicts': conflicts,
            'time': processing_time
        }

    def _insert_batch(self, batch: List[Dict]) -> tuple:

        total_conflicts = 0
        successes = 0

        with self.driver.session() as session:
            for rel in batch:

                try:
                    query = """
                        MERGE (from:Entity {title: $from})
                        ON CREATE SET from.isBase = false, from.processed_at = timestamp()
                        ON MATCH SET from.isBase = COALESCE(from.isBase, false)
                        
                        MERGE (to:Entity {title: $to})
                        ON CREATE SET to.isBase = false, to.processed_at = timestamp()
                        ON MATCH SET to.isBase = COALESCE(to.isBase, false)
                        
                        MERGE (from)-[r:LINKS_TO]->(to)
                        ON CREATE SET r.created_at = timestamp()
                        """

                    session.run(query, {'from': rel["from"], 'to': rel["to"]})
                    successes += 1

                except Exception as e:
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ['lock', 'deadlock', 'timeout']):
                        total_conflicts += 1
                    else:
                        continue

        return total_conflicts, successes