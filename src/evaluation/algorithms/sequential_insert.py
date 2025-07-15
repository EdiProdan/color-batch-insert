import time
from typing import List, Dict

from src.evaluation import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class SequentialInsert(AlgorithmBase):

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.batch_size = config.get('batch_size')
        self.name = config.get('name')

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        print(f"\n--- {self.name} ---")
        print(f"Processing {len(relationships)} relationships")
        print(f"Batch size: {self.batch_size}")

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        start_time = time.time()

        batches = [relationships[i:i + self.batch_size]
                   for i in range(0, len(relationships), self.batch_size)]

        batch_times = []
        actual_conflicts = 0
        successful_operations = 0

        for batch_idx, batch in enumerate(batches):
            batch_start = time.time()

            conflicts, successes = self._insert_batch(batch)

            actual_conflicts += conflicts
            successful_operations += successes

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
                print(f"  Processed {batch_idx + 1}/{len(batches)} batches")

        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (successful_operations / len(relationships)) * 100

        print(f"\nCompleted in {total_time:.2f} seconds")
        print(f"Throughput: {throughput:.1f} relationships/second")
        print(f"Success rate: {success_rate:.1f}%")

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",
            run_number=0,

            total_time=total_time,
            throughput=throughput,
            success_rate=success_rate,

            processing_overhead=0.0,
            conflicts=actual_conflicts,

            memory_peak=resource_metrics['memory_peak'],
            cpu_avg=resource_metrics['cpu_avg']
        )

    def _insert_batch(self, batch: List[Dict]) -> tuple:

        conflicts = 0
        successes = 0

        with self.driver.session() as session:
            for rel in batch:
                try:
                    query = """
                    MERGE (from:Entity {title: $from})
                    ON CREATE SET from.isBase = true, from.processed_at = timestamp()
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
                        conflicts += 1

        return conflicts, successes
