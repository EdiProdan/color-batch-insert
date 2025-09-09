import time
from typing import List, Dict

from src.evaluation import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class SequentialInsert(AlgorithmBase):

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.batch_size = config.get('batch_size')
        self.name = config.get('name')

        # Add tracking for thread metrics (even though sequential, for consistency)
        self.db_times = []
        self.lock_wait_times = []

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        # Reset tracking arrays
        self.db_times = []
        self.lock_wait_times = []

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

        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (successful_operations / len(relationships)) * 100

        # Calculate thread metrics
        db_insertion_time_total = sum(self.db_times)
        db_lock_wait_time = sum(self.lock_wait_times)

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",
            run_number=0,
            thread_count=1,  # Sequential is single-threaded
            batch_size=self.batch_size,

            total_time=total_time,
            throughput=throughput,
            success_rate=success_rate,

            processing_overhead=0.0,
            conflicts=actual_conflicts,

            db_insertion_time_total=db_insertion_time_total,
            db_lock_wait_time=db_lock_wait_time,

            system_cores_avg=resource_metrics.get("system_cores_avg")
        )

    def _insert_batch(self, batch: List[Dict]) -> tuple:
        conflicts = 0
        successes = 0
        total_db_time = 0
        total_lock_wait = 0

        db_start = time.time()

        query = """
                UNWIND $batch AS rel
                MERGE (from:Entity {title: rel.from})
                  ON CREATE SET from.isBase = false, from.processed_at = timestamp()
                  ON MATCH SET 
                    from.isBase = COALESCE(from.isBase, false),
                    from.processed_at = COALESCE(from.processed_at, timestamp())

                MERGE (to:Entity {title: rel.to})
                  ON CREATE SET to.isBase = false, to.processed_at = timestamp()
                  ON MATCH SET 
                    to.isBase = COALESCE(to.isBase, false),
                    to.processed_at = COALESCE(to.processed_at, timestamp())

                MERGE (from)-[r:LINKS_TO]->(to)
                  ON CREATE SET r.created_at = timestamp(), r.weight = 1
                  ON MATCH SET r.last_updated = timestamp(), r.weight = r.weight + 1
         """

        try:
            with self.driver.session() as session:
                session.run(query, {'batch': batch})

            db_time = time.time() - db_start
            total_db_time += db_time
            successes = len(batch)

        except Exception as e:
            db_time = time.time() - db_start
            total_db_time += db_time

            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['lock', 'deadlock', 'timeout']):
                total_lock_wait += db_time
                conflicts = 1  # One conflict for the whole batch
            else:
                # Other error types (syntax, driver, etc.)
                pass

        # Record metrics
        self.db_times.append(total_db_time)
        self.lock_wait_times.append(total_lock_wait)

        return conflicts, successes