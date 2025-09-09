import time
from typing import List, Dict, Tuple
from src.evaluation import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class ApocSequentialInsert(AlgorithmBase):

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.batch_size = config.get('batch_size')
        self.name = config.get('name')
        self.max_retries = config.get('max_retries')
        self.thread_count = config.get('thread_count')

        self.db_times = []
        self.lock_wait_times = []

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        self.db_times = []
        self.lock_wait_times = []

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        start_time = time.time()

        conflicts, successful = self._execute_apoc_batch_processing(relationships)

        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (successful / len(relationships)) * 100 if relationships else 0

        # Calculate thread metrics
        db_insertion_time_total = sum(self.db_times)
        db_lock_wait_time = sum(self.lock_wait_times)

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",
            run_number=0,
            thread_count=1,
            batch_size=self.batch_size,

            total_time=total_time,
            throughput=throughput,
            success_rate=success_rate,

            processing_overhead=0.0,
            conflicts=conflicts,

            db_insertion_time_total=db_insertion_time_total,
            db_lock_wait_time=db_lock_wait_time,

            system_cores_avg=resource_metrics.get("system_cores_avg")
        )

    def _execute_apoc_batch_processing(self, relationships: List[Dict]) -> Tuple[int, int]:
        print("  Executing APOC sequential batch processing...")

        conflicts = 0
        successful = 0

        db_start = time.time()

        with self.driver.session() as session:
            try:
                result = session.run("""
                    CALL apoc.periodic.iterate(
                        "UNWIND $relationships AS rel RETURN rel",
                        "
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
                        ",
                        {
                            batchSize: $batch_size,
                            parallel: $parallel,
                            concurrency: $concurrency,  // Changed from parallelWorkers
                            retries: $max_retries,
                            params: {relationships: $relationships}  // Nested properly
                        }
                    )
                    YIELD batches, total, timeTaken, committedOperations, failedOperations, 
                          failedBatches, retries, errorMessages
                    RETURN batches, total, timeTaken, committedOperations, failedOperations, 
                           failedBatches, retries, errorMessages
                """, {
                    'relationships': relationships,
                    'batch_size': self.batch_size,
                    'parallel': False,
                    'concurrency': self.thread_count,
                    'max_retries': self.max_retries
                })

                apoc_result = result.single()
                db_time = time.time() - db_start

                if apoc_result:
                    successful = apoc_result.get('committedOperations', 0)
                    failed = apoc_result.get('failedOperations', 0)
                    conflicts = failed

                    estimated_lock_wait = db_time * (failed / len(relationships)) if relationships else 0

                    error_messages = apoc_result.get('errorMessages', {})
                    if error_messages:
                        print(f"      APOC Error messages: {error_messages}")

                    print(f"      APOC Results: {apoc_result.get('batches', 0)} batches, "
                          f"{successful} successful, {failed} failed")

                    self.db_times.append(db_time)
                    self.lock_wait_times.append(estimated_lock_wait)
                else:
                    self.db_times.append(db_time)
                    self.lock_wait_times.append(0)

            except Exception as e:
                db_time = time.time() - db_start
                print(f"      APOC Exception: {str(e)}")
                conflicts = len(relationships)

                self.db_times.append(db_time)

                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['lock', 'deadlock', 'timeout']):
                    self.lock_wait_times.append(db_time)
                else:
                    self.lock_wait_times.append(0)

        return conflicts, successful