import time
from typing import List, Dict, Tuple
from src.evaluation import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class ApocInsert(AlgorithmBase):

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.batch_size = config.get('batch_size')
        self.thread_count = config.get('thread_count')
        self.name = config.get('name')
        self.max_retries = config.get('max_retries')

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        start_time = time.time()

        prep_start = time.time()
        prep_time = time.time() - prep_start

        conflicts, successful = self._execute_apoc_batch_processing(relationships)

        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (successful / len(relationships)) * 100 if relationships else 0

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",
            run_number=0,

            total_time=total_time,
            throughput=throughput,
            success_rate=success_rate,

            processing_overhead=prep_time,
            conflicts=conflicts,
            retries=conflicts,

            memory_peak=resource_metrics['memory_peak'],
            cpu_avg=resource_metrics['cpu_avg']
        )



    def _execute_apoc_batch_processing(self, relationships: List[Dict]) -> Tuple[int, int]:

        print("  Executing APOC batch processing...")

        conflicts = 0
        successful = 0

        with self.driver.session() as session:
            try:
                # Method 1: Using apoc.periodic.iterate with direct data
                result = session.run("""
                    CALL apoc.periodic.iterate(
                        // Query to generate relationship data
                        "UNWIND $relationships AS rel RETURN rel",

                        // Update query for each batch
                        "
                        MERGE (from:Entity {title: rel.from})
                        ON CREATE SET from.isBase = false, from.processed_at = timestamp()
                        ON MATCH SET from.isBase = COALESCE(from.isBase, false)

                        MERGE (to:Entity {title: rel.to})
                        ON CREATE SET to.isBase = false, to.processed_at = timestamp()
                        ON MATCH SET to.isBase = COALESCE(to.isBase, false)

                        MERGE (from)-[r:LINKS_TO]->(to)
                        ON CREATE SET r.created = timestamp()
                        ",

                        // Configuration
                        {
                            batchSize: $batch_size,
                            parallel: true,
                            parallelWorkers: $parallel_workers,
                            retries: $retry_count,
                            params: {relationships: $relationships}
                        }
                    )
                    YIELD batches, total, timeTaken, committedOperations, failedOperations, 
                          failedBatches, retries, errorMessages
                    RETURN batches, total, timeTaken, committedOperations, failedOperations, 
                           failedBatches, retries, errorMessages
                """, {
                    'relationships': relationships,
                    'batch_size': self.batch_size,
                    'parallel_workers': self.thread_count,
                    'retry_count': self.max_retries
                })

                # Process APOC results
                apoc_result = result.single()

                if apoc_result:
                    successful = apoc_result.get('committedOperations', 0)
                    failed = apoc_result.get('failedOperations', 0)
                    conflicts = failed

                    error_messages = apoc_result.get('errorMessages', {})
                    if error_messages:
                        pass

            except Exception as e:
                pass

        return conflicts, successful
