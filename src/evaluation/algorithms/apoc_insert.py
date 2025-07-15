import time
from typing import List, Dict, Tuple
from src.evaluation import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class ApocInsert(AlgorithmBase):

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.batch_size = config.get('batch_size')
        self.thread_count = config.get('thread_count')
        self.name = config.get('name')

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        print(f"\n--- {self.name} ---")
        print(f"Processing {len(relationships)} relationships")
        print(f"Configuration: batch size {self.batch_size}, {self.thread_count} threads")

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        start_time = time.time()

        prep_start = time.time()
        self._prepare_data_for_apoc(relationships)
        prep_time = time.time() - prep_start

        processing_start = time.time()
        conflicts, successful = self._execute_apoc_batch_processing(relationships)
        processing_time = time.time() - processing_start

        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (successful / len(relationships)) * 100 if relationships else 0

        print(f"\nResults:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Preparation time: {prep_time:.1f}s")
        print(f"  Processing time: {processing_time:.1f}s")
        print(f"  Throughput: {throughput:.1f} rel/s")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Conflicts detected: {conflicts}")

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",
            run_number=0,

            total_time=total_time,
            throughput=throughput,
            success_rate=success_rate,

            processing_overhead=prep_time,
            conflicts=conflicts,

            memory_peak=resource_metrics['memory_peak'],
            cpu_avg=resource_metrics['cpu_avg']
        )

    def _prepare_data_for_apoc(self, relationships: List[Dict]):

        print("  Preparing data for APOC processing...")

        # For this implementation, we'll pass data directly to APOC
        # In more advanced scenarios, you might want to:
        # 1. Create temporary nodes with the data
        # 2. Use APOC's virtual graph features
        # 3. Load data into collections for processing

        # Here we're keeping it simple and will pass the data directly
        pass

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
                    'parallel_workers': self.parallel_workers,
                    'retry_count': self.retry_count
                })

                # Process APOC results
                apoc_result = result.single()

                if apoc_result:
                    successful = apoc_result.get('committedOperations', 0)
                    failed = apoc_result.get('failedOperations', 0)

                    # APOC doesn't directly report conflicts, but failed operations
                    # can indicate conflicts or other issues
                    conflicts = failed

                    print(f"    APOC Results:")
                    print(f"      Batches processed: {apoc_result.get('batches', 0)}")
                    print(f"      Total operations: {apoc_result.get('total', 0)}")
                    print(f"      Time taken: {apoc_result.get('timeTaken', 0)}ms")
                    print(f"      Committed: {successful}")
                    print(f"      Failed: {failed}")
                    print(f"      Retries: {apoc_result.get('retries', 0)}")

                    error_messages = apoc_result.get('errorMessages', {})
                    if error_messages:
                        print(f"      Error messages: {error_messages}")

            except Exception as e:
                print(f"    APOC processing failed: {str(e)}")
                # Fallback to manual conflict detection if APOC fails

        return conflicts, successful
