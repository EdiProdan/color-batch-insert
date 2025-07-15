from datetime import datetime
import time
from typing import List, Dict

from src.evaluation.framework import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class SequentialInsert(AlgorithmBase):
    """
    The simplest possible sequential baseline for thesis research.

    This algorithm represents the most basic approach to relationship insertion:
    - Fixed batch size
    - Sequential processing (no parallelism)
    - No conflict detection or adaptation
    - No preprocessing or optimization

    This serves as the control group to demonstrate the value of adaptive algorithms.
    """

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.batch_size = config.get('batch_size')  # Single fixed batch size
        self.name = config.get('name', 'Simple Sequential Baseline')

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        """
        Insert relationships in the simplest possible way:
        1. Create fixed-size batches
        2. Insert each batch sequentially
        3. Track minimal metrics
        """
        print(f"\nExecuting {self.name}")
        print(f"Processing {len(relationships)} relationships")
        print(f"Batch size: {self.batch_size}")


        self.clear_database()
        # Start resource monitoring
        monitor = ResourceMonitor()
        monitor.start_monitoring()


        # Start timing
        start_time = time.time()


        #
        # Create simple fixed-size batches
        batches = [relationships[i:i + self.batch_size]
                   for i in range(0, len(relationships), self.batch_size)]




        print(f"Created {len(batches)} batches")

        # Track simple metrics
        batch_times = []
        actual_conflicts = 0
        retry_count = 0
        successful_operations = 0

        # Process each batch sequentially
        for batch_idx, batch in enumerate(batches):
            batch_start = time.time()

            # Insert the batch

            conflicts, retries, successes = self._insert_batch(batch, batch_idx)

            # Update metrics
            actual_conflicts += conflicts
            retry_count += retries
            successful_operations += successes

            # Record batch time
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Simple progress reporting
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
                print(f"  Processed {batch_idx + 1}/{len(batches)} batches")

        # Calculate final metrics
        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        # Calculate derived metrics
        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (successful_operations / len(relationships)) * 100

        print(f"\nCompleted in {total_time:.2f} seconds")
        print(f"Throughput: {throughput:.1f} relationships/second")
        print(f"Success rate: {success_rate:.1f}%")

        return PerformanceMetrics(
            # Identity
            algorithm_name=self.name,
            scenario="",  # Set by framework
            run_number=0,  # Set by framework

            # Core performance
            total_time=total_time,
            throughput=throughput,
            success_rate=success_rate,

            # No intelligence overhead (this is key!)
            processing_overhead_time=0.0,  # No preprocessing
            actual_conflicts=actual_conflicts,
            retry_count=retry_count,

            # No adaptation (static algorithm)
            adaptation_events=0,  # Never adapts
            final_parallelism=1,  # Always sequential

            # Resource usage
            memory_peak=resource_metrics['memory_peak'],
            cpu_avg=resource_metrics['cpu_avg'],

            # Batch timing details
            batch_processing_times=batch_times,

        )

    def _insert_batch(self, batch: List[Dict], batch_idx: int) -> tuple:
        """
        Insert a single batch of relationships.

        Returns:
            tuple: (conflicts, retries, successful_operations)
        """
        conflicts = 0
        retries = 0
        successes = 0

        with self.driver.session() as session:
            for rel in batch:
                try:
                    # Simple MERGE query - no optimization
                    query = """
                    MERGE (from:Entity {title: $from})
                    ON CREATE SET from.isBase = true
                    ON MATCH SET from.isBase = COALESCE(from.isBase, true)
                    
                    MERGE (to:Entity {title: $to})
                    ON CREATE SET to.isBase = $isBase
                    ON MATCH SET to.isBase = $isBase
                    
                    MERGE (from)-[r:LINKS_TO]->(to)
                    ON CREATE SET r.created = timestamp()
                    """

                    session.run(query, {
                        'from': rel['from'],
                        'to': rel['to'],
                        'isBase': False
                    })


                    successes += 1

                except Exception as e:
                    # Simple conflict detection
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ['lock', 'deadlock', 'timeout']):
                        conflicts += 1
                    # No retry logic - just fail and continue

        return conflicts, retries, successes

    # def _count_unique_entities(self, relationships: List[Dict]) -> int:
    #     """Count unique entities in relationships"""
    #     entities = set()
    #     for rel in relationships:
    #         entities.add(rel['from'])
    #         entities.add(rel['to'])
    #     return len(entities)