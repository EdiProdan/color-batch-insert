import time
from typing import List, Dict, Set, Tuple
from collections import defaultdict

from src.evaluation.framework import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class MixAndBatchInsert(AlgorithmBase):

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.name = config.get('name', 'Mix and Batch Algorithm')

        # Core parameters from the original algorithm
        self.num_partitions = config.get('num_partitions', 10)  # 10x10 grid
        self.max_workers = config.get('max_workers', 6)  # Threads per batch
        self.batch_size = config.get('batch_size', 1000)  # Relationships per transaction
        self.hash_digits = config.get('hash_digits', 1)  # How many digits to use

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        """Execute Mix and Batch algorithm"""
        print(f"\n--- {self.name} ---")
        print(f"Processing {len(relationships)} relationships")
        print(f"Configuration: {self.num_partitions}x{self.num_partitions} grid, "
              f"{self.max_workers} workers")

        # Clear database
        self.clear_database()

        # Start monitoring
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        start_time = time.time()

        # Step 1 & 2: Add partition codes to relationships
        prep_start = time.time()
        partitioned_rels = self._add_partition_codes(relationships)
        prep_time = time.time() - prep_start

        # Step 3 & 4: Generate diagonal batches
        diagonal_batches = self._generate_diagonal_batches()
        print(f"Created {len(diagonal_batches)} diagonal batches")

        # Step 5 & 6: Process each batch
        batch_times = []
        total_conflicts = 0
        total_successful = 0

        for batch_idx, partition_codes in enumerate(diagonal_batches):
            batch_start = time.time()

            # Filter relationships for this batch
            batch_rels = self._filter_relationships_for_batch(partitioned_rels, partition_codes)

            if not batch_rels:
                continue

            print(f"\nProcessing batch {batch_idx + 1}/{len(diagonal_batches)} "
                  f"({len(batch_rels)} relationships)")

            # Process this batch in parallel
            conflicts, successful = self._process_batch_parallel(batch_rels, partition_codes)

            total_conflicts += conflicts
            total_successful += successful
            batch_times.append(time.time() - batch_start)

        # Calculate final metrics
        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        # Results
        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (total_successful / len(relationships)) * 100

        print(f"\nResults:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Preprocessing: {prep_time:.1f}s")
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
            processing_overhead_time=prep_time,  # Partitioning overhead
            actual_conflicts=total_conflicts,
            retry_count=0,  # Mix and Batch prevents conflicts
            adaptation_events=0,  # Static algorithm
            final_parallelism=self.max_workers,
            memory_peak=resource_metrics.get('memory_peak', 0),
            cpu_avg=resource_metrics.get('cpu_avg', 0),
            batch_processing_times=batch_times
        )

    def _add_partition_codes(self, relationships: List[Dict]) -> List[Dict]:
        """
        Step 1 & 2: Add partition codes to each relationship.
        Uses last digit(s) to determine partition.
        """
        partitioned = []

        for rel in relationships:
            # Get last digit(s) for partitioning
            from_partition = self._get_partition_id(rel['from'])
            to_partition = self._get_partition_id(rel['to'])

            # Create partition code
            partition_code = f"{from_partition}-{to_partition}"

            # Add to relationship
            rel_with_partition = rel.copy()
            rel_with_partition['partition_code'] = partition_code
            partitioned.append(rel_with_partition)

        return partitioned

    def _get_partition_id(self, entity_name: str) -> int:
        """
        Get partition ID for an entity using last digit(s).
        This matches the original algorithm's approach.
        """
        # Extract numeric part from entity name
        # Handle formats like "C20138" or just numbers
        try:
            numeric_part = ''.join(filter(str.isdigit, str(entity_name)))

            if not numeric_part:
                # If no digits, use hash of the string
                return hash(entity_name) % self.num_partitions

            # Use last digit(s)
            last_digits = numeric_part[-self.hash_digits:]
            return int(last_digits) % self.num_partitions
        except ValueError:
            return hash(entity_name) % self.num_partitions


    def _generate_diagonal_batches(self) -> List[Set[str]]:
        """
        Step 3 & 4: Generate diagonal batches from the grid.
        Each batch contains partition codes that share no rows or columns.
        """
        batches = []

        # For a square grid, we need num_partitions batches
        for batch_num in range(self.num_partitions):
            batch_partitions = set()

            # Select diagonal elements
            for i in range(self.num_partitions):
                # Calculate column using modulo to wrap around
                j = (i + batch_num) % self.num_partitions
                partition_code = f"{i}-{j}"
                batch_partitions.add(partition_code)

            batches.append(batch_partitions)

        return batches

    def _filter_relationships_for_batch(self, relationships: List[Dict],
                                        partition_codes: Set[str]) -> List[Dict]:
        """Filter relationships that belong to the given partition codes"""
        return [rel for rel in relationships
                if rel.get('partition_code') in partition_codes]

    def _process_batch_parallel(self, batch_rels: List[Dict],
                                partition_codes: Set[str]) -> Tuple[int, int]:
        """
        Process a batch in parallel using threads.
        Each partition code gets its own thread.
        """
        import concurrent.futures

        # Group relationships by partition code
        partitioned_groups = defaultdict(list)
        for rel in batch_rels:
            partitioned_groups[rel['partition_code']].append(rel)

        total_conflicts = 0
        total_successful = 0

        # Process each partition in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}

            # Submit each partition to a thread
            for partition_code, partition_rels in partitioned_groups.items():
                future = executor.submit(self._process_partition,
                                         partition_rels, partition_code)
                futures[future] = partition_code

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                partition_code = futures[future]
                try:
                    conflicts, successful = future.result()
                    total_conflicts += conflicts
                    total_successful += successful
                    print(
                        f"  Partition {partition_code}: {successful}/{len(partitioned_groups[partition_code])} successful")
                except Exception as e:
                    print(f"  Partition {partition_code} failed: {str(e)}")

        return total_conflicts, total_successful

    def _process_partition(self, partition_rels: List[Dict],
                           partition_code: str) -> Tuple[int, int]:
        """
        Process all relationships in a single partition.
        This runs on a single thread and won't conflict with other partitions.
        """
        conflicts = 0
        successful = 0
        experiment_simple_parallelid = f"mix_batch_{partition_code}_{int(time.time() * 1000)}"

        # Process in smaller batches for transaction management
        for i in range(0, len(partition_rels), self.batch_size):
            batch = partition_rels[i:i + self.batch_size]

            with self.driver.session() as session:
                for rel in batch:
                    try:
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

                        successful += 1

                    except Exception as e:
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ['lock', 'deadlock', 'timeout']):
                            conflicts += 1
                            print(f"    Unexpected conflict in {partition_code}: {str(e)[:50]}")

        return conflicts, successful
