import concurrent.futures
import time
from typing import List, Dict, Set, Tuple
from collections import defaultdict

from src.evaluation import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class MixAndBatchInsert(AlgorithmBase):

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.num_partitions = config.get('num_partitions')
        self.thread_count = config.get('thread_count')
        self.batch_size = config.get('batch_size')
        self.hash_digits = config.get('hash_digits')
        self.name = config.get('name')

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        print(f"\n--- {self.name} ---")
        print(f"Processing {len(relationships)} relationships")
        print(f"Configuration: {self.num_partitions}x{self.num_partitions} grid, "
              f"{self.thread_count} threads")

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

            batch_rels = self._filter_relationships_for_batch(partitioned_rels, partition_codes)

            if not batch_rels:
                continue

            print(f"\nProcessing batch {batch_idx + 1}/{len(diagonal_batches)} "
                  f"({len(batch_rels)} relationships)")

            conflicts, successful = self._process_batch_parallel(batch_rels, partition_codes)

            total_conflicts += conflicts
            total_successful += successful
            batch_times.append(time.time() - batch_start)

        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

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

            processing_overhead=prep_time,
            conflicts=total_conflicts,

            memory_peak=resource_metrics['memory_peak'],
            cpu_avg=resource_metrics['cpu_avg']
        )

    def _add_partition_codes(self, relationships: List[Dict]) -> List[Dict]:
        partitioned = []

        for rel in relationships:
            from_partition = self._get_partition_id(rel['from'])
            to_partition = self._get_partition_id(rel['to'])

            partition_code = f"{from_partition}-{to_partition}"

            rel_with_partition = rel.copy()
            rel_with_partition['partition_code'] = partition_code
            partitioned.append(rel_with_partition)

        return partitioned

    def _get_partition_id(self, entity_name: str) -> int:

        try:
            numeric_part = ''.join(filter(str.isdigit, str(entity_name)))

            if not numeric_part:
                return hash(entity_name) % self.num_partitions

            last_digits = numeric_part[-self.hash_digits:]
            return int(last_digits) % self.num_partitions
        except ValueError:
            return hash(entity_name) % self.num_partitions


    def _generate_diagonal_batches(self) -> List[Set[str]]:

        batches = []

        for batch_num in range(self.num_partitions):
            batch_partitions = set()

            for i in range(self.num_partitions):
                j = (i + batch_num) % self.num_partitions
                partition_code = f"{i}-{j}"
                batch_partitions.add(partition_code)

            batches.append(batch_partitions)

        return batches

    def _filter_relationships_for_batch(self, relationships: List[Dict],
                                        partition_codes: Set[str]) -> List[Dict]:
        return [rel for rel in relationships
                if rel.get('partition_code') in partition_codes]

    def _process_batch_parallel(self, batch_rels: List[Dict],
                                partition_codes: Set[str]) -> Tuple[int, int]:


        partitioned_groups = defaultdict(list)
        for rel in batch_rels:
            partitioned_groups[rel['partition_code']].append(rel)

        total_conflicts = 0
        total_successful = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            futures = {}

            for partition_code, partition_rels in partitioned_groups.items():
                future = executor.submit(self._process_partition,
                                         partition_rels, partition_code)
                futures[future] = partition_code

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
        conflicts = 0
        successful = 0

        for i in range(0, len(partition_rels), self.batch_size):
            batch = partition_rels[i:i + self.batch_size]

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
                        successful += 1

                    except Exception as e:
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ['lock', 'deadlock', 'timeout']):
                            conflicts += 1
                            print(f"    Unexpected conflict in {partition_code}: {str(e)[:50]}")

        return conflicts, successful
