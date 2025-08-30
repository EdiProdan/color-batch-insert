import asyncio
import time
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict

from src.evaluation import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class MixAndBatchInsert(AlgorithmBase):

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.num_partitions = config.get('num_partitions')
        self.thread_count = config.get('thread_count')
        self.batch_size = config.get('batch_size')
        self.hash_digits = config.get('hash_digits')
        self.max_retries = config.get('max_retries')
        self.name = config.get('name')

        self.thread_times = []
        self.db_times = []
        self.lock_wait_times = []

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        return asyncio.run(self._async_insert_relationships(relationships))

    async def _async_insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        self.thread_times = []
        self.db_times = []
        self.lock_wait_times = []

        start_time = time.time()

        # Step 1 & 2: Add partition codes to relationships
        prep_start = time.time()
        partitioned_rels = self._add_partition_codes(relationships)
        prep_time = time.time() - prep_start

        # Step 3 & 4: Generate diagonal batches
        diagonal_batches = self._generate_diagonal_batches()

        # Step 5 & 6: Process each batch
        batch_times = []
        total_conflicts = 0
        total_successful = 0

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        for batch_idx, partition_codes in enumerate(diagonal_batches):
            batch_start = time.time()

            batch_rels = self._filter_relationships_for_batch(partitioned_rels, partition_codes)

            if not batch_rels:
                continue

            conflicts, successful, retries = await self._process_batch_parallel(batch_rels, partition_codes)

            total_conflicts += conflicts
            total_successful += successful
            batch_times.append(time.time() - batch_start)

        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        throughput = len(relationships) / (total_time - prep_time)
        success_rate = (total_successful / len(relationships)) * 100

        thread_wait_time_total = sum(self.lock_wait_times)
        db_insertion_time_total = sum(self.db_times)
        db_lock_wait_time = sum(self.lock_wait_times)

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",
            run_number=0,
            thread_count=self.thread_count,
            batch_size=self.batch_size,

            total_time=total_time,
            throughput=throughput,
            success_rate=success_rate,

            processing_overhead=prep_time,
            conflicts=total_conflicts,

            # New thread metrics
            db_insertion_time_total=db_insertion_time_total,
            db_lock_wait_time=db_lock_wait_time,

            # Existing fields
            system_cores_avg=resource_metrics.get("system_cores_avg")
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

    async def _process_batch_parallel(self, batch_rels: List[Dict],
                                      partition_codes: Set[str]):

        partitioned_groups = defaultdict(list)
        for rel in batch_rels:
            partitioned_groups[rel['partition_code']].append(rel)

        total_conflicts = 0
        total_successful = 0
        total_retries = 0

        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.thread_count)

        tasks = []
        for partition_code, partition_rels in partitioned_groups.items():
            task = self._process_partition_async(partition_rels, partition_code, semaphore)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                pass
            else:
                conflicts, successful, retries = result
                total_conflicts += conflicts
                total_successful += successful
                total_retries += retries

        return total_conflicts, total_successful, total_retries

    async def _process_partition_async(self, partition_rels: List[Dict], partition_code: str, semaphore):
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._process_partition, partition_rels, partition_code)

    def _process_partition(self, partition_rels: List[Dict], partition_code: str):
        thread_start = time.time()
        conflicts = 0
        retries = 0
        successful = 0
        total_db_time = 0
        total_lock_wait = 0

        for i in range(0, len(partition_rels), self.batch_size):
            batch = partition_rels[i:i + self.batch_size]

            retry_count = 0
            succeeded = False

            while retry_count <= self.max_retries and not succeeded:
                db_start = time.time()

                try:
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

                    with self.driver.session() as session:
                        session.run(query, {'batch': batch})

                    db_time = time.time() - db_start
                    total_db_time += db_time
                    successful += len(batch)
                    succeeded = True

                except Exception as e:
                    db_time = time.time() - db_start
                    total_db_time += db_time

                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ['lock', 'deadlock', 'timeout']):
                        total_lock_wait += db_time
                        conflicts += len(batch)

                        if retry_count < self.max_retries:
                            retry_count += 1
                            retries += 1
                            time.sleep(0.1)
                        else:
                            break
                    else:
                        break

        thread_total_time = time.time() - thread_start

        self.thread_times.append(thread_total_time)
        self.db_times.append(total_db_time)
        self.lock_wait_times.append(total_lock_wait)

        return conflicts, successful, retries