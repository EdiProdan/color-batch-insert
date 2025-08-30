import asyncio
import time
from collections import defaultdict
from typing import List, Dict, Set
from .coloring_cy import incremental_coloring
from src.evaluation import AlgorithmBase, PerformanceMetrics, ResourceMonitor


class ColorBatchInsert(AlgorithmBase):

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.thread_count = config.get('thread_count')
        self.name = config.get('name')
        self.batch_size = config.get('batch_size')

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

        # Step 1: Incremental coloring
        preprocess_start = time.time()
        coloring = self._incremental_coloring(relationships)

        num_colors = len(set(coloring.values()))
        print(f"  Used {num_colors} colors")

        # Step 2: Group by color
        color_groups = self._group_by_color(relationships, coloring)

        preprocess_time = time.time() - preprocess_start

        # Step 3: Process each color group sequentially
        total_conflicts = 0
        total_successful = 0

        monitor = ResourceMonitor()
        monitor.start_monitoring()
        for color in sorted(color_groups.keys()):
            group = color_groups[color]

            conflicts, successful = await self._process_color_group(group)
            total_conflicts += conflicts
            total_successful += successful

        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        throughput = len(relationships) / (total_time - preprocess_time)
        success_rate = (total_successful / len(relationships)) * 100

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

            processing_overhead=preprocess_time,
            conflicts=total_conflicts,

            db_insertion_time_total=db_insertion_time_total,
            db_lock_wait_time=db_lock_wait_time,

            system_cores_avg=resource_metrics.get("system_cores_avg")
        )

    def _incremental_coloring(self, relationships: List[Dict]) -> Dict[int, int]:
        return incremental_coloring(relationships)

    def _group_by_color(self, relationships: List[Dict], coloring: Dict[int, int]) -> Dict[int, List[Dict]]:
        color_groups = defaultdict(list)

        for i, rel in enumerate(relationships):
            color = coloring[i]
            color_groups[color].append(rel)

        return dict(color_groups)

    async def _process_color_group(self, relationships: List[Dict]) -> tuple:
        batches = [relationships[i:i + self.batch_size]
                   for i in range(0, len(relationships), self.batch_size)]

        total_conflicts = 0
        total_successful = 0

        semaphore = asyncio.Semaphore(self.thread_count)

        tasks = [self._insert_batch_async(batch, semaphore) for batch in batches]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            else:
                conflicts, successful = result
                total_conflicts += conflicts
                total_successful += successful

        return total_conflicts, total_successful

    async def _insert_batch_async(self, batch: List[Dict], semaphore) -> tuple:
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._insert_batch, batch)

    def _insert_batch(self, batch: List[Dict]) -> tuple:
        thread_start = time.time()
        conflicts = 0
        successful = 0
        total_db_time = 0
        total_lock_wait = 0

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

        except Exception as e:
            db_time = time.time() - db_start
            total_db_time += db_time

            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['lock', 'deadlock', 'timeout']):
                total_lock_wait += db_time
                conflicts += len(batch)

        thread_total_time = time.time() - thread_start

        self.thread_times.append(thread_total_time)
        self.db_times.append(total_db_time)
        self.lock_wait_times.append(total_lock_wait)

        return conflicts, successful