import asyncio
import time
from typing import List, Dict
import random

from src.evaluation import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class NaiveParallelInsert(AlgorithmBase):

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.batch_size = config.get('batch_size')
        self.thread_count = config.get('thread_count')
        self.max_retries = config.get('max_retries')
        self.name = config.get('name')

        # Add tracking for thread metrics
        self.thread_times = []
        self.db_times = []
        self.lock_wait_times = []

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        return asyncio.run(self._async_insert_relationships(relationships))

    async def _async_insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        # Reset tracking arrays
        self.thread_times = []
        self.db_times = []
        self.lock_wait_times = []

        monitor = ResourceMonitor()
        monitor.start_monitoring()

        start_time = time.time()

        # Shuffle the list in-place
        random.shuffle(relationships)

        # Split into batches
        batches = []
        for i in range(0, len(relationships), self.batch_size):
            batch = relationships[i:i + self.batch_size]
            batches.append(batch)

        results = await self._execute_parallel_batches(batches)

        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        total_successful = sum(r['successful'] for r in results)
        total_conflicts = sum(r['conflicts'] for r in results)
        total_retries = sum(r['retries'] for r in results)

        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (total_successful / len(relationships)) * 100

        # Calculate thread metrics
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

            processing_overhead=0.0,
            conflicts=total_conflicts,

            db_insertion_time_total=db_insertion_time_total,
            db_lock_wait_time=db_lock_wait_time,

            system_cores_avg=resource_metrics.get("system_cores_avg")
        )

    async def _execute_parallel_batches(self, batches: List[List[Dict]]) -> List[Dict]:
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.thread_count)

        # Create tasks for all batches
        tasks = []
        for i, batch in enumerate(batches):
            task = self._process_single_batch_async(i, batch, semaphore)
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    'successful': 0,
                    'conflicts': 0,
                    'time': 0,
                    'retries': 0
                })
            else:
                processed_results.append(result)

        return processed_results

    async def _process_single_batch_async(self, batch_id: int, batch: List[Dict], semaphore) -> Dict:
        async with semaphore:
            # Run the blocking database operation in a thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._process_single_batch, batch_id, batch)

    def _process_single_batch(self, batch_id: int, batch: List[Dict]) -> Dict:
        batch_start = time.time()

        conflicts, successful, retries = self._insert_batch(batch)

        processing_time = time.time() - batch_start

        return {
            'successful': successful,
            'conflicts': conflicts,
            'time': processing_time,
            'retries': retries,
        }

    def _insert_batch(self, batch: List[Dict]) -> tuple:
        thread_start = time.time()
        total_conflicts = 0
        total_retries = 0
        successes = 0
        total_db_time = 0
        total_lock_wait = 0

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
                successes += len(batch)
                succeeded = True

            except Exception as e:
                db_time = time.time() - db_start
                total_db_time += db_time

                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['lock', 'deadlock', 'timeout']):
                    total_lock_wait += db_time
                    total_conflicts += len(batch)  # One conflict for this whole batch attempt

                    if retry_count < self.max_retries:
                        retry_count += 1
                        total_retries += 1
                        time.sleep(0.1)  # Keep synchronous sleep since we're in executor
                    else:
                        break  # Max retries reached
                else:
                    break  # Non-retryable error

        # Record thread metrics
        thread_total_time = time.time() - thread_start
        self.thread_times.append(thread_total_time)
        self.db_times.append(total_db_time)
        self.lock_wait_times.append(total_lock_wait)

        return total_conflicts, successes, total_retries