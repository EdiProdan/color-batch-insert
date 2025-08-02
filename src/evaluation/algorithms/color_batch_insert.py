import concurrent.futures
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

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        # print(f"\n--- {self.name} ---")
        # print(f"Processing {len(relationships)} relationships")
        # print(f"Configuration: {self.thread_count} threads")


        monitor = ResourceMonitor()
        monitor.start_monitoring()

        start_time = time.time()

        # Step 1: Incremental coloring
        # print("\nPhase 1: Incremental coloring...")
        color_start = time.time()
        coloring = self._incremental_coloring(relationships)
        color_time = time.time() - color_start

        num_colors = len(set(coloring.values()))
        # print(f"  Coloring completed in {color_time:.2f}s")
        print(f"  Used {num_colors} colors")

        # Step 2: Group by color
        color_groups = self._group_by_color(relationships, coloring)
        # for color, group in sorted(color_groups.items()):
        #     print(f"  Color {color}: {len(group)} relationships")

        # Step 3: Process each color group
        # print("\nPhase 2: Processing color groups...")
        total_conflicts = 0
        total_successful = 0

        for color in sorted(color_groups.keys()):
            group = color_groups[color]
            # print(f"  Processing color {color} ({len(group)} relationships)...")

            conflicts, successful = self._process_color_group(group)
            total_conflicts += conflicts
            total_successful += successful

        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (total_successful / len(relationships)) * 100

        # print(f"\nResults:")
        # print(f"  Total time: {total_time:.1f}s")
        # print(f"  Preprocessing time: {color_time:.1f}s")
        # print(f"  Throughput: {throughput:.1f} rel/s")
        # print(f"  Success rate: {success_rate:.1f}%")
        # print(f"  Conflicts: {total_conflicts}")
        # print(f"  Colors used: {num_colors}")

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",
            run_number=0,

            total_time=total_time,
            throughput=throughput,
            success_rate=success_rate,

            processing_overhead=color_time,
            conflicts=total_conflicts,
            retries=0,
            memory_peak=resource_metrics['memory_peak'],
            cpu_avg=resource_metrics['cpu_avg']
        )

    def _incremental_coloring(self, relationships: List[Dict]) -> Dict[int, int]:
        return incremental_coloring(relationships)

    def _group_by_color(self, relationships: List[Dict], coloring: Dict[int, int]) -> Dict[int, List[Dict]]:
        color_groups = defaultdict(list)

        for i, rel in enumerate(relationships):
            color = coloring[i]
            color_groups[color].append(rel)

        return dict(color_groups)

    def _process_color_group(self, relationships: List[Dict]) -> tuple:

        batches = [relationships[i:i + self.batch_size]
                   for i in range(0, len(relationships), self.batch_size)]

        total_conflicts = 0
        total_successful = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            futures = [executor.submit(self._insert_batch, batch) for batch in batches]

            for future in concurrent.futures.as_completed(futures):
                try:
                    conflicts, successful = future.result(timeout=60)
                    total_conflicts += conflicts
                    total_successful += successful
                except Exception as e:
                    # print(f"    Batch failed: {str(e)[:50]}")
                    pass

        return total_conflicts, total_successful

    def _insert_batch(self, batch: List[Dict]) -> tuple:
        conflicts = 0
        successful = 0

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

            successful += len(batch)

        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['lock', 'deadlock', 'timeout']):
                    conflicts += 1


        return conflicts, successful
