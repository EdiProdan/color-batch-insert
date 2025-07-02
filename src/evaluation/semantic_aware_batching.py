import concurrent.futures
import random
import time
from collections import defaultdict
from typing import List, Dict, Set, Tuple

from src.evaluation.framework import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class SemanticAwareBatching(AlgorithmBase):

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.thread_count = config.get('thread_count', 10)
        self.max_retries = config.get('max_retries', 3)
        self.name = config.get('name', 'Simple Color Semantic Algorithm')

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        print(f"\n--- Simple Color Semantic Algorithm ---")
        print(f"Processing {len(relationships)} relationships")
        print(f"Configuration: {self.thread_count} threads, max retries {self.max_retries}")

        self.clear_database()

        monitor = ResourceMonitor()
        monitor.start_monitoring()
        start_time = time.time()

        print("\nPhase 1: Building conflict graph...")
        graph_start = time.time()
        conflict_graph = self._build_conflict_graph(relationships)
        graph_time = time.time() - graph_start
        print(f"  Conflict graph built in {graph_time:.2f}s")
        print(f"  Graph has {len(conflict_graph)} nodes")

        print("\nPhase 2: Coloring conflict graph...")
        color_start = time.time()
        coloring = self._greedy_coloring(conflict_graph)
        color_time = time.time() - color_start
        print(f"  Coloring completed in {color_time:.2f}s")
        print(f"  Used {len(set(coloring.values()))} colors")

        print("\nPhase 3: Grouping relationships by color...")
        color_groups = self._group_by_color(relationships, coloring)
        for color, group in sorted(color_groups.items()):
            print(f"  Color {color}: {len(group)} relationships")

        print("\nPhase 4: Processing color groups...")
        results = []
        total_conflicts = 0
        total_retries = 0
        total_successful = 0

        for color in sorted(color_groups.keys()):
            group = color_groups[color]
            print(f"\n  Processing color {color} ({len(group)} relationships)...")

            color_results = self._process_color_group(group, color)

            for r in color_results:
                total_successful += r['successful']
                total_conflicts += r['conflicts']
                total_retries += r['retries']
                results.append(r)

        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (total_successful / len(relationships)) * 100
        preprocessing_time = graph_time + color_time

        # Results summary
        print(f"\nResults:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Preprocessing time: {preprocessing_time:.1f}s")
        print(f"  Throughput: {throughput:.1f} rel/s")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Conflicts: {total_conflicts}")
        print(f"  Retries: {total_retries}")
        print(f"  Colors used: {len(color_groups)}")

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",  # Set by framework
            run_number=0,  # Set by framework
            total_time=total_time,
            throughput=throughput,
            success_rate=success_rate,
            processing_overhead_time=preprocessing_time,
            actual_conflicts=total_conflicts,
            retry_count=total_retries,
            adaptation_events=len(color_groups),  # Number of color phases
            final_parallelism=self.thread_count,
            memory_peak=resource_metrics.get('memory_peak', 0),
            cpu_avg=resource_metrics.get('cpu_avg', 0),
            batch_processing_times=[r['time'] for r in results if r['time'] > 0]
        )

    def _build_conflict_graph(self, relationships: List[Dict]) -> Dict[int, Set[int]]:
        node_to_rels = defaultdict(set)

        for i, rel in enumerate(relationships):
            node_to_rels[rel['from']].add(i)
            node_to_rels[rel['to']].add(i)

        conflict_graph = defaultdict(set)

        for node, rel_indices in node_to_rels.items():
            rel_list = list(rel_indices)
            for i in range(len(rel_list)):
                for j in range(i + 1, len(rel_list)):
                    conflict_graph[rel_list[i]].add(rel_list[j])
                    conflict_graph[rel_list[j]].add(rel_list[i])

        return conflict_graph

    def _greedy_coloring(self, graph: Dict[int, Set[int]]) -> Dict[int, int]:

        coloring = {}
        nodes = sorted(graph.keys(), key=lambda n: len(graph[n]), reverse=True)

        for node in nodes:
            neighbor_colors = set()
            for neighbor in graph.get(node, []):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])

            color = 0
            while color in neighbor_colors:
                color += 1

            coloring[node] = color

        return coloring

    def _group_by_color(self, relationships: List[Dict], coloring: Dict[int, int]) -> Dict[int, List[Dict]]:
        color_groups = defaultdict(list)

        for i, rel in enumerate(relationships):
            color = coloring.get(i, 0)  # default to color 0 if not in coloring
            color_groups[color].append(rel)

        return dict(color_groups)

    def _process_color_group(self, relationships: List[Dict], color: int) -> List[Dict]:
        batch_size = max(1, len(relationships) // self.thread_count)
        batches = []

        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]
            if batch:
                batches.append(batch)

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            future_to_batch = {
                executor.submit(self._process_batch_safe, i, batch, color): i
                for i, batch in enumerate(batches)
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    print(f"    Batch {batch_id} in color {color} failed: {str(e)[:50]}")
                    results.append({
                        'successful': 0,
                        'conflicts': 0,
                        'retries': 0,
                        'time': 0
                    })

        return results

    def _process_batch_safe(self, batch_id: int, batch: List[Dict], color: int) -> Dict:

        batch_start = time.time()
        conflicts, retries, successful = self._insert_batch_with_retry(batch, f"C{color}-B{batch_id}")
        processing_time = time.time() - batch_start

        if conflicts > 0:
            print(f"    WARNING: Color {color}, Batch {batch_id} had {conflicts} conflicts (unexpected!)")

        return {
            'successful': successful,
            'conflicts': conflicts,
            'retries': retries,
            'time': processing_time
        }

    def _insert_batch_with_retry(self, batch: List[Dict], batch_label: str) -> tuple:

        total_conflicts = 0
        total_retries = 0
        successes = 0

        with self.driver.session() as session:
            for rel in batch:
                retry_count = 0
                succeeded = False

                while retry_count <= self.max_retries and not succeeded:
                    try:

                        session.run("""
                                    MERGE (from:Entity {title: $from})
                                    ON CREATE SET from.isBase = false, from.processed_at = timestamp()
                                    ON MATCH SET from.isBase = COALESCE(from.isBase, false)

                                    MERGE (to:Entity {title: $to})
                                    ON CREATE SET to.isBase = false, to.processed_at = timestamp()
                                    ON MATCH SET to.isBase = COALESCE(to.isBase, false)

                                    MERGE (from)-[r:LINKS_TO]->(to)
                                    ON CREATE SET r.created = timestamp()

                                    """, {'from': rel["from"], 'to': rel["to"]})

                        succeeded = True
                        successes += 1

                    except Exception as e:
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ['lock', 'deadlock', 'timeout']):
                            total_conflicts += 1

                            if retry_count < self.max_retries:
                                retry_count += 1
                                total_retries += 1
                                time.sleep(0.1 * (2 ** retry_count))  # Exponential backoff
                            else:
                                break
                        else:
                            break

        return total_conflicts, total_retries, successes