import concurrent.futures
import random
import time
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple

from src.evaluation.framework import PerformanceMetrics, ResourceMonitor, AlgorithmBase


class AdaptiveDynamicBatching(AlgorithmBase):
    """
    Hybrid algorithm combining Hot Node Detection with Graph Coloring for thesis research.

    Key Innovation: Two-phase approach
    1. Hot Node Detection: Identify heavily contested nodes using degree centrality
    2. Graph Coloring: Color relationships to ensure conflicting ones are in different batches

    This addresses both localized hotspots and distributed contention patterns.
    """

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.batch_size = config.get('batch_size', 1000)
        self.thread_count = config.get('thread_count', 10)
        self.max_retries = config.get('max_retries', 3)

        # Hot node detection parameters
        self.hot_node_threshold = config.get('hot_node_threshold', 0.8)  # Top 80th percentile
        self.hot_node_serial_ratio = config.get('hot_node_serial_ratio', 0.2)  # 20% of hot relationships go to serial processing

        # Graph coloring parameters
        self.min_colors = config.get('min_colors', 1)  # Minimum number of color groups

        self.name = config.get('name', 'Hybrid Hot Node + Graph Coloring')

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        """
        Execute hybrid hot node detection + graph coloring approach
        """
        print(f"\n--- {self.name} ---")
        print(f"Processing {len(relationships)} relationships")
        print(f"Configuration: {self.thread_count} threads, hot_threshold={self.hot_node_threshold}")

        # Clear previous data
        self.clear_database()

        # Start monitoring
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        start_time = time.time()

        # Phase 1: Hot Node Detection
        print("\nPhase 1: Hot Node Detection")
        hot_nodes, node_degrees = self._detect_hot_nodes(relationships)
        print(f"Detected {len(hot_nodes)} hot nodes from {len(node_degrees)} total nodes")

        # Phase 2: Separate hot vs normal relationships
        hot_relationships, normal_relationships = self._separate_hot_relationships(
            relationships, hot_nodes
        )
        print(f"Hot relationships: {len(hot_relationships)}, Normal relationships: {len(normal_relationships)}")

        # Phase 3: Graph Coloring for normal relationships
        print("\nPhase 2: Graph Coloring for Normal Relationships")
        colored_batches = self._apply_graph_coloring(normal_relationships)
        print(f"Created {len(colored_batches)} colored batches for normal relationships")

        # Phase 4: Create hot node batches (smaller, for serial/limited parallel processing)
        hot_batches = self._create_hot_node_batches(hot_relationships)
        print(f"Created {len(hot_batches)} hot node batches")

        # Phase 5: Execute hybrid processing
        print("\nPhase 3: Executing Hybrid Processing")
        results = self._execute_hybrid_processing(hot_batches, colored_batches)

        # Calculate final metrics
        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        # Aggregate results
        total_successful = sum(r['successful'] for r in results)
        total_conflicts = sum(r['conflicts'] for r in results)
        total_retries = sum(r['retries'] for r in results)
        batch_times = [r['time'] for r in results if r['time'] > 0]

        # Calculate derived metrics
        throughput = len(relationships) / total_time if total_time > 0 else 0
        success_rate = (total_successful / len(relationships)) * 100

        print(f"\nHybrid Results:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Throughput: {throughput:.1f} rel/s")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Conflicts: {total_conflicts}")
        print(f"  Retries: {total_retries}")
        print(f"  Hot nodes detected: {len(hot_nodes)}")

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",
            run_number=0,
            total_time=total_time,
            throughput=throughput,
            success_rate=success_rate,
            processing_overhead_time=0.0,
            actual_conflicts=total_conflicts,
            retry_count=total_retries,
            adaptation_events=len(hot_nodes),  # Number of hot nodes detected
            final_parallelism=self.thread_count,
            memory_peak=resource_metrics.get('memory_peak', 0),
            cpu_avg=resource_metrics.get('cpu_avg', 0),
            batch_processing_times=batch_times
        )

    def _detect_hot_nodes(self, relationships: List[Dict]) -> Tuple[Set[str], Dict[str, int]]:
        """
        Phase 1: Detect hot nodes using degree centrality

        Hot nodes are those with high degree (many connections) that are likely
        to cause contention when multiple threads try to access them simultaneously.
        """
        # Calculate node degrees (how many relationships each node participates in)
        node_degrees = defaultdict(int)

        for rel in relationships:
            node_degrees[rel['from']] += 1
            node_degrees[rel['to']] += 1

        if not node_degrees:
            return set(), {}

        # Calculate threshold for hot nodes (e.g., top 20% by degree)
        degrees = list(node_degrees.values())
        degrees.sort(reverse=True)

        threshold_index = int(len(degrees) * (1 - self.hot_node_threshold))
        if threshold_index >= len(degrees):
            threshold_index = len(degrees) - 1

        degree_threshold = degrees[threshold_index]

        # Identify hot nodes
        hot_nodes = {
            node for node, degree in node_degrees.items()
            if degree >= degree_threshold and degree > 1  # Must have at least 2 connections
        }

        return hot_nodes, dict(node_degrees)

    def _separate_hot_relationships(self, relationships: List[Dict], hot_nodes: Set[str]) -> Tuple[List[Dict], List[Dict]]:
        """
        Separate relationships into hot (involving hot nodes) and normal categories
        """
        hot_relationships = []
        normal_relationships = []

        for rel in relationships:
            if rel['from'] in hot_nodes or rel['to'] in hot_nodes:
                hot_relationships.append(rel)
            else:
                normal_relationships.append(rel)

        return hot_relationships, normal_relationships

    def _apply_graph_coloring(self, relationships: List[Dict]) -> List[List[Dict]]:
        """
        Phase 2: Apply graph coloring to ensure conflicting relationships are in different batches

        Two relationships conflict if they share a common node (from or to).
        Graph coloring ensures that no two conflicting relationships get the same color.
        """
        if not relationships:
            return []

        # Build conflict graph: relationships that share nodes
        relationship_conflicts = defaultdict(set)
        node_to_relationships = defaultdict(list)

        # Map nodes to relationships
        for i, rel in enumerate(relationships):
            node_to_relationships[rel['from']].append(i)
            node_to_relationships[rel['to']].append(i)

        # Build conflict edges
        for node, rel_indices in node_to_relationships.items():
            for i in range(len(rel_indices)):
                for j in range(i + 1, len(rel_indices)):
                    rel_i, rel_j = rel_indices[i], rel_indices[j]
                    relationship_conflicts[rel_i].add(rel_j)
                    relationship_conflicts[rel_j].add(rel_i)

        # Apply greedy graph coloring
        colors = {}
        max_color = 0

        for rel_idx in range(len(relationships)):
            # Find available colors (not used by conflicting relationships)
            used_colors = {colors.get(conflict_idx) for conflict_idx in relationship_conflicts[rel_idx]}
            used_colors.discard(None)  # Remove None values

            # Assign the smallest available color
            color = 0
            while color in used_colors:
                color += 1

            colors[rel_idx] = color
            max_color = max(max_color, color)

        # Ensure we have at least min_colors for better parallelism
        num_colors = max(max_color + 1, self.min_colors)

        # Group relationships by color into batches
        color_groups = defaultdict(list)
        for rel_idx, color in colors.items():
            # Redistribute colors to ensure we use all available colors
            redistributed_color = color % num_colors
            color_groups[redistributed_color].append(relationships[rel_idx])

        # Convert to list of batches, respecting batch size
        colored_batches = []
        for color in range(num_colors):
            color_relationships = color_groups[color]

            # Split large color groups into multiple batches
            for i in range(0, len(color_relationships), self.batch_size):
                batch = color_relationships[i:i + self.batch_size]
                if batch:  # Only add non-empty batches
                    colored_batches.append(batch)

        return colored_batches

    def _create_hot_node_batches(self, hot_relationships: List[Dict]) -> List[List[Dict]]:
        """
        Create smaller batches for hot relationships to reduce contention
        Hot relationships need more careful handling
        """
        if not hot_relationships:
            return []

        # Use smaller batch size for hot relationships
        hot_batch_size = max(1, self.batch_size // 4)  # Quarter of normal batch size

        batches = []
        for i in range(0, len(hot_relationships), hot_batch_size):
            batch = hot_relationships[i:i + hot_batch_size]
            batches.append(batch)

        return batches

    def _execute_hybrid_processing(self, hot_batches: List[List[Dict]], colored_batches: List[List[Dict]]) -> List[Dict]:
        """
        Phase 3: Execute hybrid processing strategy

        Strategy:
        1. Process hot node batches with limited parallelism (serial or few threads)
        2. Process colored batches with full parallelism
        """
        results = []

        # Step 1: Process hot batches with limited parallelism (e.g., 1-2 threads)
        if hot_batches:
            print(f"  Processing {len(hot_batches)} hot batches with limited parallelism...")
            hot_thread_count = min(2, self.thread_count)  # Limit to 2 threads for hot nodes

            with concurrent.futures.ThreadPoolExecutor(max_workers=hot_thread_count) as executor:
                future_to_batch = {
                    executor.submit(self._process_single_batch, f"hot_{i}", batch): i
                    for i, batch in enumerate(hot_batches)
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        result = future.result(timeout=60)
                        results.append(result)
                    except Exception as e:
                        results.append({'successful': 0, 'conflicts': 0, 'retries': 0, 'time': 0})

        # Step 2: Process colored batches with full parallelism
        if colored_batches:
            print(f"  Processing {len(colored_batches)} colored batches with full parallelism...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_count) as executor:
                future_to_batch = {
                    executor.submit(self._process_single_batch, f"color_{i}", batch): i
                    for i, batch in enumerate(colored_batches)
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        result = future.result(timeout=60)
                        results.append(result)
                    except Exception as e:
                        results.append({'successful': 0, 'conflicts': 0, 'retries': 0, 'time': 0})

        return results

    def _process_single_batch(self, batch_id: str, batch: List[Dict]) -> Dict:
        """
        Process a single batch with retry logic
        """
        batch_start = time.time()
        conflicts, retries, successful = self._insert_batch_with_retry(batch, batch_id)
        processing_time = time.time() - batch_start

        print(f"    Batch {batch_id}: {successful}/{len(batch)} successful, {conflicts} conflicts, {retries} retries")

        return {
            'successful': successful,
            'conflicts': conflicts,
            'retries': retries,
            'time': processing_time
        }

    def _insert_batch_with_retry(self, batch: List[Dict], batch_id: str) -> tuple:
        """
        Insert a batch with retry logic, counting all conflicts
        """
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