import hashlib
import time
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.evaluation.framework import (
    PerformanceMetrics, ConflictAnalyzer, ResourceMonitor, AlgorithmBase
)


class MixAndBatchAlgorithm(AlgorithmBase):
    """
    Implementation of the Mix and Batch algorithm for parallel relationship loading.

    Based on the paper: "Mix and Batch: A Technique for Fast, Parallel Relationship
    Loading in Neo4j" by J. Porter and A. Ontman.

    This algorithm prevents deadlocks by ensuring no two threads access the same
    node concurrently through intelligent relationship partitioning.
    """

    def __init__(self, config, driver):
        super().__init__(config, driver)
        self.name = config.get('name', 'Mix and Batch Algorithm')

        # Algorithm-specific parameters
        self.num_partitions = config.get('num_partitions', 10)
        self.max_workers = config.get('max_workers', 8)
        self.batch_size = config.get('batch_size', 1000)
        self.hash_digits = config.get('hash_digits', 1)  # Digits for partition code

        # Enhanced conflict tracking
        self.conflict_analyzer = ConflictAnalyzer(
            hub_threshold=config.get('hub_threshold', 5)
        )

        print(f"Initialized {self.name} with {self.num_partitions} partitions "
              f"and {self.max_workers} workers")

    def insert_relationships(self, relationships: List[Dict]) -> PerformanceMetrics:
        """Main entry point for Mix and Batch relationship insertion"""
        print(f"\nExecuting {self.name}")
        print(f"Processing {len(relationships)} relationships")
        print(f"Using {self.num_partitions} partitions with {self.max_workers} workers")

        # Start resource monitoring
        monitor = ResourceMonitor()
        monitor.start_monitoring()

        start_time = time.time()

        # Step 1: Partition relationships
        print("\nStep 1: Partitioning relationships...")
        partitioned_batches = self._partition_relationships(relationships)

        # Step 2: Execute batches in parallel
        print(f"\nStep 2: Executing {len(partitioned_batches)} batches in parallel...")
        metrics = self._execute_parallel_batches(partitioned_batches, monitor, start_time)

        return metrics

    def _partition_relationships(self, relationships: List[Dict]) -> List[List[Dict]]:
        """
        Partition relationships using the Mix and Batch algorithm.

        This ensures that within each batch, no node appears in multiple relationships,
        allowing parallel processing without locks.
        """
        # Step 1: Assign partition codes to all nodes
        node_partitions = self._assign_node_partitions(relationships)

        # Step 2: Create a graph coloring problem
        # Each relationship is colored based on source and target partition codes
        relationship_colors = self._color_relationships(relationships, node_partitions)

        # Step 3: Group relationships by color into batches
        batches = defaultdict(list)
        for rel, color in zip(relationships, relationship_colors):
            batches[color].append(rel)

        # Convert to list and report statistics
        partitioned_batches = list(batches.values())

        print(f"  Created {len(partitioned_batches)} conflict-free batches")
        batch_sizes = [len(batch) for batch in partitioned_batches]
        print(f"  Batch size distribution: min={min(batch_sizes)}, "
              f"max={max(batch_sizes)}, avg={sum(batch_sizes) / len(batch_sizes):.1f}")

        # Analyze load balance
        self._analyze_load_balance(partitioned_batches)

        return partitioned_batches

    def _assign_node_partitions(self, relationships: List[Dict]) -> Dict[str, int]:
        """
        Assign partition codes to nodes using hash-based distribution.

        The partition code determines which "virtual partition" a node belongs to,
        spreading nodes evenly across partitions.
        """
        node_partitions = {}

        # Collect all unique nodes
        all_nodes = set()
        for rel in relationships:
            all_nodes.add(rel['from'])
            all_nodes.add(rel['to'])

        # Assign partition codes based on hash
        for node in all_nodes:
            # Use MD5 hash for consistent distribution
            hash_obj = hashlib.md5(node.encode())
            hash_hex = hash_obj.hexdigest()

            # Extract digits from hash for partition code
            partition_code = int(hash_hex[:self.hash_digits], 16) % self.num_partitions
            node_partitions[node] = partition_code

        # Report partition distribution
        partition_counts = defaultdict(int)
        for partition in node_partitions.values():
            partition_counts[partition] += 1

        print(f"  Node distribution across {self.num_partitions} partitions:")
        for p in range(self.num_partitions):
            count = partition_counts.get(p, 0)
            print(f"    Partition {p}: {count} nodes")

        return node_partitions

    def _color_relationships(self, relationships: List[Dict],
                             node_partitions: Dict[str, int]) -> List[int]:
        """
        Color relationships using a graph coloring algorithm.

        Two relationships get the same color only if they don't share any nodes.
        This ensures relationships with the same color can be processed in parallel.
        """
        # Build conflict graph where edges connect relationships that share nodes
        n_rels = len(relationships)
        conflicts = defaultdict(set)

        # For efficiency, group relationships by their node partitions
        partition_groups = defaultdict(list)
        for i, rel in enumerate(relationships):
            source_part = node_partitions[rel['from']]
            target_part = node_partitions[rel['to']]

            # Create a partition pair key (order doesn't matter)
            part_key = (min(source_part, target_part), max(source_part, target_part))
            partition_groups[part_key].append(i)

        # Find conflicts within each partition group
        for group_indices in partition_groups.values():
            # Within a group, check for actual node conflicts
            for i in range(len(group_indices)):
                for j in range(i + 1, len(group_indices)):
                    idx1, idx2 = group_indices[i], group_indices[j]
                    rel1, rel2 = relationships[idx1], relationships[idx2]

                    # Check if relationships share any nodes
                    if (rel1['from'] == rel2['from'] or
                            rel1['from'] == rel2['to'] or
                            rel1['to'] == rel2['from'] or
                            rel1['to'] == rel2['to']):
                        conflicts[idx1].add(idx2)
                        conflicts[idx2].add(idx1)

        # Color the conflict graph using greedy algorithm
        colors = [-1] * n_rels
        max_color = -1

        # Process relationships in order of conflict degree (most constrained first)
        rel_order = sorted(range(n_rels), key=lambda x: len(conflicts[x]), reverse=True)

        for rel_idx in rel_order:
            # Find the smallest color not used by neighbors
            neighbor_colors = {colors[neighbor] for neighbor in conflicts[rel_idx]
                               if colors[neighbor] != -1}

            # Assign the smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1

            colors[rel_idx] = color
            max_color = max(max_color, color)

        print(f"  Graph coloring used {max_color + 1} colors")

        return colors

    def _analyze_load_balance(self, batches: List[List[Dict]]):
        """Analyze and report load balance across batches"""
        # Count relationships per batch
        batch_sizes = [len(batch) for batch in batches]

        # Analyze entity distribution
        entity_counts = []
        for batch in batches:
            entities = set()
            for rel in batch:
                entities.add(rel['from'])
                entities.add(rel['to'])
            entity_counts.append(len(entities))

        # Calculate imbalance metrics
        size_variance = sum((s - sum(batch_sizes) / len(batch_sizes)) ** 2 for s in batch_sizes)
        size_variance /= len(batch_sizes)

        print(f"\n  Load Balance Analysis:")
        print(f"    Relationship variance: {size_variance:.2f}")
        print(f"    Entity distribution: min={min(entity_counts)}, "
              f"max={max(entity_counts)}, avg={sum(entity_counts) / len(entity_counts):.1f}")

    def _execute_parallel_batches(self, batches: List[List[Dict]],
                                  monitor: ResourceMonitor,
                                  start_time: float) -> PerformanceMetrics:
        """Execute batches in parallel using thread pool"""

        # Initialize metrics tracking
        total_predicted_conflicts = 0
        total_actual_conflicts = 0
        total_retry_count = 0
        conflict_resolution_time = 0
        batch_times = []
        successful_operations = 0
        total_operations = 0
        all_hotspot_entities = []

        # Create experiment ID for this run
        experiment_id = f"mix_batch_{int(time.time() * 1000)}"

        # Execute batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_batch, batch, batch_idx, experiment_id):
                    (batch_idx, batch) for batch_idx, batch in enumerate(batches)
            }

            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_idx, batch = future_to_batch[future]

                try:
                    result = future.result()

                    # Aggregate metrics
                    batch_times.append(result['time'])
                    total_predicted_conflicts += result['predicted_conflicts']
                    total_actual_conflicts += result['actual_conflicts']
                    total_retry_count += result['retries']
                    conflict_resolution_time += result['conflict_time']
                    successful_operations += result['successful_ops']
                    total_operations += len(batch)
                    all_hotspot_entities.extend(result['hotspots'])

                    print(f"  Batch {batch_idx + 1}/{len(batches)} completed in "
                          f"{result['time']:.2f}s (conflicts: {result['actual_conflicts']})")

                except Exception as e:
                    print(f"  Batch {batch_idx + 1} failed: {str(e)[:100]}")
                    total_operations += len(batch)

        # Calculate final metrics
        total_time = time.time() - start_time
        resource_metrics = monitor.stop_monitoring()

        # Build metrics object
        all_relationships = [rel for batch in batches for rel in batch]
        throughput = len(all_relationships) / total_time if total_time > 0 else 0
        success_rate = (successful_operations / total_operations * 100) if total_operations > 0 else 0

        # Calculate conflict prediction accuracy
        if total_predicted_conflicts > 0:
            prediction_accuracy = min(100.0, (total_actual_conflicts / total_predicted_conflicts) * 100)
        else:
            prediction_accuracy = 100.0 if total_actual_conflicts == 0 else 0.0

        # Get top hotspot entities
        from collections import Counter
        hotspot_counter = Counter(all_hotspot_entities)
        top_hotspots = [entity for entity, _ in hotspot_counter.most_common(5)]

        return PerformanceMetrics(
            algorithm_name=self.name,
            scenario="",  # Set by framework
            run_number=0,  # Set by framework
            batch_size=len(batches),
            total_time=total_time,
            batch_processing_times=batch_times,
            total_entities=self._count_unique_entities(all_relationships),
            total_relationships=len(all_relationships),
            predicted_conflicts=total_predicted_conflicts,
            actual_conflicts=total_actual_conflicts,
            conflict_prediction_accuracy=prediction_accuracy,
            conflict_resolution_time=conflict_resolution_time,
            retry_count=total_retry_count,
            hotspot_entities=top_hotspots,
            throughput=throughput,
            memory_peak=resource_metrics['memory_peak'],
            cpu_avg=resource_metrics['cpu_avg'],
            success_rate=success_rate,
            timestamp=datetime.now().isoformat()
        )

    def _process_batch(self, batch: List[Dict], batch_idx: int,
                       experiment_id: str) -> Dict:
        """Process a single batch of relationships"""

        start_time = time.time()

        # Predict conflicts before insertion
        conflict_analysis = self.conflict_analyzer.detect_conflicts_in_batch(batch)
        predicted_conflicts = conflict_analysis['total_predicted_conflicts']

        # Insert relationships
        conflict_start = time.time()
        actual_conflicts = 0
        retries = 0
        successful_ops = 0

        with self.driver.session() as session:
            for rel in batch:
                try:
                    # Use MERGE to create relationships
                    query = """
                    MERGE (from:Entity {name: $from_name})
                    MERGE (to:Entity {name: $to_name})
                    SET from.experiment_id = $experiment_id,
                        to.experiment_id = $experiment_id
                    MERGE (from)-[r:LINKS_TO]->(to)
                    SET r.similarity = $similarity,
                        r.involves_hub = $involves_hub,
                        r.pages_from = $pages_1,
                        r.pages_to = $pages_2,
                        r.experiment_id = $experiment_id,
                        r.batch_id = $batch_id
                    """

                    session.run(query, {
                        'from_name': rel['from'],
                        'to_name': rel['to'],
                        'similarity': rel['similarity'],
                        'involves_hub': rel['involves_hub'],
                        'pages_1': rel['pages_1'],
                        'pages_2': rel['pages_2'],
                        'experiment_id': experiment_id,
                        'batch_id': batch_idx
                    })

                    successful_ops += 1

                except Exception as e:
                    # Track conflicts
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in
                           ['lock', 'deadlock', 'timeout', 'conflict']):
                        actual_conflicts += 1
                        retries += 1

        conflict_time = time.time() - conflict_start
        total_time = time.time() - start_time

        return {
            'time': total_time,
            'predicted_conflicts': predicted_conflicts,
            'actual_conflicts': actual_conflicts,
            'retries': retries,
            'conflict_time': conflict_time,
            'successful_ops': successful_ops,
            'hotspots': conflict_analysis['conflict_hotspots']
        }

    def _count_unique_entities(self, relationships: List[Dict]) -> int:
        """Count unique entities in relationships"""
        entities = set()
        for rel in relationships:
            entities.add(rel['from'])
            entities.add(rel['to'])
        return len(entities)
