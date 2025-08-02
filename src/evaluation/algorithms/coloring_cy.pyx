from collections import defaultdict
from typing import List, Dict

def incremental_coloring(relationships) -> Dict[int, int]:

    entity_to_relationships = defaultdict(set)
    coloring = {}

    for i, rel in enumerate(relationships):
        # Find all relationships that conflict with current one
        conflicting_rels = set()
        for entity in [rel['from'], rel['to']]:
            conflicting_rels.update(entity_to_relationships[entity])

        # Find forbidden colors from conflicting relationships
        forbidden_colors = set()
        for conflict_rel in conflicting_rels:
            if conflict_rel in coloring:
                forbidden_colors.add(coloring[conflict_rel])

        # Assign first available color
        color = 0
        while color in forbidden_colors:
            color += 1

        coloring[i] = color

        # Update entity tracking
        entity_to_relationships[rel['from']].add(i)
        entity_to_relationships[rel['to']].add(i)

    return coloring
