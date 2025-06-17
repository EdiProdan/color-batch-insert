import json

# Step 1: Load relationships and extract hotspots
with open("data/output/new/relationships.json") as json_file:
    data = json.load(json_file)

relationships = data["LINKS_TO"]

# Step 2: Group entities by hotspot (from entity) - GET TOP 15 HOTSPOTS
hotspots = {}
for rel in relationships:
    hotspot = rel["from"]
    if hotspot not in hotspots:
        hotspots[hotspot] = set()
    hotspots[hotspot].add(rel["to"])

# Sort hotspots by number of entities and take top 15
sorted_hotspots = sorted(hotspots.items(), key=lambda x: len(x[1]), reverse=True)[:40]

print(f"Selected top {len(sorted_hotspots)} hotspots with most entities")
for i, (hotspot, entities) in enumerate(sorted_hotspots):
    print(f"  {i + 1}. '{hotspot}': {len(entities)} entities")

# Step 3: Generate fully connected relationships within each hotspot
output_path = "data/output/new/relationships_15_hotspots_circular.json"

with open(output_path, "w") as outfile:
    outfile.write('{\n  "LINKS_TO": [\n')

    total_relationships = 0
    first_relationship = True

    # Process each hotspot one at a time
    for hotspot_idx, (hotspot, entities) in enumerate(sorted_hotspots):
        # Truncate to exactly 20 entities per hotspot
        entities_list = list(entities)[:40]
        entity_count = len(entities_list)
        relationships_in_hotspot = entity_count * (entity_count - 1)

        print(
            f"Processing hotspot {hotspot_idx + 1}/{len(sorted_hotspots)}: '{hotspot}' ({entity_count} entities, {relationships_in_hotspot} relationships)")

        # Connect every entity to every other entity within the hotspot
        for i, from_entity in enumerate(entities_list):
            for j, to_entity in enumerate(entities_list):
                if from_entity == to_entity:
                    continue  # skip self-links

                # Create relationship entry
                entry = {
                    "from": from_entity,
                    "to": to_entity
                }

                # Write to file immediately
                if not first_relationship:
                    outfile.write(",\n")
                else:
                    first_relationship = False

                json_str = json.dumps(entry)
                outfile.write(f"    {json_str}")
                total_relationships += 1

                # Progress indicator every 1000 relationships
                if total_relationships % 1000 == 0:
                    print(f"  Written {total_relationships} relationships...")

    outfile.write("\n  ]\n}")
    print(
        f"Done. Created {len(sorted_hotspots)} hotspots with {total_relationships} total interconnected relationships.")

# Print final statistics
print("\nFinal Hotspot Statistics:")
for i, (hotspot, entities) in enumerate(sorted_hotspots):
    entity_count = min(len(entities), 40)  # Truncated to 20
    relationships_in_hotspot = entity_count * (entity_count - 1)
    print(f"  {i + 1}. '{hotspot}': {entity_count} entities, {relationships_in_hotspot} relationships")

print(f"\nTotal: {len(sorted_hotspots)} hotspots, {total_relationships} relationships")
print(f"Expected conflicts with naive parallel: {total_relationships // 4} - {total_relationships // 2}")
print(f"Expected conflicts with semantic-aware: 0-10")