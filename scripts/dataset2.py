import json

with open("data/output/new/relationships.json") as json_file:
    data = json.load(json_file)

relationships = data["LINKS_TO"]

hotspots = {}
for rel in relationships:
    hotspot = rel["from"]
    if hotspot not in hotspots:
        hotspots[hotspot] = set()
    hotspots[hotspot].add(rel["to"])

sorted_hotspots = sorted(hotspots.items(), key=lambda x: len(x[1]), reverse=True)[:40]

print(f"Selected top {len(sorted_hotspots)} hotspots with most entities")
for i, (hotspot, entities) in enumerate(sorted_hotspots):
    print(f"  {i + 1}. '{hotspot}': {len(entities)} entities")

output_path = "data/output/s2.json"

with open(output_path, "w") as outfile:
    outfile.write('{\n  "LINKS_TO": [\n')

    total_relationships = 0
    first_relationship = True
    for hotspot_idx, (hotspot, entities) in enumerate(sorted_hotspots):
        entities_list = list(entities)[:60]
        entity_count = len(entities_list)
        relationships_in_hotspot = entity_count * (entity_count - 1)

        print(
            f"Processing hotspot {hotspot_idx + 1}/{len(sorted_hotspots)}: '{hotspot}' ({entity_count} entities, {relationships_in_hotspot} relationships)")

        for i, from_entity in enumerate(entities_list):
            for j, to_entity in enumerate(entities_list):
                if from_entity == to_entity:
                    continue  # skip self-links
                entry = {
                    "from": from_entity,
                    "to": to_entity
                }

                if not first_relationship:
                    outfile.write(",\n")
                else:
                    first_relationship = False

                json_str = json.dumps(entry)
                outfile.write(f"    {json_str}")
                total_relationships += 1

                if total_relationships % 1000 == 0:
                    print(f"  Written {total_relationships} relationships...")

    outfile.write("\n  ]\n}")
    print(
        f"Done. Created {len(sorted_hotspots)} hotspots with {total_relationships} total interconnected relationships.")
