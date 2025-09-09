import json

with open("data/output/new/relationships.json") as json_file:
   data = json.load(json_file)

relationships = data["LINKS_TO"]

hotspots = {}
for rel in relationships:
   hotspot = rel["from"]
   if hotspot == "BBC News":
       if hotspot not in hotspots:
           hotspots[hotspot] = set()
       hotspots[hotspot].add(rel["to"])

print(f"Found {len(hotspots)} hotspots (BBC News only)")
output_path = "data/output/s1.json"

with open(output_path, "w") as outfile:
   outfile.write('{\n  "LINKS_TO": [\n')

   total_relationships = 0
   first_relationship = True

   for hotspot_idx, (hotspot, entities) in enumerate(hotspots.items()):
       entities_list = list(entities)[:200]
       entity_count = len(entities_list)
       relationships_in_hotspot = entity_count * (entity_count - 1)

       print(
           f"Processing hotspot {hotspot_idx + 1}/{len(hotspots)}: '{hotspot}' ({entity_count} entities, {relationships_in_hotspot} relationships)")

       for i, from_entity in enumerate(entities_list):
           for j, to_entity in enumerate(entities_list):
               if from_entity == to_entity:
                   continue
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

               if total_relationships % 10000 == 0:
                   print(f"  Written {total_relationships} relationships...")

   outfile.write("\n  ]\n}")
   print(f"Done. Created {len(hotspots)} hotspots with {total_relationships} total interconnected relationships.")

print("\nHotspot Statistics:")
for hotspot, entities in hotspots.items():
   entity_count = len(entities)
   relationships_in_hotspot = entity_count * (entity_count - 1)
   print(f"'{hotspot}': {entity_count} entities, {relationships_in_hotspot} relationships")