import json

# Step 1: Load relationships and extract hotspots
with open("data/output/new/relationships.json") as json_file:
   data = json.load(json_file)

relationships = data["LINKS_TO"]

# Step 2: Group entities by hotspot (from entity) - FILTER FOR BBC ONLY
hotspots = {}
for rel in relationships:
   hotspot = rel["from"]
   # Only process BBC News hotspot
   if hotspot == "BBC News":
       if hotspot not in hotspots:
           hotspots[hotspot] = set()
       hotspots[hotspot].add(rel["to"])

print(f"Found {len(hotspots)} hotspots (BBC News only)")

# Step 3: Generate fully connected relationships within each hotspot
output_path = "data/output/new/relationships_bbc_200_connected.json"

with open(output_path, "w") as outfile:
   outfile.write('{\n  "LINKS_TO": [\n')

   total_relationships = 0
   first_relationship = True

   # Process each hotspot one at a time
   for hotspot_idx, (hotspot, entities) in enumerate(hotspots.items()):
       entities_list = list(entities)[:200]
       entity_count = len(entities_list)
       relationships_in_hotspot = entity_count * (entity_count - 1)

       print(
           f"Processing hotspot {hotspot_idx + 1}/{len(hotspots)}: '{hotspot}' ({entity_count} entities, {relationships_in_hotspot} relationships)")

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

               # Progress indicator every 10000 relationships
               if total_relationships % 10000 == 0:
                   print(f"  Written {total_relationships} relationships...")

   outfile.write("\n  ]\n}")
   print(f"Done. Created {len(hotspots)} hotspots with {total_relationships} total interconnected relationships.")

# Print final statistics
print("\nHotspot Statistics:")
for hotspot, entities in hotspots.items():
   entity_count = len(entities)
   relationships_in_hotspot = entity_count * (entity_count - 1)
   print(f"'{hotspot}': {entity_count} entities, {relationships_in_hotspot} relationships")