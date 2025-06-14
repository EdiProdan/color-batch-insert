import json

with open("data/output/new/relationships.json") as json_file:
    data = json.load(json_file)


def duplicate_entities(relationships_data, copies_per_entity=10):
    """Duplicates ONLY 'from' nodes to increase contention on them."""
    duplicated_relationships = []
    for rel in relationships_data["LINKS_TO"]:
        original_from = rel["from"]
        for i in range(copies_per_entity):
            duplicated_relationships.append({
                "from": f"{original_from}",  # Duplicate "from" only
                "to": rel["to"]                   # Keep "to" intact
            })
    return {"LINKS_TO": duplicated_relationships}


# --- APPLY DUPLICATION ---
duplicated_data = duplicate_entities(data, copies_per_entity=10)

with open('data/output/new/relationships.json', 'w') as f:
    json.dump(data, f, indent=4)

print(f"Duplication complete! Original entities: {len(data['LINKS_TO'])}")
print(f"New relationships: {len(duplicated_data['LINKS_TO'])}")