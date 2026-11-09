"""Build original fictional fixtures. No model output is used to create labels."""

import hashlib
import json
from pathlib import Path


def build(names, split):
    tasks = []
    for index, (name, old, new) in enumerate(names):
        category = ("late", "contradiction", "buried_correction")[index % 3]
        question = f"What is the current {name} access code?"
        docs = []
        if category != "late":
            docs.append({"id": "old", "order": 0,
                         "text": f"The current {name} access code is {old}."})
        long = index % 2 == 1
        for j in range(8 if long else 2):
            docs.append({"id": f"note{j}", "order": j + 1,
                         "text": f"Archive note {j}: The training room contains chairs and desks. "
                         "Visitors should keep the corridors clear. Equipment checks happen weekly. "
                         "The office handbook lists cleaning procedures and contact details."})
        fact = f"The current {name} access code is {new}."
        text = fact
        if category == "buried_correction":
            text = "This record contains a routine administrative update. " + fact
        docs.append({"id": "latest", "order": 100, "text": text})
        tasks.append({"id": f"{split}-{index + 1:02}", "category": category,
                      "question": question, "documents": docs, "answers": [new],
                      "evidence": {"latest": fact}})
    return {"split": split, "tasks": tasks}


def main():
    heldout = [("Orchid", "1122", "7391"), ("Harbor", "2211", "8642"),
               ("Cedar", "3443", "5926"), ("Willow", "4554", "8173"),
               ("Quartz", "5665", "9362"), ("Maple", "6776", "2849"),
               ("Copper", "7887", "6518"), ("Falcon", "8998", "4729"),
               ("Birch", "9009", "1836"), ("Coral", "1010", "7258"),
               ("Aspen", "2020", "3496"), ("Juniper", "3030", "9581")]
    for split, names in [("dev", [("Demo", "1111", "1234"), ("Sample", "2222", "5678")]),
                         ("heldout", heldout)]:
        path = Path("data") / f"{split}.json"
        path.write_text(json.dumps(build(names, split), indent=2) + "\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path("data").glob("*.json")) if p.name != "manifest.json"}
    Path("data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
