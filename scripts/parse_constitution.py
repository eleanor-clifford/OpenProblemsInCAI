import json
import sys

obj = json.load(open(sys.argv[1]))

try:
    constitution = obj["metadata"]["const_annotator_details"]["constitution"]
except (KeyError, TypeError):
    constitution = ""

if isinstance(constitution, list):
    constitution = "\n".join(f"{i+1}. {p}" for i, p in enumerate(constitution))

print(constitution.strip("\n"))
