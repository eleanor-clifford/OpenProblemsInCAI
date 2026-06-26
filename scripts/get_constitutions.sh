#!/bin/sh

find exp/saved_outputs -path '*/results/train.json' | grep -v '_unfair' | while read -r p; do
    save_path="$(echo "$p" | sed 's|exp/saved_outputs|exp/constitutions|;s|/results/train.json|.txt|')"
    constitution="$(python3 scripts/parse_constitution.py "$p")"
    test -z "$constitution" && continue
    mkdir -p "$(dirname "$save_path")"
    printf '%s\n' "$constitution" > "$save_path"
    echo "$save_path"
done
