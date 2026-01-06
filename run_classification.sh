#!/bin/bash

# Argumentos:
#   $1 = type (default: classify)
#   $2 = max_workers (default: $MAX_WORKERS o 8)

type="${1:-classify}"
max_workers="${2:-${MAX_WORKERS:-8}}"

running=0

for file in data/classified/$type/*; do
    # solo procesar PDFs u otros archivos válidos
    if [[ -f "$file" ]]; then
        fname=$(basename "$file")
        folder=$(basename "$(dirname "$file")")
        out="output/$type/${fname}.json"
        mkdir -p "$(dirname "$out")"

        echo "Procesando $file"
        (
            docflow run "$type" "$file" \
                --output-format json \
                --output-path "$out" 
        ) &

        ((running++))
        if (( running >= max_workers )); then
            wait -n
            ((running--))
        fi
    fi
done

wait
