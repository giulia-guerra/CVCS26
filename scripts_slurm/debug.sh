#!/bin/bash

# Verifica se l'ambiente è attivo, altrimenti lo attiva
if [ -z "$VIRTUAL_ENV" ]; then
    source ~/envs/cvcs_env/bin/activate
fi

if [ $# -eq 0 ]; then
    echo "Uso: $0 <script.py> [argomenti...]"
    exit 1
fi

python -m debugpy --listen 5678 --wait-for-client "$@"