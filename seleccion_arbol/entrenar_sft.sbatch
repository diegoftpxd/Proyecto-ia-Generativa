#!/bin/bash
# SFT de Ministral 3 3B Reasoning en Slurm. Enviar desde la carpeta seleccion_arbol:
#   mkdir -p slurm/logs                     (una vez: Slurm no crea la carpeta de los logs)
#   sbatch entrenar_sft.sbatch
#   sbatch entrenar_sft.sbatch --max-pasos 5 --salida resultados/sft_prueba   (los argumentos van a entrenar_sft.py)
# Si el trabajo se corta por tiempo, volver a enviarlo con la misma --salida retoma desde el último checkpoint.
#
# Los valores por defecto de entrenar_sft.py están pensados para una GPU de 80 GB; con una de 24 GB,
# agrega --lote 1 --acumulacion 16.
#SBATCH --job-name=sft-ministral
#SBATCH --comment="Proyecto IA Generativa: SFT con LoRA de Ministral 3 3B Reasoning sobre datos/sft.jsonl"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --partition=ialab-low-unlimit
#SBATCH --qos=debug
#SBATCH --nodelist=peteroa               # H100 HGX de 80 GB
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=1:00:00                   # ~84 pasos; si se corta, reenviar y retoma

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

if ! command -v uv >/dev/null 2>&1; then
    echo "No se encontró uv. Instálalo con: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# Los pesos (~8 GB) y los paquetes se guardan una sola vez. Cambia estas rutas si el home tiene poca cuota.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export PYTHONUNBUFFERED=1  # El log de Slurm muestra el avance al momento

echo "Nodo: $(hostname)  Trabajo: ${SLURM_JOB_ID:-local}"
nvidia-smi -L

uv run entrenar_sft.py "$@"
