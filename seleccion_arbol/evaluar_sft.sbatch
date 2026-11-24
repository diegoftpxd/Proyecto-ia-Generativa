#!/bin/bash
# Evaluación de Ministral 3 3B Reasoning en datos/test.jsonl, con el presupuesto del baseline. Desde seleccion_arbol:
#   sbatch evaluar_sft.sbatch --nombre "Ministral 3 3B SFT solver" --adaptador resultados/06_sft_ministral_solver/final
#   sbatch evaluar_sft.sbatch --nombre "Ministral 3 3B Reasoning bf16"      (modelo base, sin adaptador)
# Los resultados quedan en BaseLines/<nombre>/. Si el trabajo se corta, reenviarlo con el mismo --nombre retoma.
#SBATCH --job-name=eval-ministral
#SBATCH --comment="Proyecto IA Generativa: evaluación de Ministral 3 3B Reasoning (con o sin LoRA) en datos/test.jsonl"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --partition=peteroa-default
#SBATCH --qos=regular
#SBATCH --gres=gpu:h100:1                # H100 HGX de 80 GB
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=1-00:00:00                # el máximo de regular; si se corta, reenviar y retoma

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

uv run evaluar_sft.py "$@"
