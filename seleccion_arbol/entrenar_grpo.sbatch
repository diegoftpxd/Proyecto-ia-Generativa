#!/bin/bash
# GRPO de Ministral 3 3B Reasoning sobre el SFT de solver masivo, en una GPU de Slurm. Enviar desde la carpeta seleccion_arbol:
#   mkdir -p slurm/logs                     (una vez: Slurm no crea la carpeta de los logs)
#   sbatch entrenar_grpo.sbatch
#   sbatch --qos=debug --time=0:30:00 entrenar_grpo.sbatch --salida resultados/grpo_prueba --pasos 2 --problemas 2
#   Desde el modelo base, sin SFT: razona mucho más largo, así que necesita más tokens y genera en tandas.
#   sbatch entrenar_grpo.sbatch --sin-sft --salida resultados/09_grpo_ministral_base --niveles 2 3 4 5 6 \
#       --tokens 3400 --lote-generacion 64 --micro-lote 2
# Los argumentos van a entrenar_grpo.py. Si el trabajo se corta, volver a enviarlo con la misma --salida retoma
# desde el último checkpoint. Las recompensas de cada paso y nivel quedan en <salida>/metricas.csv y recompensas.csv.
#SBATCH --job-name=grpo-ministral
#SBATCH --comment="Proyecto IA Generativa: GRPO con LoRA de Ministral 3 3B Reasoning sobre el SFT, con la recompensa de src/entorno.py"
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

uv run entrenar_grpo.py "$@"
