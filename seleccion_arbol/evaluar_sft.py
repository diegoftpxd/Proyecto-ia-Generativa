# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.6",
#     "transformers>=5.0",
#     "peft>=0.17",
#     "accelerate>=1.10",
#     "mistral-common>=1.8",
#     "pandas>=2.0",
# ]
# ///
# Evalúa Ministral 3 3B Reasoning, con o sin un adaptador de entrenar_sft.py, en datos/test.jsonl.
# Usa el presupuesto del baseline de Ministral, así que sus resultados se comparan con BaseLines/. Ejecutar desde esta carpeta:
#   uv run evaluar_sft.py --nombre "Ministral 3 3B SFT solver" --adaptador resultados/06_sft_ministral_solver/final
#   uv run evaluar_sft.py --nombre "Ministral 3 3B Reasoning bf16"        el modelo base, sin adaptador
# Cada intento se guarda al terminar: si el trabajo se corta, repetir el comando retoma donde quedó.
import argparse
import json
from datetime import date
from pathlib import Path

from src.evaluacion import evaluar_dataset, resumir_pass
from src.modelo_hf import ModeloTransformers

MODELO = "mistralai/Ministral-3-3B-Reasoning-2512"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa Ministral, con o sin adaptador LoRA, en el test.")
    parser.add_argument("--nombre", required=True, help="nombre del modelo en los resultados y de su carpeta en BaseLines")
    parser.add_argument("--adaptador", help="carpeta final/ de entrenar_sft.py; sin ella se evalúa el modelo base")
    parser.add_argument("--datos", default="datos/test.jsonl")
    parser.add_argument("--intentos", type=int, default=4, help="intentos por árbol, como el baseline")
    # El presupuesto y la temperatura del baseline de Ministral (BaseLines/Ministral 3 3B Reasoning 4bit).
    parser.add_argument("--tokens-pensamiento", type=int, default=3400)
    parser.add_argument("--tokens-respuesta", type=int, default=200)
    parser.add_argument("--temperatura", type=float, default=0.7)
    argumentos = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No hay GPU disponible: la evaluación necesita CUDA.")
    if argumentos.adaptador and not (Path(argumentos.adaptador) / "adapter_config.json").exists():
        raise SystemExit(f"No hay un adaptador en {argumentos.adaptador}: falta adapter_config.json.")

    carpeta = Path("BaseLines") / argumentos.nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "baseline_test.jsonl"
    with open(carpeta / "configuracion.json", "w", encoding="utf-8") as archivo:
        json.dump({
            "modelo": argumentos.nombre,
            "archivo": ruta.name,
            "modelo_base": MODELO,
            "adaptador": argumentos.adaptador,
            "ejecucion": "Transformers en bf16, con el adaptador LoRA fusionado" if argumentos.adaptador
                         else "Transformers en bf16",
            "hardware": torch.cuda.get_device_name(0),
            "datos": argumentos.datos,
            "intentos": argumentos.intentos,
            "tokens_pensamiento": argumentos.tokens_pensamiento,
            "tokens_respuesta": argumentos.tokens_respuesta,
            "temperatura": argumentos.temperatura,
            "razonar": True,
            "prompt": "El de prompt.py (inglés)",
            "fecha": date.today().isoformat(),
        }, archivo, ensure_ascii=False, indent=2)

    modelo = ModeloTransformers(
        MODELO,
        razonar=True,
        temperatura=argumentos.temperatura,
        tokens_pensamiento=argumentos.tokens_pensamiento,
        tokens_respuesta=argumentos.tokens_respuesta,
        adaptador=argumentos.adaptador,
    )
    fallas = evaluar_dataset(modelo, argumentos.nombre, argumentos.datos, ruta, intentos=argumentos.intentos)

    print(resumir_pass(ruta, k=argumentos.intentos).to_string())
    if fallas:
        raise SystemExit(f"Quedaron {fallas} intentos pendientes: repetir el comando para completarlos.")


if __name__ == "__main__":
    main()
