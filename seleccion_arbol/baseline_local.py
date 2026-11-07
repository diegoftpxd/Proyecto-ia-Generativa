"""Línea base en test con la GPU de tu PC, con la misma configuración que colab_evaluacion.ipynb.

Escribe resultados/<nombre>/baseline_test_local.jsonl y configuracion.json. Si se corta, repite el
mismo comando y continúa donde quedó. Ejecútalo desde seleccion_arbol:

    python baseline_local.py "OLMo 2 1B" allenai/OLMo-2-0425-1B-Instruct
    python baseline_local.py "Ministral 3 3B Reasoning 4bit" mistralai/Ministral-3-3B-Reasoning-2512 --4bit
"""
import argparse
import datetime
import json
import subprocess
import threading
from pathlib import Path

from evaluacion import evaluar_dataset, leer_resultados, resumir, resumir_pass
from modelo_hf import ModeloTransformers

CARPETA = Path(__file__).parent
DATOS = CARPETA / "datos" / "test.jsonl"
NIVELES = [2, 3, 4, 5, 6]
INTENTOS = 3
PRESUPUESTO = dict(tokens_pensamiento=3400, tokens_respuesta=200)


class ModeloTransformers4bit(ModeloTransformers):
    # Para modelos que no caben en bf16 en la GPU, como Ministral 3 3B (unos 8 GB) en una de 6 GB.
    # Solo cambia la carga. La parte de visión y la capa de salida no se cuantizan.
    def _cargar(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

        opciones = dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"],
            ),
            dtype=torch.bfloat16,
            device_map={"": 0},  # Todo en la GPU: si no cabe, falla en vez de pasar capas a la CPU
        )
        self._tokenizador = AutoTokenizer.from_pretrained(self.nombre)
        try:
            self._modelo = AutoModelForImageTextToText.from_pretrained(self.nombre, **opciones)
        except ValueError:
            self._modelo = AutoModelForCausalLM.from_pretrained(self.nombre, **opciones)


def crear_modelo(nombre_hf: str, en_4bit: bool) -> ModeloTransformers:
    clase = ModeloTransformers4bit if en_4bit else ModeloTransformers
    return clase(nombre_hf, razonar=True, **PRESUPUESTO)


def estado_gpu() -> str:
    # P0 cerca de 100 W es normal en la RTX 3060 Laptop. P5 a 34 W indica el límite de energía del perfil de Razer Synapse.
    consulta = "pstate,utilization.gpu,power.draw,enforced.power.limit,clocks.mem,temperature.gpu,memory.used"
    resultado = subprocess.run(
        ["nvidia-smi", f"--query-gpu={consulta}", "--format=csv,noheader"], capture_output=True, text=True
    )
    return resultado.stdout.strip()


def guardar_configuracion(ruta: Path, nombre: str, nombre_hf: str, en_4bit: bool) -> None:
    import torch
    import transformers

    versiones = {"transformers": transformers.__version__, "torch": torch.__version__}
    precision = "bf16"
    if en_4bit:
        import bitsandbytes

        versiones["bitsandbytes"] = bitsandbytes.__version__
        precision = "4 bits NF4 con bitsandbytes (cálculo en bf16; visión y lm_head sin cuantizar)"
    episodios = leer_resultados(DATOS)
    configuracion = {
        "modelo": f"{nombre} ({nombre_hf}) con Hugging Face transformers, en local",
        "hardware": f"{torch.cuda.get_device_name(0)}, {precision}",
        "versiones": versiones,
        "tokens_pensamiento": PRESUPUESTO["tokens_pensamiento"],
        "tokens_respuesta": PRESUPUESTO["tokens_respuesta"],
        "temperatura": 0.7,
        "razonar": True,
        "niveles": NIVELES,
        "arboles_por_nivel": sum(e["nivel"] == NIVELES[0] for e in episodios),
        "intentos": INTENTOS,
        "episodios": sum(e["nivel"] in NIVELES for e in episodios) * INTENTOS,
        "prompt": "El de prompt.py (inglés), igual que 02_prompt_ingles",
        "tokenizacion": "Prompt tokenizado dentro de apply_chat_template (sin token de inicio duplicado)",
        "fecha": datetime.date.today().isoformat(),
    }
    ruta.write_text(json.dumps(configuracion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nombre", help="Nombre del modelo en los resultados y de su carpeta")
    parser.add_argument("nombre_hf", help="Modelo en Hugging Face, por ejemplo unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--4bit", dest="en_4bit", action="store_true", help="Cargar en 4 bits (bitsandbytes)")
    argumentos = parser.parse_args()

    carpeta = CARPETA / "resultados" / argumentos.nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "baseline_test_local.jsonl"
    if not (carpeta / "configuracion.json").exists():
        guardar_configuracion(carpeta / "configuracion.json", argumentos.nombre, argumentos.nombre_hf, argumentos.en_4bit)
    total = json.loads((carpeta / "configuracion.json").read_text(encoding="utf-8"))["episodios"]

    parar = threading.Event()

    def registrar_gpu():
        # Cada 5 minutos, para que un límite de energía que vuelva quede en el registro.
        while not parar.wait(300):
            hechos = len(leer_resultados(ruta)) if ruta.exists() else 0
            print(f"[{datetime.datetime.now():%H:%M}] GPU {estado_gpu()} | episodios guardados: {hechos}/{total}", flush=True)

    print(f"{argumentos.nombre} ({argumentos.nombre_hf}) -> {ruta}\nGPU {estado_gpu()}", flush=True)
    threading.Thread(target=registrar_gpu, daemon=True).start()
    modelo = crear_modelo(argumentos.nombre_hf, argumentos.en_4bit)
    try:
        evaluar_dataset(modelo, argumentos.nombre, DATOS, ruta, niveles=NIVELES, intentos=INTENTOS)
    finally:
        parar.set()
        modelo.liberar()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== pass@k por nivel ===")
    print(resumir_pass(ruta, k=INTENTOS))
    print("\n=== resumen por nivel ===")
    print(resumir(ruta))
    print("\n=== total ===")
    print(resumir_pass(ruta, k=INTENTOS, por=["modelo"]))
    print(resumir(ruta, por=["modelo"]))


if __name__ == "__main__":
    main()
