# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.6",
#     "transformers>=5.0",
#     "peft>=0.17",
#     "accelerate>=1.10",
#     "mistral-common>=1.8",
#     "matplotlib>=3.8",
# ]
# ///
# Entrenamiento SFT con LoRA de Ministral 3 3B Reasoning sobre datos/sft.jsonl. Ejecutar desde esta carpeta:
#   uv run entrenar_sft.py                                  todo con los valores por defecto
#   uv run entrenar_sft.py --salida resultados/sft_prueba --max-pasos 5
# Si en la carpeta de salida ya hay checkpoints, retoma desde el último (útil si Slurm corta el trabajo).
import argparse
import json
import random
from pathlib import Path

MODELO = "mistralai/Ministral-3-3B-Reasoning-2512"
# LoRA solo en el modelo de lenguaje: la torre de visión no se usa con prompts de texto.
MODULOS_LORA = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
IGNORAR = -100  # Etiqueta que la pérdida no cuenta: el prompt no se aprende, solo la respuesta


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT con LoRA de Ministral 3 3B Reasoning.")
    parser.add_argument("--modelo", default=MODELO, help="id en Hugging Face o carpeta local")
    parser.add_argument("--datos", default="datos/sft.jsonl")
    parser.add_argument("--salida", default="resultados/05_sft_ministral")
    parser.add_argument("--epocas", type=float, default=3)
    parser.add_argument("--max-pasos", type=int, default=-1, help="corta antes de las épocas; -1 no corta")
    parser.add_argument("--tasa", type=float, default=1e-4, help="learning rate")
    # Por defecto, pensado para una GPU de 80 GB: lote efectivo de 16 (4 × 4). Con 24 GB, usar --lote 1 --acumulacion 16.
    parser.add_argument("--lote", type=int, default=4, help="ejemplos por GPU en cada paso")
    parser.add_argument("--acumulacion", type=int, default=4, help="pasos que se acumulan antes de actualizar")
    parser.add_argument("--largo-maximo", type=int, default=6144, help="tokens; los ejemplos más largos se descartan")
    parser.add_argument("--fraccion-validacion", type=float, default=0.05)
    parser.add_argument("--rango-lora", type=int, default=64, help="alpha es el doble del rango")
    parser.add_argument("--pasos-guardado", type=int, default=20, help="cada cuántos pasos se guarda un checkpoint")
    parser.add_argument("--checkpoints-maximos", type=int, default=5, help="se borran los más antiguos")
    parser.add_argument("--pasos-validacion", type=int, default=10)
    parser.add_argument("--semilla", type=int, default=0)
    return parser.parse_args()


def tokenizar(tokenizador, ejemplo: dict, largo_maximo: int) -> dict | None:
    # El prompt se tokeniza con la plantilla de chat, igual que en ModeloTransformers._preparar, para que el
    # modelo entrenado vea en la evaluación exactamente el mismo comienzo que vio al entrenar.
    usuario, asistente = ejemplo["messages"]
    prompt = tokenizador.apply_chat_template(
        [usuario], tokenize=True, add_generation_prompt=True, return_dict=True
    )["input_ids"]
    respuesta = tokenizador(asistente["content"], add_special_tokens=False)["input_ids"]
    respuesta = respuesta + [tokenizador.eos_token_id]
    if len(prompt) + len(respuesta) > largo_maximo:
        return None  # Truncar cortaría el bloque <respuesta>, y el modelo aprendería a no cerrarlo
    return {
        "input_ids": prompt + respuesta,
        "labels": [IGNORAR] * len(prompt) + respuesta,
    }


def preparar_datos(tokenizador, ruta: str, largo_maximo: int, fraccion_validacion: float, semilla: int):
    with open(ruta, encoding="utf-8") as archivo:
        ejemplos = [json.loads(linea) for linea in archivo if linea.strip()]
    tokenizados = [tokenizar(tokenizador, ejemplo, largo_maximo) for ejemplo in ejemplos]
    validos = [t for t in tokenizados if t is not None]
    print(f"{len(validos)} de {len(ejemplos)} ejemplos caben en {largo_maximo} tokens")

    # Cada ejemplo de sft.jsonl es un problema distinto, así que basta separar al azar.
    random.Random(semilla).shuffle(validos)
    cantidad_validacion = max(1, round(len(validos) * fraccion_validacion)) if fraccion_validacion > 0 else 0
    return validos[cantidad_validacion:], validos[:cantidad_validacion]


class Agrupador:
    # Rellena cada lote al largo del ejemplo más largo. El relleno no cuenta en la atención ni en la pérdida.
    def __init__(self, id_relleno: int):
        self.id_relleno = id_relleno

    def __call__(self, ejemplos: list[dict]) -> dict:
        import torch

        largo = max(len(e["input_ids"]) for e in ejemplos)
        lote = {"input_ids": [], "labels": [], "attention_mask": []}
        for ejemplo in ejemplos:
            relleno = largo - len(ejemplo["input_ids"])
            lote["input_ids"].append(ejemplo["input_ids"] + [self.id_relleno] * relleno)
            lote["labels"].append(ejemplo["labels"] + [IGNORAR] * relleno)
            lote["attention_mask"].append([1] * len(ejemplo["input_ids"]) + [0] * relleno)
        return {clave: torch.tensor(valores) for clave, valores in lote.items()}


def guardar_perdida(historial: list[dict], carpeta: Path) -> None:
    # Se redibuja desde el historial completo del Trainer, que se restaura al retomar un checkpoint.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entrenamiento = [(r["step"], r["loss"]) for r in historial if "loss" in r]
    validacion = [(r["step"], r["eval_loss"]) for r in historial if "eval_loss" in r]

    with open(carpeta / "perdida.csv", "w", encoding="utf-8") as archivo:
        archivo.write("paso,conjunto,perdida\n")
        for conjunto, puntos in (("entrenamiento", entrenamiento), ("validacion", validacion)):
            for paso, perdida in puntos:
                archivo.write(f"{paso},{conjunto},{perdida}\n")

    figura, eje = plt.subplots(figsize=(8, 5))
    for nombre, puntos, estilo in (("Entrenamiento", entrenamiento, "-"), ("Validación", validacion, "o-")):
        if puntos:
            eje.plot(*zip(*puntos), estilo, label=nombre)
    eje.set_xlabel("Paso de optimización")
    eje.set_ylabel("Pérdida (entropía cruzada por token)")
    eje.set_title("SFT de Ministral 3 3B Reasoning")
    eje.grid(alpha=0.3)
    eje.legend()
    figura.tight_layout()
    figura.savefig(carpeta / "perdida.png", dpi=120)
    plt.close(figura)


def main() -> None:
    argumentos = leer_argumentos()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available():
        raise SystemExit("No hay GPU disponible: el entrenamiento necesita CUDA.")

    salida = Path(argumentos.salida)
    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / "configuracion.json", "w", encoding="utf-8") as archivo:
        json.dump(vars(argumentos), archivo, ensure_ascii=False, indent=2)

    tokenizador = AutoTokenizer.from_pretrained(argumentos.modelo)
    entrenamiento, validacion = preparar_datos(
        tokenizador, argumentos.datos, argumentos.largo_maximo, argumentos.fraccion_validacion, argumentos.semilla
    )
    id_relleno = tokenizador.pad_token_id if tokenizador.pad_token_id is not None else tokenizador.eos_token_id

    modelo = AutoModelForImageTextToText.from_pretrained(argumentos.modelo, dtype=torch.bfloat16)
    modelo.config.use_cache = False  # La caché no sirve al entrenar y choca con gradient checkpointing
    modelo = get_peft_model(modelo, LoraConfig(
        r=argumentos.rango_lora,
        lora_alpha=2 * argumentos.rango_lora,
        lora_dropout=0.05,
        target_modules=MODULOS_LORA,
        task_type="CAUSAL_LM",
    ))
    modelo.print_trainable_parameters()

    class GuardarPerdida(TrainerCallback):
        def on_log(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                guardar_perdida(state.log_history, salida)

    hay_validacion = len(validacion) > 0
    trainer = Trainer(
        model=modelo,
        args=TrainingArguments(
            output_dir=str(salida),
            num_train_epochs=argumentos.epocas,
            max_steps=argumentos.max_pasos,
            learning_rate=argumentos.tasa,
            lr_scheduler_type="cosine",
            warmup_steps=0.05,  # En transformers 5, un valor menor que 1 es la fracción de los pasos totales
            per_device_train_batch_size=argumentos.lote,
            per_device_eval_batch_size=argumentos.lote,
            gradient_accumulation_steps=argumentos.acumulacion,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True,
            logging_steps=1,
            eval_strategy="steps" if hay_validacion else "no",
            eval_steps=argumentos.pasos_validacion,
            save_strategy="steps",
            save_steps=argumentos.pasos_guardado,
            save_total_limit=argumentos.checkpoints_maximos,
            report_to="none",
            seed=argumentos.semilla,
            remove_unused_columns=False,
        ),
        train_dataset=entrenamiento,
        eval_dataset=validacion if hay_validacion else None,
        data_collator=Agrupador(id_relleno),
        callbacks=[GuardarPerdida()],
    )

    ultimo = get_last_checkpoint(str(salida))
    if ultimo:
        print(f"Retomando desde {ultimo}")
    trainer.train(resume_from_checkpoint=ultimo)

    # El adaptador final queda aparte de los checkpoints, que save_total_limit puede borrar.
    trainer.save_model(str(salida / "final"))
    tokenizador.save_pretrained(str(salida / "final"))
    guardar_perdida(trainer.state.log_history, salida)
    print(f"Listo: adaptador en {salida / 'final'}, pérdida en {salida / 'perdida.png'}")


if __name__ == "__main__":
    main()
