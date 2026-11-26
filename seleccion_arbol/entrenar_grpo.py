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
# GRPO con LoRA sobre Ministral 3 3B Reasoning ya ajustado con SFT (por defecto, el de solver masivo) o, con
# --sin-sft, sobre el modelo base. En una GPU.
# Ejecutar desde esta carpeta:
#   uv run entrenar_grpo.py                                              todo con los valores por defecto
#   uv run entrenar_grpo.py --salida resultados/grpo_prueba --pasos 2 --problemas 2
#   uv run entrenar_grpo.py --sin-sft --salida resultados/09_grpo_ministral_base ...   desde el modelo base
# Si en la carpeta de salida ya hay checkpoints, retoma desde el último (útil si Slurm corta el trabajo).
#
# Cada paso:
#   1. Toma --problemas problemas de datos/entrenamiento.jsonl y genera --generaciones respuestas por problema.
#      Una respuesta cortada por --tokens no tiene bloque de respuesta y vale -1.
#   2. reward() de src/entorno.py califica cada respuesta, igual que en la evaluación.
#   3. La ventaja de una respuesta es su recompensa menos la media de su grupo (Dr. GRPO: sin dividir por la
#      desviación, que da más peso a los grupos casi iguales). Un grupo con recompensas iguales no aporta.
#   4. Un solo paso de gradiente por lote generado: como el lote viene del modelo actual, la razón de
#      probabilidades vale 1 y el recorte de PPO no actúa, así que la pérdida es -ventaja × log p por token.
#
# Registros en la carpeta de salida:
#   recompensas.csv   una fila por respuesta: paso, problema, nivel, recompensa, tokens y si se cortó
#   metricas.csv      una fila por paso y nivel, más una con nivel "todos": media, desviación, mínimo y
#                     máximo de la recompensa, fracción de óptimas y de cortadas, tokens y grupos con señal
#   recompensa.png    la recompensa media y la fracción de óptimas por nivel a lo largo de los pasos
#   muestras.jsonl    las respuestas completas de un grupo por paso, para revisar las trazas a mano
#
# No usa TRL: el tokenizador de Mistral (MistralCommonBackend) no tiene varios métodos que usan esas
# bibliotecas (ver HastaCierre en src/modelo_hf.py). Aquí todo se hace con ids de tokens, como en entrenar_sft.py.
import argparse
import csv
import json
import random
import shutil
import statistics
import time
from pathlib import Path

from entrenar_sft import MODULOS_LORA
from src.entorno import reward
from src.modelo_hf import extraer_respuesta, tokenizar_prompt

MODELO = "mistralai/Ministral-3-3B-Reasoning-2512"
SFT = "resultados/07_sft_ministral_solver_masivo/final"
VENTANA = 10  # Pasos de la media móvil del gráfico
COLUMNAS_RECOMPENSAS = ["paso", "id", "nivel", "generacion", "recompensa", "optima", "tokens", "cortada"]
COLUMNAS_METRICAS = [
    "paso", "nivel", "respuestas", "recompensa_media", "recompensa_desviacion", "recompensa_min", "recompensa_max",
    "optimas", "cortadas", "tokens_medios", "grupos", "grupos_con_senal", "tasa", "norma_gradiente", "segundos",
]


def leer_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO con LoRA sobre Ministral 3 3B Reasoning ajustado con SFT.")
    parser.add_argument("--modelo", default=MODELO, help="id en Hugging Face o carpeta local del modelo base")
    parser.add_argument("--sft", default=SFT, help="adaptador de entrenar_sft.py, que se fusiona antes de GRPO")
    parser.add_argument("--sin-sft", dest="sft", action="store_const", const=None,
                        help="parte del modelo base, sin fusionar ningún SFT")
    parser.add_argument("--datos", default="datos/entrenamiento.jsonl")
    # Con el SFT de solver masivo, los niveles 1 a 4 tienen pass@1 de 0,91 o más: casi todos sus grupos
    # tienen recompensas iguales y no aportan gradiente.
    parser.add_argument("--niveles", type=int, nargs="+", default=[5, 6, 7, 8])
    parser.add_argument("--salida", default="resultados/08_grpo_ministral")
    parser.add_argument("--pasos", type=int, default=300, help="pasos de optimización")
    parser.add_argument("--problemas", type=int, default=16, help="problemas en cada paso")
    parser.add_argument("--generaciones", type=int, default=8, help="respuestas por problema: el grupo de GRPO")
    # Con los valores por defecto se generan 128 respuestas a la vez. Si la memoria no alcanza, bajar este
    # valor: genera en tandas, sin cambiar el lote de GRPO.
    parser.add_argument("--lote-generacion", type=int, default=128, help="respuestas que se generan a la vez")
    # El SFT usa una mediana de 1365 tokens en el nivel 8 y menos de 1000 en el 7. Cortar en 2048 empuja a
    # responder antes del presupuesto de la evaluación (3400) sin castigar las respuestas normales.
    parser.add_argument("--tokens", type=int, default=2048, help="máximo de tokens generados por respuesta")
    parser.add_argument("--temperatura", type=float, default=1.0)
    parser.add_argument("--tasa", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--calentamiento", type=int, default=10, help="pasos de subida lineal del learning rate")
    parser.add_argument("--beta", type=float, default=0.0, help="peso del KL contra el modelo SFT; con 0 no se calcula")
    # Con 4 secuencias de ~2600 tokens, los logits en fp32 ocupan unos 5 GB. Con más, crecen en proporción.
    parser.add_argument("--micro-lote", type=int, default=4, help="secuencias por pasada hacia atrás")
    parser.add_argument("--rango-lora", type=int, default=64, help="alpha es el doble del rango")
    parser.add_argument("--pasos-guardado", type=int, default=20, help="cada cuántos pasos se guarda un checkpoint")
    parser.add_argument("--checkpoints-maximos", type=int, default=3, help="se borran los más antiguos")
    parser.add_argument("--semilla", type=int, default=0)
    return parser.parse_args()


def cargar_problemas(ruta: str, niveles: list[int]) -> list[dict]:
    with open(ruta, encoding="utf-8") as archivo:
        problemas = [json.loads(linea) for linea in archivo if linea.strip()]
    elegidos = [{"id": p["id"], "nivel": p["nivel"], "prompt": p["prompt"]} for p in problemas if p["nivel"] in niveles]
    if not elegidos:
        raise SystemExit(f"No hay problemas de los niveles {niveles} en {ruta}")
    return elegidos


def problemas_del_paso(problemas: list[dict], paso: int, cantidad: int, semilla: int) -> list[dict]:
    # La muestra depende solo del paso: al retomar, los pasos siguientes usan los mismos problemas que sin el corte.
    return random.Random(f"{semilla}-{paso}").sample(problemas, cantidad)


def ventajas(recompensas: list[float], generaciones: int) -> list[float]:
    # Las recompensas vienen en grupos consecutivos de `generaciones`, uno por problema.
    resultado = []
    for inicio in range(0, len(recompensas), generaciones):
        grupo = recompensas[inicio:inicio + generaciones]
        if max(grupo) == min(grupo):  # Exactamente 0: restar la media en punto flotante podría no darlo
            resultado.extend(0.0 for _ in grupo)
            continue
        media = sum(grupo) / len(grupo)
        resultado.extend(recompensa - media for recompensa in grupo)
    return resultado


def recortar(ids: list[int], fin: set[int]) -> tuple[list[int], bool]:
    # Deja los tokens hasta el primer fin de secuencia, incluido, que el modelo debe aprender a emitir.
    # Lo que sigue es relleno de generate. Sin fin, la respuesta se cortó por el presupuesto.
    for posicion, token in enumerate(ids):
        if token in fin:
            return ids[:posicion + 1], True
    return ids, False


def tasa_del_paso(paso: int, tasa: float, calentamiento: int) -> float:
    return tasa * min(1.0, (paso + 1) / calentamiento) if calentamiento > 0 else tasa


def suavizar(valores: list[float], ventana: int = VENTANA) -> list[float]:
    return [sum(valores[max(0, i - ventana + 1):i + 1]) / len(valores[max(0, i - ventana + 1):i + 1])
            for i in range(len(valores))]


def filas_recompensas(paso: int, grupos: list[dict]) -> list[dict]:
    return [
        {
            "paso": paso, "id": grupo["id"], "nivel": grupo["nivel"], "generacion": generacion,
            "recompensa": recompensa, "optima": int(recompensa == 1.0),
            "tokens": grupo["tokens"][generacion], "cortada": int(grupo["cortadas"][generacion]),
        }
        for grupo in grupos
        for generacion, recompensa in enumerate(grupo["recompensas"])
    ]


def filas_metricas(paso: int, grupos: list[dict], tasa: float, norma: float, segundos: float) -> list[dict]:
    # Una fila por nivel, más una con todos los niveles juntos.
    conjuntos = [(nivel, [g for g in grupos if g["nivel"] == nivel]) for nivel in sorted({g["nivel"] for g in grupos})]
    filas = []
    for nivel, propios in conjuntos + [("todos", grupos)]:
        recompensas = [r for g in propios for r in g["recompensas"]]
        filas.append({
            "paso": paso,
            "nivel": nivel,
            "respuestas": len(recompensas),
            "recompensa_media": statistics.fmean(recompensas),
            "recompensa_desviacion": statistics.pstdev(recompensas),
            "recompensa_min": min(recompensas),
            "recompensa_max": max(recompensas),
            "optimas": sum(r == 1.0 for r in recompensas) / len(recompensas),
            "cortadas": sum(c for g in propios for c in g["cortadas"]) / len(recompensas),
            "tokens_medios": statistics.fmean(t for g in propios for t in g["tokens"]),
            "grupos": len(propios),
            "grupos_con_senal": sum(len(set(g["recompensas"])) > 1 for g in propios),
            "tasa": tasa,
            "norma_gradiente": norma,
            "segundos": round(segundos, 1),
        })
    return filas


def preparar_csv(ruta: Path, columnas: list[str], inicio: int) -> list[dict]:
    # Al retomar se descartan los pasos desde el checkpoint en adelante, que se vuelven a hacer.
    filas = []
    if ruta.exists():
        with open(ruta, encoding="utf-8", newline="") as archivo:
            filas = [f for f in csv.DictReader(archivo) if int(f["paso"]) < inicio]
    with open(ruta, "w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)
    return filas


def agregar_csv(ruta: Path, columnas: list[str], filas: list[dict]) -> None:
    with open(ruta, "a", encoding="utf-8", newline="") as archivo:
        csv.DictWriter(archivo, fieldnames=columnas).writerows(filas)


def guardar_grafico(metricas: list[dict], carpeta: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Las filas leídas del CSV al retomar traen texto: se comparan y convierten como texto.
    niveles = sorted({str(f["nivel"]) for f in metricas} - {"todos"}, key=int) + ["todos"]
    figura, ejes = plt.subplots(1, 2, figsize=(13, 5))
    for eje, clave, titulo in ((ejes[0], "recompensa_media", "Recompensa media"),
                               (ejes[1], "optimas", "Fracción de respuestas óptimas")):
        for nivel in niveles:
            puntos = [(int(f["paso"]), float(f[clave])) for f in metricas if str(f["nivel"]) == nivel]
            pasos, valores = zip(*puntos)
            if nivel == "todos":
                eje.plot(pasos, suavizar(list(valores)), color="black", linewidth=2, label="Todos")
            else:
                eje.plot(pasos, suavizar(list(valores)), label=f"Nivel {nivel}", alpha=0.8)
        eje.set_title(f"{titulo} (media móvil de {VENTANA} pasos)")
        eje.set_xlabel("Paso de optimización")
        eje.grid(alpha=0.3)
        eje.legend()
    figura.suptitle(f"GRPO de Ministral 3 3B Reasoning ({carpeta.name})")
    figura.tight_layout()
    figura.savefig(carpeta / "recompensa.png", dpi=120)
    plt.close(figura)


def ultimo_checkpoint(salida: Path) -> Path | None:
    # estado.pt se escribe al final: un checkpoint sin él quedó a medias.
    completos = [c for c in salida.glob("checkpoint-*") if (c / "estado.pt").exists()]
    return max(completos, key=lambda c: int(c.name.split("-")[1]), default=None)


def guardar_checkpoint(modelo, optimizador, paso: int, salida: Path, maximos: int) -> None:
    import torch

    carpeta = salida / f"checkpoint-{paso}"
    modelo.save_pretrained(str(carpeta))  # Es un adaptador completo: evaluar_sft.py lo acepta como --adaptador
    torch.save({"paso": paso, "optimizador": optimizador.state_dict()}, carpeta / "estado.pt")
    completos = sorted((c for c in salida.glob("checkpoint-*") if (c / "estado.pt").exists()),
                       key=lambda c: int(c.name.split("-")[1]))
    for viejo in completos[:-maximos]:
        shutil.rmtree(viejo)


def generar(modelo, prompts: list[list[int]], argumentos, relleno: int, fin: set[int], dispositivo):
    import torch

    # Cada prompt se repite una vez por generación, con relleno a la izquierda para generar en lote.
    filas = [ids for ids in prompts for _ in range(argumentos.generaciones)]
    respuestas = []
    modelo.eval()
    for inicio in range(0, len(filas), argumentos.lote_generacion):
        tanda = filas[inicio:inicio + argumentos.lote_generacion]
        largo = max(map(len, tanda))
        entrada = torch.tensor([[relleno] * (largo - len(f)) + f for f in tanda], device=dispositivo)
        atencion = torch.tensor([[0] * (largo - len(f)) + [1] * len(f) for f in tanda], device=dispositivo)
        with torch.inference_mode():
            # top_k=0 y top_p=1 apagan los filtros que traiga generation_config: así se muestrea de la misma
            # distribución (logits / temperatura) cuyas probabilidades se optimizan después.
            salida = modelo.generate(
                input_ids=entrada,
                attention_mask=atencion,
                max_new_tokens=argumentos.tokens,
                do_sample=True,
                temperature=argumentos.temperatura,
                top_k=0,
                top_p=1.0,
                eos_token_id=sorted(fin),
                pad_token_id=relleno,
                use_cache=True,
            )
        respuestas += [recortar(fila.tolist(), fin) for fila in salida[:, largo:]]
    modelo.train()
    return respuestas


def log_probabilidades(modelo, ids, atencion, temperatura: float):
    import torch

    # La posición t predice el token t + 1.
    logits = modelo(input_ids=ids, attention_mask=atencion, use_cache=False).logits[:, :-1].float() / temperatura
    return torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)


def rellenar(secuencias: list[list[int]], largos_prompt: list[int], relleno: int, dispositivo):
    import torch

    # Relleno a la derecha: no cambia las posiciones de los tokens reales. La máscara marca, entre las
    # predicciones, solo las de los tokens de la respuesta.
    largo = max(map(len, secuencias))
    ids = [s + [relleno] * (largo - len(s)) for s in secuencias]
    atencion = [[1] * len(s) + [0] * (largo - len(s)) for s in secuencias]
    mascara = [[0] * (p - 1) + [1] * (len(s) - p) + [0] * (largo - len(s)) for s, p in zip(secuencias, largos_prompt)]
    return (torch.tensor(ids, device=dispositivo), torch.tensor(atencion, device=dispositivo),
            torch.tensor(mascara, device=dispositivo, dtype=torch.float32))


def optimizar(modelo, optimizador, secuencias, largos_prompt, ventaja, argumentos, relleno, dispositivo) -> float | None:
    import torch

    # Sin KL, una secuencia con ventaja 0 no aporta gradiente: solo se calculan las demás.
    indices = [i for i in range(len(secuencias)) if argumentos.beta > 0 or ventaja[i] != 0]
    if not indices:
        return None  # Ningún grupo tuvo señal: no se actualiza, para que el momento de AdamW no mueva los pesos
    # Dr. GRPO: se divide por una constante y no por el largo de cada respuesta, que favorecería a las largas
    # cuando la ventaja es negativa.
    normalizador = len(secuencias) * argumentos.tokens
    optimizador.zero_grad(set_to_none=True)
    for inicio in range(0, len(indices), argumentos.micro_lote):
        micro = indices[inicio:inicio + argumentos.micro_lote]
        ids, atencion, mascara = rellenar(
            [secuencias[i] for i in micro], [largos_prompt[i] for i in micro], relleno, dispositivo
        )
        actual = log_probabilidades(modelo, ids, atencion, argumentos.temperatura)
        pesos = torch.tensor([ventaja[i] for i in micro], device=dispositivo)[:, None]
        perdida = -pesos * actual
        if argumentos.beta > 0:
            # El modelo sin el LoRA de GRPO es el de partida (SFT fusionado o base): sirve de referencia sin otra copia.
            with torch.no_grad(), modelo.disable_adapter():
                referencia = log_probabilidades(modelo, ids, atencion, argumentos.temperatura)
            diferencia = referencia - actual
            perdida = perdida + argumentos.beta * (torch.exp(diferencia) - diferencia - 1)
        ((perdida * mascara).sum() / normalizador).backward()
    parametros = [p for p in modelo.parameters() if p.requires_grad]
    norma = torch.nn.utils.clip_grad_norm_(parametros, 1.0).item()
    optimizador.step()
    return norma


def main() -> None:
    argumentos = leer_argumentos()

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("No hay GPU disponible: el entrenamiento necesita CUDA.")
    if argumentos.sft and not (Path(argumentos.sft) / "adapter_config.json").exists():
        raise SystemExit(f"No hay un adaptador SFT en {argumentos.sft}: falta adapter_config.json.")

    dispositivo = torch.device("cuda", 0)
    torch.manual_seed(argumentos.semilla)
    salida = Path(argumentos.salida)
    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / "configuracion.json", "w", encoding="utf-8") as archivo:
        json.dump(vars(argumentos), archivo, ensure_ascii=False, indent=2)
    ultimo = ultimo_checkpoint(salida)

    tokenizador = AutoTokenizer.from_pretrained(argumentos.modelo)
    modelo = AutoModelForImageTextToText.from_pretrained(argumentos.modelo, dtype=torch.bfloat16)
    # GRPO parte del modelo con el SFT fusionado, o del modelo base con --sin-sft, y entrena un LoRA nuevo encima.
    if argumentos.sft:
        modelo = PeftModel.from_pretrained(modelo, argumentos.sft).merge_and_unload()
    modelo.config.use_cache = False
    modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    fin = {tokenizador.eos_token_id}
    eos = modelo.generation_config.eos_token_id
    fin.update(eos if isinstance(eos, list) else [] if eos is None else [eos])
    relleno = tokenizador.pad_token_id if tokenizador.pad_token_id is not None else tokenizador.eos_token_id

    if ultimo:
        modelo = PeftModel.from_pretrained(modelo, str(ultimo), is_trainable=True)
    else:
        # Sin dropout: con él, las probabilidades que se optimizan no serían las del modelo que generó.
        modelo = get_peft_model(modelo, LoraConfig(
            r=argumentos.rango_lora,
            lora_alpha=2 * argumentos.rango_lora,
            lora_dropout=0.0,
            target_modules=MODULOS_LORA,
            task_type="CAUSAL_LM",
        ))
    modelo.to(dispositivo)
    modelo.print_trainable_parameters()

    optimizador = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=argumentos.tasa, weight_decay=0.0)
    inicio = 0
    if ultimo:
        estado = torch.load(ultimo / "estado.pt", map_location=dispositivo)
        optimizador.load_state_dict(estado["optimizador"])
        inicio = estado["paso"]
    print(f"Retomando desde {ultimo}" if ultimo else f"Empezando desde {argumentos.sft or 'el modelo base, sin SFT'}")

    ruta_recompensas = salida / "recompensas.csv"
    ruta_metricas = salida / "metricas.csv"
    preparar_csv(ruta_recompensas, COLUMNAS_RECOMPENSAS, inicio)
    metricas = preparar_csv(ruta_metricas, COLUMNAS_METRICAS, inicio)

    problemas = cargar_problemas(argumentos.datos, argumentos.niveles)
    generaciones = argumentos.generaciones
    print(f"{argumentos.problemas} problemas y {argumentos.problemas * generaciones} respuestas por paso")
    for paso in range(inicio, argumentos.pasos):
        comienzo = time.time()
        tasa = tasa_del_paso(paso, argumentos.tasa, argumentos.calentamiento)
        for grupo in optimizador.param_groups:
            grupo["lr"] = tasa

        lote = problemas_del_paso(problemas, paso, argumentos.problemas, argumentos.semilla)
        prompts = [list(tokenizar_prompt(tokenizador, p["prompt"], razonar=True)["input_ids"]) for p in lote]
        respuestas = generar(modelo, prompts, argumentos, relleno, fin, dispositivo)

        grupos, recompensas, textos = [], [], []
        for numero, problema in enumerate(lote):
            texto_prompt = tokenizador.decode(prompts[numero], skip_special_tokens=False)
            propias = respuestas[numero * generaciones:(numero + 1) * generaciones]
            # Se califica igual que en la evaluación: lo que queda después del razonamiento, si lo hay.
            completos = [tokenizador.decode(ids, skip_special_tokens=True) for ids, _ in propias]
            valores = [reward(problema["prompt"], extraer_respuesta(texto_prompt, texto)) for texto in completos]
            recompensas += valores
            textos.append(completos)
            grupos.append({
                "id": problema["id"],
                "nivel": problema["nivel"],
                "recompensas": valores,
                "tokens": [len(ids) for ids, _ in propias],
                "cortadas": [not terminada for _, terminada in propias],
            })

        secuencias = [prompts[i // generaciones] + ids for i, (ids, _) in enumerate(respuestas)]
        largos_prompt = [len(prompts[i // generaciones]) for i in range(len(respuestas))]
        norma = optimizar(modelo, optimizador, secuencias, largos_prompt, ventajas(recompensas, generaciones),
                          argumentos, relleno, dispositivo)

        # norma_gradiente vacía en el CSV: ningún grupo tuvo señal y el paso no actualizó el modelo.
        nuevas = filas_metricas(paso, grupos, tasa, "" if norma is None else norma, time.time() - comienzo)
        metricas += nuevas
        agregar_csv(ruta_recompensas, COLUMNAS_RECOMPENSAS, filas_recompensas(paso, grupos))
        agregar_csv(ruta_metricas, COLUMNAS_METRICAS, nuevas)
        # Un grupo por paso, para revisar a mano si el modelo encuentra un atajo que el filtro no cubre.
        with open(salida / "muestras.jsonl", "a", encoding="utf-8") as archivo:
            archivo.write(json.dumps({
                "paso": paso, "id": lote[0]["id"], "nivel": lote[0]["nivel"],
                "recompensas": grupos[0]["recompensas"], "respuestas": textos[0],
            }, ensure_ascii=False) + "\n")
        guardar_grafico(metricas, salida)

        por_nivel = ", ".join(f"n{f['nivel']} {f['recompensa_media']:.2f}" for f in nuevas[:-1])
        total = nuevas[-1]
        print(f"Paso {paso + 1}/{argumentos.pasos}: recompensa {total['recompensa_media']:.3f} ({por_nivel}), "
              f"óptimas {total['optimas']:.3f}, cortadas {total['cortadas']:.3f}, tokens {total['tokens_medios']:.0f}, "
              f"grupos con señal {total['grupos_con_senal']}/{total['grupos']}, {total['segundos']} s")

        if (paso + 1) % argumentos.pasos_guardado == 0 or paso + 1 == argumentos.pasos:
            guardar_checkpoint(modelo, optimizador, paso + 1, salida, argumentos.checkpoints_maximos)

    # El adaptador final queda aparte de los checkpoints, que se van borrando.
    modelo.save_pretrained(str(salida / "final"))
    tokenizador.save_pretrained(str(salida / "final"))
    print(f"Listo: adaptador en {salida / 'final'}, recompensas en {ruta_recompensas} y {ruta_metricas}")


if __name__ == "__main__":
    main()
