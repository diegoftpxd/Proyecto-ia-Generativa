"""Prueba rápida de un modelo en la GPU de tu PC, para saber si cabe y cuánto tardaría baseline_local.py.

Mide la velocidad de generación y juega un episodio por nivel con la misma configuración de la línea base.
Los episodios van a un archivo temporal, no a resultados/. Ejecútalo desde seleccion_arbol:

    python benchmark_local.py mistralai/Ministral-3-3B-Reasoning-2512 --4bit
    python benchmark_local.py unsloth/Llama-3.2-1B-Instruct --niveles 4,5,6 --sin-velocidad
"""
import argparse
import tempfile
import time
from collections import Counter
from pathlib import Path

from baseline_local import DATOS, INTENTOS, NIVELES, PRESUPUESTO, crear_modelo, estado_gpu
from evaluacion import _evaluar_episodio, leer_resultados
from instancia import Instancia

TOKENS_VELOCIDAD = 1000


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nombre_hf", help="Modelo en Hugging Face")
    parser.add_argument("--4bit", dest="en_4bit", action="store_true", help="Cargar en 4 bits (bitsandbytes)")
    parser.add_argument("--niveles", default=",".join(map(str, NIVELES)), help="Niveles separados por comas")
    parser.add_argument("--sin-velocidad", action="store_true", help="No medir la velocidad de generación")
    parser.add_argument("--salida", type=Path, help="Archivo de episodios (por defecto, uno temporal)")
    argumentos = parser.parse_args()
    niveles = [int(nivel) for nivel in argumentos.niveles.split(",")]
    salida = argumentos.salida or Path(tempfile.gettempdir()) / "benchmark_local.jsonl"

    episodios = leer_resultados(DATOS)
    arboles = Counter(e["nivel"] for e in episodios)
    primeros = {}
    for episodio in episodios:
        primeros.setdefault(episodio["nivel"], episodio)

    modelo = crear_modelo(argumentos.nombre_hf, argumentos.en_4bit)
    inicio = time.perf_counter()
    modelo._cargar()
    print(f"Cargado en {time.perf_counter() - inicio:.0f} s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB en la GPU. "
          f"GPU {estado_gpu()}", flush=True)

    if not argumentos.sin_velocidad:
        # Se fuerzan TOKENS_VELOCIDAD tokens nuevos sobre un prompt real, tras un calentamiento.
        _, entrada = modelo._preparar(primeros[niveles[0]]["prompt"])
        entrada = entrada.to(modelo._modelo.device)
        with torch.inference_mode():
            modelo._modelo.generate(**entrada, max_new_tokens=10)
            torch.cuda.synchronize()
            inicio = time.perf_counter()
            modelo._modelo.generate(
                **entrada, max_new_tokens=TOKENS_VELOCIDAD, min_new_tokens=TOKENS_VELOCIDAD,
                do_sample=True, temperature=modelo.temperatura,
            )
            torch.cuda.synchronize()
        velocidad = TOKENS_VELOCIDAD / (time.perf_counter() - inicio)
        tope = PRESUPUESTO["tokens_pensamiento"]
        print(f"Velocidad: {velocidad:.1f} tokens/s; un episodio que agota los {tope} tokens tardaría "
              f"unos {tope / velocidad:.0f} s. GPU {estado_gpu()}", flush=True)

    salida.unlink(missing_ok=True)
    for nivel in niveles:
        e = primeros[nivel]
        instancia = Instancia(tuple(e["valores"]), tuple(map(tuple, e["aristas"])), e["costo"])
        _evaluar_episodio(modelo, f"benchmark {argumentos.nombre_hf}", instancia, nivel, e["id"], None, salida, 1)
    modelo.liberar()

    filas = leer_resultados(salida)
    horas = sum(f["segundos"] * arboles[f["nivel"]] * INTENTOS for f in filas) / 3600
    print("\n=== Resumen ===")
    print(f"Máximo en la GPU: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"Episodios: {len(filas)}, agotaron los tokens: {sum(f['fin'] == 'length' for f in filas)}, "
          f"respuestas forzadas: {sum(bool(f['respuesta_forzada']) for f in filas)}")
    print(f"Recompensas: {[round(f['recompensa'], 3) for f in filas]}")
    print(f"Errores: {[f['error'] for f in filas]}")
    print(f"Estimación de baseline_local.py en los niveles {niveles} "
          f"({sum(arboles[n] for n in niveles) * INTENTOS} episodios): {horas:.1f} h")
    print(f"Episodios guardados en {salida}")


if __name__ == "__main__":
    main()
