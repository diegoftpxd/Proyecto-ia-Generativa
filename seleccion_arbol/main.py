# Ejemplo de uso del ambiente de selección sobre árboles. Ejecutar desde esta carpeta:
#   python main.py                          respuestas de ejemplo, sin modelo
#   python main.py --nivel 4 --semilla 7    otro árbol
#   python main.py --lmstudio qwen3.5-2b    responde un modelo cargado en LM Studio
import argparse

from src.dataset import resolver
from src.entorno import EntornoSeleccion, reward
from src.generador import NIVELES
from src.modelo import ModeloLMStudio


def main() -> None:
    parser = argparse.ArgumentParser(description="Califica respuestas en el ambiente de selección sobre árboles.")
    parser.add_argument("--nivel", type=int, default=2, choices=range(1, len(NIVELES) + 1))
    parser.add_argument("--semilla", type=int, default=0, help="la misma semilla genera el mismo árbol")
    parser.add_argument("--lmstudio", metavar="MODELO", help="nombre del modelo en LM Studio (servidor en el puerto 1235)")
    argumentos = parser.parse_args()

    # El ambiente genera un árbol del nivel y el prompt que recibe el modelo.
    entorno = EntornoSeleccion(semilla=argumentos.semilla)
    prompt = entorno.reset(NIVELES[argumentos.nivel - 1])
    instancia = entorno.instancia
    print(prompt, end="\n\n")

    if argumentos.lmstudio:
        completions = {"LM Studio": ModeloLMStudio(nombre=argumentos.lmstudio).generar(prompt)}
        print(completions["LM Studio"], end="\n\n")
    else:
        seleccion, _ = resolver(instancia)
        completions = {
            "óptima": "<respuesta>" + " ".join(map(str, sorted(seleccion))) + "</respuesta>",
            "todos los nodos": "<respuesta>" + " ".join(map(str, range(1, instancia.cantidad_nodos + 1))) + "</respuesta>",
            "sin bloque de respuesta": "No estoy seguro.",
        }

    # reward(prompt, completion) -> float: recupera el árbol desde el prompt y califica la respuesta.
    # Va de -1 (respuesta inválida) a 1 (óptima); igualar el mejor atajo vale 0.5.
    for nombre, completion in completions.items():
        print(f"{nombre}: recompensa = {reward(prompt, completion):.3f}")


if __name__ == "__main__":
    main()
