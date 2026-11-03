import re

from instancia import Instancia

PATRON_PROMPT = re.compile(
    r"Valor de cada nodo:\n(?P<valores>(?:\d+: -?\d+\n)+)\n"
    r"Conexiones:\n(?P<aristas>\(ninguna\)|(?:\d+-\d+\n?)+)\n*"
    r"Costo c = (?P<costo>\d+)\n"
)


def leer_prompt(prompt: str) -> Instancia:
    # Recupera el árbol desde el prompt, para calificar sin guardar nada aparte.
    coincidencia = PATRON_PROMPT.search(prompt)
    if coincidencia is None:
        raise ValueError("El prompt no tiene el formato de este ambiente")

    pares = [linea.split(": ") for linea in coincidencia["valores"].splitlines()]
    if [int(nodo) for nodo, _ in pares] != list(range(1, len(pares) + 1)):
        raise ValueError("Los nodos del prompt no están numerados desde 1")

    texto_aristas = "" if coincidencia["aristas"] == "(ninguna)" else coincidencia["aristas"]
    aristas = tuple(tuple(map(int, linea.split("-"))) for linea in texto_aristas.splitlines())
    if len(aristas) != len(pares) - 1 or any(not 1 <= n <= len(pares) for arista in aristas for n in arista):
        raise ValueError("Las conexiones del prompt no forman un árbol válido")

    return Instancia(tuple(int(valor) for _, valor in pares), aristas, int(coincidencia["costo"]))


def generar_prompt(instancia: Instancia) -> str:
    valores = "\n".join(f"{i}: {valor}" for i, valor in enumerate(instancia.valores, 1))
    aristas = "\n".join(f"{a}-{b}" for a, b in instancia.aristas) or "(ninguna)"

    return f"""Elige un conjunto de nodos de este árbol para obtener el mayor puntaje posible.

Valor de cada nodo:
{valores}

Conexiones:
{aristas}

Costo c = {instancia.costo}

Puntaje: cada nodo elegido aporta su valor menos c por cada vecino que también esté elegido. Los nodos no elegidos no aportan nada. Si no eliges ninguno, el puntaje es 0.

Termina con los números de los nodos elegidos, por ejemplo:
<respuesta>2 5 9</respuesta>
Si no eliges ninguno, escribe <respuesta></respuesta>."""
