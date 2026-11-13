import re

from src.instancia import Instancia

PATRON_PROMPT = re.compile(
    r"Node values \(node: value\):\n(?P<valores>(?:\d+: -?\d+\n)+)\n"
    r"Edges:\n(?P<aristas>\(none\)|(?:\d+-\d+\n?)+)\n*"
    r"Cost c = (?P<costo>\d+)\n"
)


def leer_prompt(prompt: str) -> Instancia:
    # Recupera el árbol desde el prompt, para calificar sin guardar nada aparte.
    coincidencia = PATRON_PROMPT.search(prompt)
    if coincidencia is None:
        raise ValueError("El prompt no tiene el formato de este ambiente")

    pares = [linea.split(": ") for linea in coincidencia["valores"].splitlines()]
    if [int(nodo) for nodo, _ in pares] != list(range(1, len(pares) + 1)):
        raise ValueError("Los nodos del prompt no están numerados desde 1")

    texto_aristas = "" if coincidencia["aristas"] == "(none)" else coincidencia["aristas"]
    aristas = tuple(tuple(map(int, linea.split("-"))) for linea in texto_aristas.splitlines())
    if len(aristas) != len(pares) - 1 or any(not 1 <= n <= len(pares) for arista in aristas for n in arista):
        raise ValueError("Las conexiones del prompt no forman un árbol válido")

    return Instancia(tuple(int(valor) for _, valor in pares), aristas, int(coincidencia["costo"]))


def generar_prompt(instancia: Instancia) -> str:
    n = instancia.cantidad_nodos
    valores = "\n".join(f"{i}: {valor}" for i, valor in enumerate(instancia.valores, 1))
    aristas = "\n".join(f"{a}-{b}" for a, b in instancia.aristas) or "(none)"

    return f"""Choose a set of nodes of the tree below to obtain the highest possible score.

The tree:
- It has {n} nodes, numbered from 1 to {n}, and {n - 1} edges. It is connected and has no cycles.
- Edges are bidirectional: "a-b" means that a is a neighbor of b and b is a neighbor of a.
- Each edge is listed exactly once. There are no duplicate edges, and no edge joins a node to itself.
- Node values are integers and can be positive, zero, or negative.
- The cost c is the same for every edge.

Node values (node: value):
{valores}

Edges:
{aristas}

Cost c = {instancia.costo}

Scoring rules:
- You may choose any set of nodes: none, some, or all of them. Two neighbors may both be chosen.
- Each chosen node contributes its value minus c for each of its neighbors that is also chosen.
- Only direct neighbors matter. Nodes further away do not change a node's contribution.
- So each edge with both of its nodes chosen lowers the total score by 2c, because each of the two nodes pays c.
- Nodes that are not chosen contribute 0 and pay no cost.
- The score is the sum of the contributions of the chosen nodes. Choosing no nodes scores 0.

Example with a different tree, only to show how the score works. It has 3 nodes with values 5, 6, and 5, edges 1-2 and 2-3, and cost c = 2:
- Choosing nodes 1 and 3 scores 5 + 5 = 10, because they are not neighbors.
- Choosing nodes 1, 2, and 3 scores (5 - 2) + (6 - 2 - 2) + (5 - 2) = 8.
- Choosing node 2 scores 6.
- The best choice is nodes 1 and 3, so the answer is <respuesta>1 3</respuesta>.

Answer format:
- End your reply with exactly one answer block that lists the chosen node numbers, separated by spaces.
- Use only node numbers from 1 to {n}, each at most once. Do not write anything else inside the block.
- If the best choice is no nodes, write <respuesta></respuesta>."""
