import json
import random
from pathlib import Path

from generador import NIVELES, Nivel, generar_instancia
from instancia import Instancia
from prompt import generar_prompt

ENTRENAMIENTO_POR_NIVEL = 800
EVALUACION_POR_NIVEL = 20
TEST_POR_NIVEL = 20
INTENTOS_POR_EJEMPLO = 50  # Los niveles chicos tienen pocos árboles distintos y se agotan antes
CARPETA = Path(__file__).parent / "datos"


def resolver(instancia: Instancia) -> tuple[set[int], str]:
    # Misma programación dinámica del verificador, pero recuerda qué nodos elegir y lo explica.
    vecinos = instancia.vecinos()
    costo_arista = 2 * instancia.costo
    padre = {1: None}
    orden = [1]
    for nodo in orden:
        for vecino in vecinos[nodo]:
            if vecino != padre[nodo]:
                padre[vecino] = nodo
                orden.append(vecino)

    sin_elegir, elegido = {}, {}
    pasos = [
        "Tomo el nodo 1 como raíz y voy de las hojas a la raíz. Para cada nodo calculo su mejor "
        f"puntaje sin elegirlo (N) y eligiéndolo (E). Elegir un nodo y su hijo resta 2c = {costo_arista}."
    ]
    for nodo in reversed(orden):
        hijos = [vecino for vecino in vecinos[nodo] if vecino != padre[nodo]]
        sin_elegir[nodo] = sum(max(sin_elegir[h], elegido[h]) for h in hijos)
        elegido[nodo] = instancia.valores[nodo - 1] + sum(
            max(sin_elegir[h], elegido[h] - costo_arista) for h in hijos
        )
        if hijos:
            detalle = ", ".join(f"{h}: N={sin_elegir[h]} E={elegido[h]}" for h in hijos)
            pasos.append(f"Nodo {nodo} (hijos {detalle}): N={sin_elegir[nodo]}, E={elegido[nodo]}.")
        else:
            pasos.append(f"Nodo {nodo} (hoja): N=0, E={elegido[nodo]}.")

    # Se recorre desde la raíz. En un empate se prefiere no elegir, para que el resultado sea único.
    seleccion = set()
    for nodo in orden:
        descuento = costo_arista if padre[nodo] in seleccion else 0
        if elegido[nodo] - descuento > sin_elegir[nodo]:
            seleccion.add(nodo)

    puntaje = max(sin_elegir[1], elegido[1])
    nodos = ", ".join(map(str, sorted(seleccion))) or "ninguno"
    pasos.append(f"El mejor puntaje es {puntaje}. Bajando desde la raíz, los nodos elegidos son: {nodos}.")
    return seleccion, "\n".join(pasos)


def forma_canonica(instancia: Instancia) -> tuple:
    # Es igual para dos árboles que solo difieren en la numeración de sus nodos.
    vecinos = instancia.vecinos()

    def subarbol(nodo, padre):
        hijos = sorted(subarbol(vecino, nodo) for vecino in vecinos[nodo] if vecino != padre)
        return instancia.valores[nodo - 1], tuple(hijos)

    raices = range(1, instancia.cantidad_nodos + 1)
    return instancia.costo, min(subarbol(raiz, None) for raiz in raices)


def crear_ejemplo(instancia: Instancia, nivel: int, id_ejemplo: str) -> dict:
    seleccion, razonamiento = resolver(instancia)
    optimo = instancia.puntaje_optimo()
    if instancia.puntaje(seleccion) != optimo:
        raise RuntimeError(f"La selección no coincide con el verificador: {instancia}")

    return {
        "id": id_ejemplo,
        "nivel": nivel,
        "nodos": instancia.cantidad_nodos,
        "valores": list(instancia.valores),
        "aristas": [list(arista) for arista in instancia.aristas],
        "costo": instancia.costo,
        "prompt": generar_prompt(instancia),
        "razonamiento": razonamiento,
        "respuesta": "<respuesta>" + " ".join(map(str, sorted(seleccion))) + "</respuesta>",
        "puntaje_optimo": optimo,
        "puntaje_voraz": instancia.puntaje_voraz(),
        "puntaje_sin_vecinos": instancia.puntaje_sin_vecinos(),
    }


def instancias_distintas(nivel: Nivel, cantidad: int, rng: random.Random) -> list[Instancia]:
    vistas = {}
    for _ in range(cantidad * INTENTOS_POR_EJEMPLO):
        if len(vistas) == cantidad:
            break
        instancia = generar_instancia(nivel, rng)
        vistas.setdefault(forma_canonica(instancia), instancia)
    return list(vistas.values())


def generar_datasets(
    semilla: int = 0,
    entrenamiento_por_nivel: int = ENTRENAMIENTO_POR_NIVEL,
    evaluacion_por_nivel: int = EVALUACION_POR_NIVEL,
    test_por_nivel: int = TEST_POR_NIVEL,
) -> dict[str, list[dict]]:
    # Se reparte dentro de cada nivel: las tres partes tienen todos los niveles y ningún árbol en común.
    datasets = {"entrenamiento": [], "evaluacion": [], "test": []}
    for numero, nivel in enumerate(NIVELES, 1):
        rng = random.Random(f"dataset-{semilla}-{numero}")
        total = entrenamiento_por_nivel + evaluacion_por_nivel + test_por_nivel
        instancias = instancias_distintas(nivel, total, rng)
        # Si un nivel no tiene árboles suficientes, el faltante sale de entrenamiento.
        fin_entrenamiento = len(instancias) - evaluacion_por_nivel - test_por_nivel
        if fin_entrenamiento < 1:
            raise RuntimeError(f"El nivel {numero} solo tiene {len(instancias)} árboles distintos")
        partes = {
            "entrenamiento": instancias[:fin_entrenamiento],
            "evaluacion": instancias[fin_entrenamiento:fin_entrenamiento + evaluacion_por_nivel],
            "test": instancias[fin_entrenamiento + evaluacion_por_nivel:],
        }
        for nombre, parte in partes.items():
            for indice, instancia in enumerate(parte):
                datasets[nombre].append(crear_ejemplo(instancia, numero, f"{nombre}-{numero}-{indice}"))

    for nombre, ejemplos in datasets.items():
        random.Random(f"orden-{semilla}-{nombre}").shuffle(ejemplos)
    return datasets


def guardar_datasets(datasets: dict[str, list[dict]], carpeta=CARPETA) -> None:
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    for nombre, ejemplos in datasets.items():
        with (carpeta / f"{nombre}.jsonl").open("w", encoding="utf-8", newline="\n") as archivo:
            for ejemplo in ejemplos:
                archivo.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    datasets = generar_datasets()
    guardar_datasets(datasets)
    for nombre, ejemplos in datasets.items():
        print(f"{nombre}: {len(ejemplos)} ejemplos")
