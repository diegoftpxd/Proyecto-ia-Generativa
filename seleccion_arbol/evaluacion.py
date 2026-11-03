import json
import random
import time
from pathlib import Path

from entorno import evaluar_respuesta
from generador import NIVELES, generar_instancia
from instancia import Instancia
from modelo import ModeloLLM
from prompt import generar_prompt


def instancia_del_episodio(nivel: int, episodio: int, semilla: int = 0) -> Instancia:
    # Depende solo de estos datos, así todos los modelos reciben los mismos árboles.
    return generar_instancia(NIVELES[nivel - 1], random.Random(f"{semilla}-{nivel}-{episodio}"))


def leer_resultados(ruta) -> list[dict]:
    ruta = Path(ruta)
    if not ruta.exists():
        return []
    return [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]


def evaluar_modelo(
    modelo: ModeloLLM,
    nombre: str,
    niveles: list[int],
    episodios: int,
    ruta,
    semilla: int = 0,
) -> None:
    # Cada episodio se guarda al terminar, y al repetir la llamada se saltan los ya guardados.
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    hechos = {
        (fila["modelo"], fila["nivel"], fila["episodio"], fila["semilla"])
        for fila in leer_resultados(ruta)
    }

    for nivel in niveles:
        for episodio in range(1, episodios + 1):
            if (nombre, nivel, episodio, semilla) in hechos:
                continue

            instancia = instancia_del_episodio(nivel, episodio, semilla)
            inicio = time.perf_counter()
            try:
                respuesta = modelo.generar(generar_prompt(instancia))
            except RuntimeError as error:
                # Es una falla de infraestructura: no se guarda, para reintentarla al reanudar.
                print(f"{nombre} nivel {nivel} ep {episodio}: falló la consulta: {error}")
                continue

            resultado = evaluar_respuesta(instancia, respuesta)
            fila = {
                "modelo": nombre,
                "nivel": nivel,
                "episodio": episodio,
                "semilla": semilla,
                "nodos": instancia.cantidad_nodos,
                "recompensa": resultado.recompensa,
                "puntaje": resultado.puntaje,
                "puntaje_voraz": resultado.puntaje_voraz,
                "puntaje_optimo": resultado.puntaje_optimo,
                "error": resultado.error,
                "segundos": round(time.perf_counter() - inicio, 1),
                "tokens": getattr(modelo, "ultimo_tokens", None),
                "respuesta": respuesta,
                "texto_completo": getattr(modelo, "ultimo_texto", None),
                "razonamiento_cerrado": getattr(modelo, "ultimo_razonamiento_cerrado", None),
            }
            with ruta.open("a", encoding="utf-8") as archivo:
                archivo.write(json.dumps(fila, ensure_ascii=False) + "\n")
            print(
                f"{nombre} nivel {nivel} ep {episodio}: recompensa={resultado.recompensa:.3f} "
                f"puntaje={resultado.puntaje} óptimo={resultado.puntaje_optimo} "
                f"{fila['segundos']}s {fila['tokens']}tok"
            )


def resumir(ruta):
    import pandas as pd

    tabla = pd.DataFrame(leer_resultados(ruta))
    if tabla.empty:
        return tabla
    tabla["optimo"] = tabla["recompensa"] == 1.0
    tabla["invalido"] = tabla["recompensa"] == -1.0
    tabla["supera_voraz"] = tabla["puntaje"].notna() & (tabla["puntaje"] > tabla["puntaje_voraz"])
    return tabla.groupby(["modelo", "nivel"]).agg(
        episodios=("recompensa", "size"),
        recompensa=("recompensa", "mean"),
        optimos=("optimo", "mean"),
        supera_voraz=("supera_voraz", "mean"),
        invalidos=("invalido", "mean"),
        segundos=("segundos", "mean"),
        tokens=("tokens", "mean"),
    ).round(3)
