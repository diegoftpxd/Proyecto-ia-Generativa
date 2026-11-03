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


def _episodios_hechos(ruta) -> set[tuple]:
    return {
        (fila["modelo"], fila["nivel"], fila["episodio"], fila["semilla"], fila.get("intento", 1))
        for fila in leer_resultados(ruta)
    }


def _evaluar_episodio(
    modelo: ModeloLLM, nombre: str, instancia: Instancia, nivel: int, episodio, semilla, ruta, intento: int = 1
) -> None:
    inicio = time.perf_counter()
    try:
        respuesta = modelo.generar(generar_prompt(instancia))
    except RuntimeError as error:
        # Es una falla de infraestructura: no se guarda, para reintentarla al reanudar.
        print(f"{nombre} nivel {nivel} ep {episodio} intento {intento}: falló la consulta: {error}")
        return

    resultado = evaluar_respuesta(instancia, respuesta)
    fila = {
        "modelo": nombre,
        "nivel": nivel,
        "episodio": episodio,
        "semilla": semilla,
        "intento": intento,
        "nodos": instancia.cantidad_nodos,
        "recompensa": resultado.recompensa,
        "puntaje": resultado.puntaje,
        "puntaje_voraz": resultado.puntaje_voraz,
        "puntaje_referencia": resultado.puntaje_referencia,
        "puntaje_optimo": resultado.puntaje_optimo,
        "puntaje_sin_vecinos": instancia.puntaje_sin_vecinos(),
        "error": resultado.error,
        "segundos": round(time.perf_counter() - inicio, 1),
        "tokens": getattr(modelo, "ultimo_tokens", None),
        "fin": getattr(modelo, "ultimo_fin", None),
        "respuesta": respuesta,
        "texto_completo": getattr(modelo, "ultimo_texto", None),
        "respuesta_forzada": getattr(modelo, "ultimo_respuesta_forzada", None),
    }
    with Path(ruta).open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(fila, ensure_ascii=False) + "\n")
    print(
        f"{nombre} nivel {nivel} ep {episodio} intento {intento}: recompensa={resultado.recompensa:.3f} "
        f"puntaje={resultado.puntaje} óptimo={resultado.puntaje_optimo} "
        f"{fila['segundos']}s {fila['tokens']}tok"
    )


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
    hechos = _episodios_hechos(ruta)

    for nivel in niveles:
        for episodio in range(1, episodios + 1):
            if (nombre, nivel, episodio, semilla, 1) not in hechos:
                instancia = instancia_del_episodio(nivel, episodio, semilla)
                _evaluar_episodio(modelo, nombre, instancia, nivel, episodio, semilla, ruta)


def evaluar_dataset(modelo: ModeloLLM, nombre: str, ruta_datos, ruta, niveles=None, intentos: int = 1) -> None:
    # Igual que evaluar_modelo, pero con los árboles de un archivo. En orden de nivel, para ver avances parciales.
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    hechos = _episodios_hechos(ruta)

    ejemplos = sorted(leer_resultados(ruta_datos), key=lambda e: e["nivel"])
    for e in ejemplos:
        if niveles is not None and e["nivel"] not in niveles:
            continue
        instancia = Instancia(tuple(e["valores"]), tuple(map(tuple, e["aristas"])), e["costo"])
        for intento in range(1, intentos + 1):
            if (nombre, e["nivel"], e["id"], None, intento) not in hechos:
                _evaluar_episodio(modelo, nombre, instancia, e["nivel"], e["id"], None, ruta, intento)


def resumir(ruta, por=("modelo", "nivel")):
    import pandas as pd

    tabla = pd.DataFrame(leer_resultados(ruta))
    if tabla.empty:
        return tabla
    # Los resultados guardados con versiones anteriores no traen estas columnas.
    tabla = tabla.reindex(columns=tabla.columns.union(["fin", "puntaje_sin_vecinos", "puntaje_referencia"], sort=False))
    tabla["optimo"] = tabla["recompensa"] == 1.0
    tabla["invalido"] = tabla["recompensa"] == -1.0
    tabla["corte"] = tabla["fin"] == "length"
    tabla["supera_voraz"] = tabla["puntaje"].notna() & (tabla["puntaje"] > tabla["puntaje_voraz"])
    tabla["supera_atajos"] = tabla["puntaje"].notna() & (tabla["puntaje"] > tabla["puntaje_referencia"])
    tabla["optimo_atajo"] = (tabla["puntaje_sin_vecinos"] == tabla["puntaje_optimo"]).where(
        tabla["puntaje_sin_vecinos"].notna()
    ).astype(float)
    return tabla.groupby(list(por)).agg(
        episodios=("recompensa", "size"),
        recompensa=("recompensa", "mean"),
        optimos=("optimo", "mean"),
        optimos_atajo=("optimo_atajo", "mean"),
        supera_voraz=("supera_voraz", "mean"),
        supera_atajos=("supera_atajos", "mean"),
        invalidos=("invalido", "mean"),
        cortes=("corte", "mean"),
        segundos=("segundos", "mean"),
        tokens=("tokens", "mean"),
    ).round(3)


def resumir_pass(ruta, k: int = 3, por=("modelo", "nivel")):
    # pass@1: intentos que logran el óptimo. pass@k: árboles con k intentos donde al menos uno lo logra.
    import pandas as pd

    tabla = pd.DataFrame(leer_resultados(ruta))
    if tabla.empty:
        return tabla
    por = list(por)
    tabla["paso"] = tabla["recompensa"] == 1.0
    arboles = tabla[tabla["intento"] <= k].groupby(["modelo", "nivel", "episodio"], as_index=False).agg(
        intentos=("paso", "size"),
        paso=("paso", "max"),
    )
    completos = arboles[arboles["intentos"] == k].astype({"paso": float})

    resumen = tabla.groupby(por).agg(intentos=("paso", "size"), **{"pass@1": ("paso", "mean")})
    resumen["arboles_completos"] = completos.groupby(por).size().reindex(resumen.index, fill_value=0)
    resumen[f"pass@{k}"] = completos.groupby(por)["paso"].mean().reindex(resumen.index)
    return resumen.round(3)
