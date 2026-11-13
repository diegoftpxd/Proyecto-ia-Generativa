"""Genera las trazas de SFT y los pares de DPO con un modelo profesor.

Uso, desde seleccion_arbol/:

    python trazas.py resultados/03_trazas_qwen95 --verificar
    python trazas.py resultados/03_trazas_qwen95 --niveles 2 --max-problemas 10
    python trazas.py resultados/03_trazas_qwen95
    python trazas.py resultados/03_trazas_qwen95 --solo-construir

La primera etapa muestrea el modelo contra los problemas de datos/entrenamiento.jsonl
y guarda todo en resultados/<carpeta>/muestras.jsonl. La segunda filtra esas muestras
y escribe datos/sft.jsonl y datos/dpo.jsonl. Las etapas son independientes:
--solo-construir rehace los datasets sin volver a llamar al modelo.

El muestreo es secuencial hasta el primer éxito: por problema se generan hasta
`muestras_por_problema` respuestas y se corta al obtener una correcta. Los intentos
fallidos del camino son los rechazos que después usa DPO.

Criterios:
- Una traza es aceptable si recompensa == 1.0, no fue forzada y no se cortó.
- Un rechazo sirve para DPO si su recompensa está entre `rejected_minimo` y
  `rejected_maximo`, no fue forzado y no se cortó.
"""
import argparse
import hashlib
import json
import random
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from entorno import evaluar_respuesta
from evaluacion import leer_resultados
from instancia import Instancia
from modelo import ModeloLMStudio

COLUMNAS_RESUMEN = [
    "nivel", "problemas", "muestras", "correctas", "aceptables", "yield", "yield_aceptable",
    "invalidos", "cortes", "trazas_sft", "pares_dpo", "tokens_medio", "segundos_medio",
]


def a_instancia(ejemplo: dict) -> Instancia:
    # Misma conversión que usa evaluacion.evaluar_dataset.
    return Instancia(tuple(ejemplo["valores"]), tuple(map(tuple, ejemplo["aristas"])), ejemplo["costo"])


def _sin_corte(muestra: dict) -> bool:
    return not muestra.get("forzada") and muestra.get("fin") != "length"


def aceptable(muestra: dict) -> bool:
    return muestra["recompensa"] == 1.0 and _sin_corte(muestra)


def rechazo_util(muestra: dict, minimo: float, maximo: float) -> bool:
    return minimo <= muestra["recompensa"] <= maximo and _sin_corte(muestra)


def texto_de(muestra: dict) -> str:
    # Se prefiere el texto completo (razonamiento + respuesta); si no trae el bloque,
    # se usa lo que devolvió modelo.generar, que sí fue calificado.
    texto = muestra.get("texto_completo") or ""
    if "</respuesta>" not in texto:
        texto = muestra.get("respuesta") or ""
    return texto


def leer_muestras(ruta) -> list[dict]:
    ruta = Path(ruta)
    if not ruta.exists():
        return []
    return [json.loads(linea) for linea in ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]


def cuota_de(configuracion: dict, nivel: int) -> int:
    # Permite repartir el objetivo con cuotas_por_nivel; si no, todos los niveles
    # usan la misma cuota_por_nivel.
    cuotas = configuracion.get("cuotas_por_nivel")
    if cuotas:
        return int(cuotas.get(str(nivel), cuotas.get(nivel, 0)))
    return int(configuracion["cuota_por_nivel"])


def guardar_jsonl(ejemplos: list[dict], ruta) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="\n") as archivo:
        for ejemplo in ejemplos:
            archivo.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")


def verificar_servidor(nombre_modelo: str, url_base: str) -> None:
    try:
        with urllib.request.urlopen(f"{url_base}/models", timeout=15) as respuesta:
            disponibles = [modelo["id"] for modelo in json.load(respuesta).get("data", [])]
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"No se pudo consultar {url_base}/models: {error}\n"
            "Revisa que el servidor de LM Studio esté iniciado y el puerto sea el correcto."
        )
    if nombre_modelo not in disponibles:
        raise SystemExit(
            f"El modelo '{nombre_modelo}' no está en {url_base}.\n"
            f"Disponibles: {disponibles}"
        )


def verificar_datos(configuracion: dict) -> None:
    esperado = configuracion.get("datos_sha256")
    if not esperado:
        return
    ruta = Path(configuracion["ruta_datos"])
    actual = hashlib.sha256(ruta.read_bytes()).hexdigest()
    if actual != esperado:
        print(
            f"AVISO: {ruta} no coincide con datos_sha256 de configuracion.json.\n"
            f"  esperado: {esperado}\n  actual:   {actual}\n"
            "Las trazas no serán comparables con las corridas anteriores."
        )


def mostrar_plan(problemas: list[dict], muestras: list[dict], niveles: list[int], configuracion: dict) -> None:
    print(f"modelo {configuracion['modelo_lm_studio']} en {configuracion['url_base']}")
    cuotas = ", ".join(f"nivel {nivel}: {cuota_de(configuracion, nivel)}" for nivel in niveles)
    print(f"{configuracion['muestras_por_problema']} muestras por problema como máximo; {cuotas}")
    for nivel in niveles:
        disponibles = sum(1 for ejemplo in problemas if ejemplo["nivel"] == nivel)
        aceptadas = len({
            muestra["problema"] for muestra in muestras
            if muestra["nivel"] == nivel and aceptable(muestra)
        })
        faltan = max(0, cuota_de(configuracion, nivel) - aceptadas)
        print(f"  nivel {nivel}: {disponibles} problemas, {aceptadas} aceptadas, faltan {faltan}")


def generar_muestras(
    modelo: ModeloLMStudio,
    carpeta: Path,
    configuracion: dict,
    niveles: list[int],
    max_problemas: int | None = None,
) -> None:
    ruta_datos = Path(configuracion["ruta_datos"])
    ruta_muestras = carpeta / "muestras.jsonl"
    k = int(configuracion["muestras_por_problema"])
    paciencia = int(configuracion.get("paciencia_por_nivel", 60))
    semilla = configuracion.get("semilla", 0)

    problemas = leer_resultados(ruta_datos)
    existentes = leer_muestras(ruta_muestras)
    claves = {(muestra["problema"], muestra["muestra"]) for muestra in existentes}
    aceptados = {
        (muestra["nivel"], muestra["problema"]) for muestra in existentes if aceptable(muestra)
    }
    aceptadas = defaultdict(int)
    for nivel, _ in aceptados:
        aceptadas[nivel] += 1

    for nivel in niveles:
        cuota = cuota_de(configuracion, nivel)
        disponibles = [ejemplo for ejemplo in problemas if ejemplo["nivel"] == nivel]
        random.Random(f"trazas-{semilla}-{nivel}").shuffle(disponibles)
        if max_problemas:
            disponibles = disponibles[:max_problemas]

        print(
            f"\nnivel {nivel}: {len(disponibles)} problemas, "
            f"{aceptadas[nivel]} trazas aceptadas de {cuota}",
            flush=True,
        )
        if aceptadas[nivel] >= cuota:
            print(f"nivel {nivel}: cuota ya cumplida")
            continue

        sin_exito = 0
        for ejemplo in disponibles:
            if aceptadas[nivel] >= cuota:
                break

            problema = ejemplo["id"]
            if (nivel, problema) in aceptados:
                continue

            instancia = a_instancia(ejemplo)
            exito = False
            for muestra in range(1, k + 1):
                if (problema, muestra) in claves:
                    continue

                inicio = time.perf_counter()
                try:
                    respuesta = modelo.generar(ejemplo["prompt"])
                except RuntimeError as error:
                    # Falla de infraestructura: no se guarda, para reintentarla al reanudar.
                    print(f"nivel {nivel} {problema} muestra {muestra}: falló la consulta: {error}")
                    continue
                segundos = round(time.perf_counter() - inicio, 1)
                resultado = evaluar_respuesta(instancia, respuesta)

                fila = {
                    "problema": problema,
                    "nivel": nivel,
                    "muestra": muestra,
                    "modelo": modelo.nombre,
                    "temperatura": modelo.temperatura,
                    "recompensa": resultado.recompensa,
                    "puntaje": resultado.puntaje,
                    "puntaje_optimo": resultado.puntaje_optimo,
                    "tokens": modelo.ultimo_tokens,
                    "fin": modelo.ultimo_fin,
                    "forzada": modelo.ultimo_respuesta_forzada,
                    "segundos": segundos,
                    "error": resultado.error,
                    "prompt": ejemplo["prompt"],
                    "respuesta": respuesta,
                    "texto_completo": modelo.ultimo_texto,
                }
                with ruta_muestras.open("a", encoding="utf-8", newline="\n") as archivo:
                    archivo.write(json.dumps(fila, ensure_ascii=False) + "\n")
                claves.add((problema, muestra))
                existentes.append(fila)

                print(
                    f"nivel {nivel} {problema} muestra {muestra}: "
                    f"recompensa={resultado.recompensa:.3f} {segundos:.0f}s "
                    f"tok={modelo.ultimo_tokens} fin={modelo.ultimo_fin} "
                    f"forzada={modelo.ultimo_respuesta_forzada}",
                    flush=True,
                )

                if aceptable(fila):
                    aceptados.add((nivel, problema))
                    aceptadas[nivel] += 1
                    exito = True
                    break

            if exito:
                sin_exito = 0
            else:
                sin_exito += 1
                if sin_exito >= paciencia:
                    print(
                        f"nivel {nivel}: {paciencia} problemas seguidos sin éxito; "
                        f"se corta el nivel con {aceptadas[nivel]} trazas",
                        flush=True,
                    )
                    break


def construir_sft(
    muestras: list[dict],
    niveles: list[int],
    cuota_por_nivel: int,
    cuotas_por_nivel: dict | None = None,
) -> list[dict]:
    # Una traza por problema, la primera respuesta aceptable. Así el SFT ve la mayor
    # variedad de problemas posible con el mismo presupuesto.
    elegidas = []
    por_nivel = defaultdict(list)
    for muestra in muestras:
        if aceptable(muestra):
            por_nivel[muestra["nivel"]].append(muestra)

    for nivel in niveles:
        cuota = cuota_por_nivel
        if cuotas_por_nivel:
            cuota = int(cuotas_por_nivel.get(str(nivel), cuotas_por_nivel.get(nivel, cuota_por_nivel)))
        vistos = set()
        for muestra in sorted(por_nivel[nivel], key=lambda m: (m["problema"], m["muestra"])):
            if muestra["problema"] in vistos:
                continue
            vistos.add(muestra["problema"])
            elegidas.append({
                "id": f"sft-{muestra['problema']}",
                "problema": muestra["problema"],
                "nivel": nivel,
                "modelo": muestra["modelo"],
                "recompensa": muestra["recompensa"],
                "tokens": muestra["tokens"],
                "messages": [
                    {"role": "user", "content": muestra["prompt"]},
                    {"role": "assistant", "content": texto_de(muestra)},
                ],
            })
            if len(vistos) >= cuota:
                break
    return elegidas


def _pareja_dpo(chosen: dict, rejected: dict) -> dict:
    return {
        "id": f"dpo-{chosen['problema']}-{rejected['muestra']}",
        "problema": chosen["problema"],
        "nivel": chosen["nivel"],
        "prompt": chosen["prompt"],
        "chosen": texto_de(chosen),
        "rejected": texto_de(rejected),
        "recompensa_chosen": chosen["recompensa"],
        "recompensa_rejected": rejected["recompensa"],
    }


def construir_dpo(
    muestras: list[dict],
    niveles: list[int],
    objetivo_pares: int,
    pares_por_problema: int,
    rejected_minimo: float,
    rejected_maximo: float,
) -> list[dict]:
    por_nivel = defaultdict(lambda: defaultdict(list))
    for muestra in muestras:
        por_nivel[muestra["nivel"]][muestra["problema"]].append(muestra)

    def candidatos(nivel: int) -> list[dict]:
        salida = []
        for problema in sorted(por_nivel[nivel]):
            grupo = sorted(por_nivel[nivel][problema], key=lambda m: m["muestra"])
            correctas = [muestra for muestra in grupo if aceptable(muestra)]
            if not correctas:
                continue
            rechazos = [
                muestra for muestra in grupo
                if rechazo_util(muestra, rejected_minimo, rejected_maximo)
            ]
            vistos = set()
            for rechazo in rechazos:
                texto = texto_de(rechazo)
                if not texto or texto in vistos:
                    continue
                vistos.add(texto)
                salida.append(_pareja_dpo(correctas[0], rechazo))
                if len([p for p in salida if p["problema"] == problema]) >= pares_por_problema:
                    break
        return salida

    # Reparto parejo por nivel; si un nivel no alcanza, se rellena con los demás.
    cuota_nivel = max(1, objetivo_pares // max(1, len(niveles)))
    pares = []
    for nivel in niveles:
        pares.extend(candidatos(nivel)[:cuota_nivel])
    faltan = objetivo_pares - len(pares)
    if faltan > 0:
        for nivel in niveles:
            extra = candidatos(nivel)[cuota_nivel:cuota_nivel + faltan]
            pares.extend(extra)
            faltan -= len(extra)
            if faltan <= 0:
                break
    return pares[:objetivo_pares]


def construir_solver(ruta_datos, niveles: list[int]) -> list[dict]:
    # Plus: trazas correctas por construcción, narradas por la programación dinámica.
    ejemplos = []
    for ejemplo in leer_resultados(ruta_datos):
        if ejemplo["nivel"] not in niveles:
            continue
        ejemplos.append({
            "id": f"solver-{ejemplo['id']}",
            "problema": ejemplo["id"],
            "nivel": ejemplo["nivel"],
            "modelo": "programacion dinamica",
            "recompensa": 1.0,
            "messages": [
                {"role": "user", "content": ejemplo["prompt"]},
                {"role": "assistant", "content": f"{ejemplo['razonamiento']}\n{ejemplo['respuesta']}"},
            ],
        })
    return ejemplos


def resumir_generacion(
    muestras: list[dict],
    sft: list[dict],
    dpo: list[dict],
    niveles: list[int],
):
    import pandas as pd

    filas = []
    for nivel in niveles:
        del_nivel = [muestra for muestra in muestras if muestra["nivel"] == nivel]
        if not del_nivel:
            continue
        n = len(del_nivel)
        filas.append({
            "nivel": nivel,
            "problemas": len({muestra["problema"] for muestra in del_nivel}),
            "muestras": n,
            "correctas": sum(muestra["recompensa"] == 1.0 for muestra in del_nivel),
            "aceptables": sum(aceptable(muestra) for muestra in del_nivel),
            "yield": round(sum(muestra["recompensa"] == 1.0 for muestra in del_nivel) / n, 3),
            "yield_aceptable": round(sum(aceptable(muestra) for muestra in del_nivel) / n, 3),
            "invalidos": round(sum(muestra["recompensa"] == -1.0 for muestra in del_nivel) / n, 3),
            "cortes": round(sum(muestra["fin"] == "length" for muestra in del_nivel) / n, 3),
            "trazas_sft": sum(1 for ejemplo in sft if ejemplo["nivel"] == nivel),
            "pares_dpo": sum(1 for pareja in dpo if pareja["nivel"] == nivel),
            "tokens_medio": round(sum((muestra["tokens"] or 0) for muestra in del_nivel) / n),
            "segundos_medio": round(sum(muestra["segundos"] for muestra in del_nivel) / n, 1),
        })

    if muestras:
        n = len(muestras)
        filas.append({
            "nivel": "total",
            "problemas": len({muestra["problema"] for muestra in muestras}),
            "muestras": n,
            "correctas": sum(muestra["recompensa"] == 1.0 for muestra in muestras),
            "aceptables": sum(aceptable(muestra) for muestra in muestras),
            "yield": round(sum(muestra["recompensa"] == 1.0 for muestra in muestras) / n, 3),
            "yield_aceptable": round(sum(aceptable(muestra) for muestra in muestras) / n, 3),
            "invalidos": round(sum(muestra["recompensa"] == -1.0 for muestra in muestras) / n, 3),
            "cortes": round(sum(muestra["fin"] == "length" for muestra in muestras) / n, 3),
            "trazas_sft": len(sft),
            "pares_dpo": len(dpo),
            "tokens_medio": round(sum((muestra["tokens"] or 0) for muestra in muestras) / n),
            "segundos_medio": round(sum(muestra["segundos"] for muestra in muestras) / n, 1),
        })

    return pd.DataFrame(filas, columns=COLUMNAS_RESUMEN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("carpeta", help="carpeta de la corrida, por ejemplo resultados/03_trazas_qwen95")
    parser.add_argument("--niveles", help="niveles separados por coma; por defecto los de configuracion.json")
    parser.add_argument("--max-problemas", type=int, help="tope de problemas por nivel, para una prueba corta")
    parser.add_argument("--solo-construir", action="store_true", help="rehace SFT/DPO desde muestras.jsonl sin llamar al modelo")
    parser.add_argument("--verificar", action="store_true", help="solo comprueba servidor y plan; no genera nada")
    argumentos = parser.parse_args()

    carpeta = Path(argumentos.carpeta)
    configuracion = json.loads((carpeta / "configuracion.json").read_text(encoding="utf-8"))
    niveles = (
        [int(n) for n in argumentos.niveles.split(",")]
        if argumentos.niveles
        else configuracion["niveles"]
    )

    ruta_muestras = carpeta / "muestras.jsonl"

    if not argumentos.solo_construir:
        verificar_servidor(configuracion["modelo_lm_studio"], configuracion["url_base"])
        verificar_datos(configuracion)
        problemas = leer_resultados(Path(configuracion["ruta_datos"]))
        mostrar_plan(problemas, leer_muestras(ruta_muestras), niveles, configuracion)
        if argumentos.verificar:
            print("solo verificación: no se generó ninguna muestra")
            return

        modelo = ModeloLMStudio(
            nombre=configuracion["modelo_lm_studio"],
            url_base=configuracion["url_base"],
            temperatura=configuracion.get("temperatura", 0.8),
            tokens_pensamiento=configuracion["tokens_pensamiento"],
            tokens_respuesta=configuracion["tokens_respuesta"],
            segundos_limite=configuracion.get("segundos_limite", 300),
        )
        inicio = time.perf_counter()
        generar_muestras(modelo, carpeta, configuracion, niveles, argumentos.max_problemas)
        print(f"\nmuestreo terminado en {time.perf_counter() - inicio:.0f}s")

    muestras = leer_muestras(ruta_muestras)
    if not muestras:
        raise SystemExit(f"No hay muestras en {ruta_muestras}")

    sft = construir_sft(
        muestras,
        niveles,
        int(configuracion.get("cuota_por_nivel", 0)),
        configuracion.get("cuotas_por_nivel"),
    )
    dpo = construir_dpo(
        muestras,
        niveles,
        int(configuracion["objetivo_pares"]),
        int(configuracion.get("pares_por_problema", 2)),
        float(configuracion.get("rejected_minimo", 0.2)),
        float(configuracion.get("rejected_maximo", 0.9)),
    )

    guardar_jsonl(sft, configuracion["ruta_sft"])
    guardar_jsonl(dpo, configuracion["ruta_dpo"])
    print(f"\n{len(sft)} trazas SFT en {configuracion['ruta_sft']}")
    print(f"{len(dpo)} pares DPO en {configuracion['ruta_dpo']}")

    if configuracion.get("incluir_solver", True):
        niveles_solver = configuracion.get("niveles_solver", niveles)
        solver = construir_solver(configuracion["ruta_datos"], niveles_solver)
        guardar_jsonl(solver, configuracion["ruta_solver"])
        print(f"{len(solver)} trazas del solver en {configuracion['ruta_solver']} (plus)")

    resumen = resumir_generacion(muestras, sft, dpo, niveles)
    resumen.to_csv(carpeta / "generacion.csv", index=False)
    print()
    print(resumen.to_string(index=False))
    print(f"\nescrito {carpeta / 'generacion.csv'}")


if __name__ == "__main__":
    main()
