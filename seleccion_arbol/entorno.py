import random
import re
from dataclasses import dataclass

from generador import Nivel, generar_instancia
from instancia import Instancia
from modelo import ModeloLLM, ModeloLMStudio
from prompt import generar_prompt, leer_prompt

RECOMPENSA_FORMATO_INVALIDO = -1.0
RECOMPENSA_REFERENCIA = 0.5  # Igualar el mejor atajo vale esto; el resto se gana al superarlo

APERTURA, CIERRE = "<respuesta>", "</respuesta>"
LARGO_MAXIMO_BLOQUE = 2000  # Sobra para cualquier nivel. Evita que un texto enorme vuelva lento al parser
_SEPARADOR = r"(?:\s*[,;]\s*|\s+)(?:[ye]\s+)?"
PATRON_NODOS = re.compile(
    rf"\s*(?:nodos?\s*:?\s*)?[\[{{(]?\s*(?:\d{{1,6}}(?:{_SEPARADOR}\d{{1,6}})*)?\s*[\]}})]?\s*\.?\s*",
    re.ASCII | re.IGNORECASE,
)
PATRON_NINGUNO = re.compile(r"\s*(?:ningun[oa]|vac[ií]o|none)\s*\.?\s*", re.IGNORECASE)


class ErrorFormato(ValueError):
    pass


@dataclass(frozen=True)
class Resultado:
    recompensa: float
    puntaje: int | None
    puntaje_voraz: int
    puntaje_referencia: int
    puntaje_optimo: int
    error: str | None


def _ultimo_bloque(respuesta: str) -> str:
    # Se usa el último bloque porque el modelo puede razonar antes de responder.
    texto = respuesta.lower()
    fin = texto.rfind(CIERRE)
    inicio = texto.rfind(APERTURA, 0, fin) if fin != -1 else -1
    if inicio == -1:
        raise ErrorFormato("Falta el bloque <respuesta>...</respuesta>")
    # Una apertura previa sin cerrar es un bloque anidado: no se sabe cuál es la respuesta.
    if texto.rfind(APERTURA, 0, inicio) > texto.rfind(CIERRE, 0, inicio):
        raise ErrorFormato("El bloque <respuesta> no debe estar anidado")
    bloque = texto[inicio + len(APERTURA):fin]
    if len(bloque) > LARGO_MAXIMO_BLOQUE:
        raise ErrorFormato("El bloque <respuesta> es demasiado largo")
    return " ".join(bloque.split())


def parsear_respuesta(instancia: Instancia, respuesta: str) -> set[int]:
    bloque = _ultimo_bloque(respuesta)
    if PATRON_NINGUNO.fullmatch(bloque) is not None:
        return set()
    if PATRON_NODOS.fullmatch(bloque) is None:
        raise ErrorFormato("El bloque <respuesta> solo debe contener números de nodos")

    nodos = [int(nodo) for nodo in re.findall(r"\d+", bloque, re.ASCII)]
    if len(nodos) != len(set(nodos)):
        raise ErrorFormato("Hay nodos repetidos")
    if any(not 1 <= nodo <= instancia.cantidad_nodos for nodo in nodos):
        raise ErrorFormato(f"Los nodos deben estar entre 1 y {instancia.cantidad_nodos}")
    return set(nodos)


def calcular_recompensa(puntaje: int, referencia: int, optimo: int) -> float:
    if optimo <= 0:
        return float(puntaje == 0)
    # Si no hay nada que superar, o la referencia no aporta, la recompensa es proporcional al óptimo.
    if referencia <= 0 or referencia == optimo:
        return max(0, puntaje) / optimo
    # Peor que la referencia: de 0 a RECOMPENSA_REFERENCIA. Mejor que ella: de RECOMPENSA_REFERENCIA a 1.
    if puntaje <= referencia:
        return RECOMPENSA_REFERENCIA * max(0, puntaje) / referencia
    return RECOMPENSA_REFERENCIA + (1 - RECOMPENSA_REFERENCIA) * (puntaje - referencia) / (optimo - referencia)


def evaluar_respuesta(instancia: Instancia, respuesta: str) -> Resultado:
    voraz = instancia.puntaje_voraz()
    referencia = instancia.puntaje_referencia()
    optimo = instancia.puntaje_optimo()

    try:
        seleccion = parsear_respuesta(instancia, respuesta)
    except ErrorFormato as error:
        return Resultado(RECOMPENSA_FORMATO_INVALIDO, None, voraz, referencia, optimo, str(error))

    puntaje = instancia.puntaje(seleccion)
    return Resultado(calcular_recompensa(puntaje, referencia, optimo), puntaje, voraz, referencia, optimo, None)


def reward(prompt: str, completion: str) -> float:
    # La interfaz común del curso: basta con el prompt y lo que respondió el modelo.
    return evaluar_respuesta(leer_prompt(prompt), completion).recompensa


class EntornoSeleccion:
    # Cada episodio tiene un solo paso: el modelo entrega la selección completa.
    def __init__(self, semilla: int | None = None):
        self._rng = random.Random(semilla)
        self.instancia: Instancia | None = None

    def reset(self, nivel: Nivel) -> str:
        self.instancia = generar_instancia(nivel, self._rng)
        return generar_prompt(self.instancia)

    def step(self, respuesta: str) -> Resultado:
        if self.instancia is None:
            raise RuntimeError("Hay que llamar a reset antes de step")

        instancia, self.instancia = self.instancia, None
        return evaluar_respuesta(instancia, respuesta)


def jugar_episodio(entorno: EntornoSeleccion, modelo: ModeloLLM, nivel: Nivel) -> Resultado:
    prompt = entorno.reset(nivel)
    return entorno.step(modelo.generar(prompt))


if __name__ == "__main__":
    from generador import NIVELES

    print(jugar_episodio(EntornoSeleccion(), ModeloLMStudio(), NIVELES[0]))
