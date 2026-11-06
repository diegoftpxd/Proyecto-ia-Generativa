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
_SEPARADOR = r"(?:\s*[,;]\s*|\s+)(?:(?:[ye]|and)\s+)?"
PATRON_NODOS = re.compile(
    rf"\s*(?:nodos?\s*:?\s*)?[\[{{(]?\s*(?:\d{{1,6}}(?:{_SEPARADOR}\d{{1,6}})*)?\s*[\]}})]?\s*\.?\s*",
    re.ASCII | re.IGNORECASE,
)
PATRON_NINGUNO = re.compile(r"\s*(?:ningun[oa]|vac[ií]o|none)\s*\.?\s*", re.IGNORECASE)
PATRON_SEGMENTO = re.compile(r">([^<>]*)(?=<)")  # El texto entre un ">" y el "<" siguiente


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


def _es_lista(texto: str) -> bool:
    return re.search(r"\d", texto, re.ASCII) is not None and PATRON_NODOS.fullmatch(texto) is not None


def _normalizar(texto: str) -> str | None:
    # Un texto demasiado largo no es una lista de nodos, y revisarlo volvería lento al parser.
    return " ".join(texto.split()) if len(texto) <= LARGO_MAXIMO_BLOQUE else None


def _ultima_lista(respuesta: str) -> str:
    # 1) Lo último escrito entre ">" y "<" que sea una lista de nodos o "ninguno". Así se aceptan
    #    cierres mal escritos, como "</ respuesta>", y se ignora el texto que rodea las etiquetas.
    texto = respuesta.lower()
    elegido = None
    for segmento in PATRON_SEGMENTO.finditer(texto):
        contenido = _normalizar(segmento[1])
        if contenido is None:
            continue
        # Un bloque <respuesta></respuesta> vacío es la selección vacía que pide el prompt.
        vacio = not contenido and texto.endswith(APERTURA, 0, segmento.start() + 1) and texto.startswith(CIERRE, segmento.end())
        if vacio or _es_lista(contenido) or PATRON_NINGUNO.fullmatch(contenido):
            elegido = contenido
    if elegido is not None:
        return elegido

    # 2) Una respuesta que solo tiene números de nodos, sin etiquetas.
    completa = _normalizar(respuesta)
    if completa is not None and _es_lista(completa):
        return completa
    raise ErrorFormato("No hay números de nodos entre > y <, ni una respuesta que solo tenga números")


def parsear_respuesta(instancia: Instancia, respuesta: str) -> set[int]:
    bloque = _ultima_lista(respuesta)
    if not bloque or PATRON_NINGUNO.fullmatch(bloque) is not None:
        return set()

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
