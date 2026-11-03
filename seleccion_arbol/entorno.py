import random
import re
from dataclasses import dataclass

from generador import Nivel, generar_instancia
from instancia import Instancia
from modelo import ModeloLLM, ModeloLMStudio
from prompt import generar_prompt

RECOMPENSA_FORMATO_INVALIDO = -1.0
RECOMPENSA_VORAZ = 0.5  # Igualar la regla simple vale esto; el resto se gana al superarla

PATRON_BLOQUE = re.compile(r"<respuesta>(.*?)</respuesta>", re.DOTALL)
PATRON_NODOS = re.compile(r"\s*(?:\d{1,6}(?:[\s,]+\d{1,6})*)?\s*", re.ASCII)


class ErrorFormato(ValueError):
    pass


@dataclass(frozen=True)
class Resultado:
    recompensa: float
    puntaje: int | None
    puntaje_voraz: int
    puntaje_optimo: int
    error: str | None


def parsear_respuesta(instancia: Instancia, respuesta: str) -> set[int]:
    # Se usa el último bloque porque el modelo puede razonar antes de responder.
    bloques = PATRON_BLOQUE.findall(respuesta)
    if not bloques:
        raise ErrorFormato("Falta el bloque <respuesta>...</respuesta>")
    if PATRON_NODOS.fullmatch(bloques[-1]) is None:
        raise ErrorFormato("El bloque <respuesta> solo debe contener números de nodos")

    nodos = [int(nodo) for nodo in re.split(r"[\s,]+", bloques[-1].strip()) if nodo]
    if len(nodos) != len(set(nodos)):
        raise ErrorFormato("Hay nodos repetidos")
    if any(not 1 <= nodo <= instancia.cantidad_nodos for nodo in nodos):
        raise ErrorFormato(f"Los nodos deben estar entre 1 y {instancia.cantidad_nodos}")
    return set(nodos)


def calcular_recompensa(puntaje: int, voraz: int, optimo: int) -> float:
    # Peor que la regla simple: de 0 a RECOMPENSA_VORAZ. Mejor que ella: de RECOMPENSA_VORAZ a 1.
    if optimo == voraz:
        # No hay nada que superar. El generador lo evita, pero una instancia a mano sí podría.
        return max(0, puntaje) / optimo if optimo > 0 else float(puntaje == 0)
    if puntaje <= voraz:
        return RECOMPENSA_VORAZ * max(0, puntaje) / voraz if voraz > 0 else 0.0
    return RECOMPENSA_VORAZ + (1 - RECOMPENSA_VORAZ) * (puntaje - voraz) / (optimo - voraz)


def evaluar_respuesta(instancia: Instancia, respuesta: str) -> Resultado:
    voraz = instancia.puntaje_voraz()
    optimo = instancia.puntaje_optimo()

    try:
        seleccion = parsear_respuesta(instancia, respuesta)
    except ErrorFormato as error:
        return Resultado(RECOMPENSA_FORMATO_INVALIDO, None, voraz, optimo, str(error))

    puntaje = instancia.puntaje(seleccion)
    return Resultado(calcular_recompensa(puntaje, voraz, optimo), puntaje, voraz, optimo, None)


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
