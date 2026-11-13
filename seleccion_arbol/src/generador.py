import random
from dataclasses import dataclass

from src.instancia import Instancia

INTENTOS_MAXIMOS = 2000


@dataclass(frozen=True)
class Nivel:
    nodos: int
    valor_minimo: int
    valor_maximo: int
    costo_maximo: int
    # Fracción máxima del óptimo que puede lograr la regla de elegir todos los positivos.
    voraz_maximo: float = 0.7
    # Fracción máxima del óptimo que puede lograr el mejor atajo (regla simple o "sin vecinos").
    atajo_maximo: float = 1.0

    def __post_init__(self):
        if self.nodos < 1:
            raise ValueError("El árbol debe tener al menos un nodo")
        if self.valor_minimo > self.valor_maximo:
            raise ValueError("Se requiere valor_minimo <= valor_maximo")
        if self.valor_maximo < 1:
            raise ValueError("Se requiere valor_maximo >= 1 para que exista un puntaje positivo")
        if self.costo_maximo < 1:
            raise ValueError("Se requiere costo_maximo >= 1")
        if not 0 <= self.voraz_maximo <= 1:
            raise ValueError("Se requiere 0 <= voraz_maximo <= 1")
        if not 0 <= self.atajo_maximo <= 1:
            raise ValueError("Se requiere 0 <= atajo_maximo <= 1")


# Currículum de menor a mayor dificultad. Los niveles 1 y 2 admiten árboles que un atajo resuelve,
# para que el modelo tenga aciertos de los que aprender. Desde el nivel 3 hay que optimizar.
NIVELES = [
    Nivel(nodos=3,  valor_minimo=1,   valor_maximo=10, costo_maximo=3),
    Nivel(nodos=4,  valor_minimo=1,   valor_maximo=10, costo_maximo=3),
    Nivel(nodos=5,  valor_minimo=-5,  valor_maximo=10, costo_maximo=3,  atajo_maximo=0.9),
    Nivel(nodos=7,  valor_minimo=-5,  valor_maximo=10, costo_maximo=4,  atajo_maximo=0.9),
    Nivel(nodos=10, valor_minimo=-10, valor_maximo=15, costo_maximo=5,  atajo_maximo=0.9),
    Nivel(nodos=15, valor_minimo=-10, valor_maximo=20, costo_maximo=6,  atajo_maximo=0.9),
    Nivel(nodos=25, valor_minimo=-20, valor_maximo=30, costo_maximo=8,  atajo_maximo=0.9),
    Nivel(nodos=40, valor_minimo=-30, valor_maximo=40, costo_maximo=10, atajo_maximo=0.9),
]


def _crear_instancia(nivel: Nivel, rng: random.Random) -> Instancia:
    # Cada nodo se une a uno anterior. Números, orden y sentido de las aristas se barajan para ocultar la raíz.
    numeros = list(range(1, nivel.nodos + 1))
    rng.shuffle(numeros)
    aristas = [
        (numeros[i], numeros[rng.randrange(i)])
        for i in range(1, nivel.nodos)
    ]
    rng.shuffle(aristas)
    aristas = tuple((a, b) if rng.random() < 0.5 else (b, a) for a, b in aristas)
    valores = tuple(rng.randint(nivel.valor_minimo, nivel.valor_maximo) for _ in numeros)
    return Instancia(valores, aristas, rng.randint(1, nivel.costo_maximo))


def generar_instancia(nivel: Nivel, rng: random.Random | None = None) -> Instancia:
    rng = rng or random.Random()

    for _ in range(INTENTOS_MAXIMOS):
        instancia = _crear_instancia(nivel, rng)
        optimo = instancia.puntaje_optimo()

        if (
            optimo > 0
            and instancia.puntaje_voraz() <= nivel.voraz_maximo * optimo
            and instancia.puntaje_referencia() <= nivel.atajo_maximo * optimo
        ):
            return instancia

    raise RuntimeError(f"No se encontró una instancia para {nivel} en {INTENTOS_MAXIMOS} intentos")
