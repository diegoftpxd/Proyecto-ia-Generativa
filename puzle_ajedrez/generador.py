import random
from collections import defaultdict
from dataclasses import dataclass

from busqueda import a_estrella
from Piezas import Arfil, Caballo, Torre
from tablero import Tablero

PASOS_MAXIMOS = 300
INTENTOS_MAXIMOS = 2000


@dataclass(frozen=True)
class Nivel:
    filas: int
    columnas: int
    vacias: int
    huecos: int
    min_movimientos: int
    max_movimientos: int
    pieza_objetivo: type = Caballo

    def __post_init__(self):
        # Sobre 9 filas se desalinea la impresión y sobre 26 columnas se acaban las letras
        if not (1 <= self.filas <= 9 and 1 <= self.columnas <= 26):
            raise ValueError("El tablero debe tener entre 1 y 9 filas y entre 1 y 26 columnas")
        if self.vacias < 1 or self.huecos < 0:
            raise ValueError("Debe haber al menos una casilla vacía y los huecos no pueden ser negativos")
        # El caballo negro también necesita una casilla
        if self.vacias + self.huecos + 1 > self.filas * self.columnas:
            raise ValueError("Las casillas vacías y los huecos no caben en el tablero")
        if not 1 <= self.min_movimientos <= self.max_movimientos:
            raise ValueError("Se requiere 1 <= min_movimientos <= max_movimientos")


@dataclass(frozen=True)
class Ejercicio:
    tablero: Tablero
    movimientos_optimos: int


# Currículum de menor a mayor largo óptimo.
NIVELES = [
    Nivel(filas=3, columnas=3, vacias=1, huecos=0, min_movimientos=1, max_movimientos=2),
    Nivel(filas=3, columnas=4, vacias=1, huecos=0, min_movimientos=3, max_movimientos=4),
    Nivel(filas=3, columnas=6, vacias=1, huecos=4, min_movimientos=5, max_movimientos=6),
    Nivel(filas=4, columnas=4, vacias=1, huecos=0, min_movimientos=7, max_movimientos=8),
    Nivel(filas=5, columnas=6, vacias=2, huecos=0, min_movimientos=9, max_movimientos=10),
    Nivel(filas=4, columnas=5, vacias=1, huecos=0, min_movimientos=11, max_movimientos=12),
    Nivel(filas=5, columnas=5, vacias=1, huecos=0, min_movimientos=13, max_movimientos=14),
    Nivel(filas=3, columnas=6, vacias=1, huecos=4, min_movimientos=15, max_movimientos=18),
]


def _crear_tablero_resuelto(nivel: Nivel, rng: random.Random) -> Tablero:
    # Cada disposición es una subclase de Tablero, igual que TableroGrande.
    objetivo = (nivel.columnas - 1, nivel.filas - 1)
    casillas = [
        (x, y)
        for x in range(nivel.columnas)
        for y in range(nivel.filas)
        if (x, y) != objetivo
    ]
    rng.shuffle(casillas)

    fin_huecos = nivel.huecos
    fin_vacias = fin_huecos + nivel.vacias
    vacias = casillas[fin_huecos:fin_vacias]
    piezas = [nivel.pieza_objetivo(*objetivo, "Negro")]
    piezas += [rng.choice((Torre, Arfil, Caballo))(*casilla) for casilla in casillas[fin_vacias:]]

    def _iniciar_posicion(self):
        self.casillas_actuales = defaultdict(
            lambda: None,
            {pieza.posicion: pieza.copy() for pieza in piezas}
        )
        self.casillas_vacias = set(vacias)

    clase = type("TableroGenerado", (Tablero,), {
        "FILAS": nivel.filas,
        "COLUMNAS": nivel.columnas,
        "CASILLA_OBJETIVO": objetivo,
        "_iniciar_posicion": _iniciar_posicion,
    })
    return clase()


def _revolver(tablero: Tablero, pasos: int, rng: random.Random) -> Tablero:
    for _ in range(pasos):
        vecinos = tablero.obtener_tablero_alcanzables()
        if not vecinos:
            break
        tablero = rng.choice(vecinos)

    # El tablero revuelto es el inicio del ejercicio, así que su camino parte en 0.
    return type(tablero)(tablero.casillas_actuales, tablero.casillas_vacias)


def generar_ejercicio(nivel: Nivel, rng: random.Random | None = None) -> Ejercicio:
    # Los movimientos son reversibles: al revolver desde un tablero resuelto, el resultado
    # siempre tiene solución. A* mide el largo óptimo y se descartan los que no calzan.
    rng = rng or random.Random()

    for _ in range(INTENTOS_MAXIMOS):
        resuelto = _crear_tablero_resuelto(nivel, rng)
        tablero = _revolver(resuelto, rng.randint(1, PASOS_MAXIMOS), rng)
        if tablero.es_tablero_terminado():
            continue

        _, movimientos_optimos, _ = a_estrella(tablero)
        if nivel.min_movimientos <= movimientos_optimos <= nivel.max_movimientos:
            return Ejercicio(tablero, movimientos_optimos)

    raise RuntimeError(f"No se encontró un tablero para {nivel} en {INTENTOS_MAXIMOS} intentos")
