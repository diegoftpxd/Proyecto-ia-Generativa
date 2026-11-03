# https://www.tiktok.com/@cubypuzzles/video/7636446292096339222
from Piezas import *
from typing import Self, List, Dict
from collections import defaultdict
from collections import deque
from functools import lru_cache
from itertools import combinations


# La tabla solo depende de la forma del tablero, así que los tableros generados la comparten.
@lru_cache(maxsize=64)
def _distancias_a_la_meta(casillas: frozenset, objetivo, cantidad_vacias, tipo_pieza):
    # Realizo BFS desde todas las metas. Un estado es (pieza negra, casillas vacías).
    distancias = {
        (objetivo, frozenset(vacias)): 0
        for vacias in combinations(casillas - {objetivo}, cantidad_vacias)
    }
    cola = deque(distancias)

    while cola:
        estado = cola.popleft()
        negra, vacias = estado

        for vacia in vacias:
            resto = vacias - {vacia}
            vecinos = []

            # La pieza negra solo puede moverse a una casilla vacía
            if vacia in tipo_pieza(*negra).casillas_objetivo:
                vecinos.append((vacia, resto | {negra}))

            # Las demás piezas son comodines: pueden moverse como cualquier pieza
            comodines = (
                Torre(*vacia).casillas_objetivo
                | Arfil(*vacia).casillas_objetivo
                | Caballo(*vacia).casillas_objetivo
            )
            for origen in comodines & casillas - vacias - {negra}:
                vecinos.append((negra, resto | {origen}))

            for vecino in vecinos:
                if vecino not in distancias:
                    distancias[vecino] = distancias[estado] + 1
                    cola.append(vecino)

    return distancias


class Tablero:
    FILAS = 3
    COLUMNAS = 6
    CASILLA_OBJETIVO = (5, 2)
    FUERA = "##"
    VACIA = ". "

    @classmethod
    def _calcular_heuristica(cls):
        inicial = cls()
        casillas = frozenset(
            (x, y)
            for x in range(cls.COLUMNAS)
            for y in range(cls.FILAS)
            if inicial._es_casilla_valida((x, y))
        )
        return _distancias_a_la_meta(
            casillas,
            cls.CASILLA_OBJETIVO,
            len(inicial.casillas_vacias),
            type(inicial.pieza_objetivo()),
        )


    @classmethod
    def nombre_casilla(cls, posicion):
        # Convierte una posición interna a la notación que se imprime. Ej: (5, 2) -> "f1"
        x, y = posicion
        return f"{chr(ord('a') + x)}{cls.FILAS - y}"


    @classmethod
    def heuristica_precomputada(cls):
        # Se calcula 1 sola vez y luego se deja guardado en la clase.
        # Se revisa __dict__ para que una subclase no herede la tabla de otro tablero.
        if "_heuristica" not in cls.__dict__:
            cls._heuristica = cls._calcular_heuristica()
        return cls._heuristica

    def pieza_objetivo(self) -> Pieza:
        for pieza in self.casillas_actuales.values():
            if pieza is not None and pieza.color == "Negro":
                return pieza

    def heuristica(self):
        # La heurística es: ¿Cuantos movimientos harían falta para llevar la pieza negra
        # al objetivo si las demás piezas pudieran moverse como cualquier pieza?
        posicion = self.pieza_objetivo().posicion
        return type(self).heuristica_precomputada()[(posicion, frozenset(self.casillas_vacias))]


    def __init__(self, casillas_actuales=None, casillas_vacias=None, largo_camino = 0):
        if casillas_actuales is None and casillas_vacias is None:
            self._iniciar_posicion()
        else:
            self.casillas_actuales: Dict[Pieza] = casillas_actuales
            self.casillas_vacias = casillas_vacias
        self.largo_camino = largo_camino
        self._actualizar_estado()

    def _iniciar_posicion(self):
        piezas_inicio = [
            Caballo(0, 0, "Negro"),
            Arfil(1, 0),
            Arfil(2, 0),
            Arfil(3, 0),
            Arfil(4, 0),
            Torre(5, 0),
            Caballo(0, 1),
            Caballo(1, 1),
            Caballo(2, 1),
            Caballo(3, 1),
            Torre(4, 1),
            Torre(5, 1),
            Torre(4, 2),
        ]

        self.casillas_actuales = defaultdict(
            lambda: None,
            {pieza.posicion: pieza for pieza in piezas_inicio}
        )

        self.casillas_actuales[self.CASILLA_OBJETIVO] = None
        self.casillas_vacias = {self.CASILLA_OBJETIVO}

    def _actualizar_estado(self):
        self.estado = tuple(
            None
            if self.casillas_actuales[(i, j)] is None
            else (
                type(self.casillas_actuales[(i, j)]).__name__,
                self.casillas_actuales[(i, j)].color,
            )
            for j in range(self.FILAS)
            for i in range(self.COLUMNAS)
        )

    def _mover_pieza(self, origen, objetivo):
        self.casillas_actuales[objetivo] = self.casillas_actuales[origen]
        self.casillas_actuales[origen] = None

        self.casillas_vacias.discard(objetivo)
        self.casillas_vacias.add(origen)

        self.casillas_actuales[objetivo].mover(objetivo)

        self._actualizar_estado()

    def movimientos_legales(self) -> List[tuple]:
        movimientos = []

        for origen, pieza in self.casillas_actuales.items():
            if pieza is not None:
                for vacia in self.casillas_vacias:
                    if pieza.se_puede_mover_a(vacia):
                        movimientos.append((origen, vacia))

        return movimientos

    def obtener_tablero_alcanzables(self) -> List[Self]:
        return [
            self.aplicar_movimiento(origen, destino)
            for origen, destino in self.movimientos_legales()
        ]

    def aplicar_movimiento(self, origen, destino) -> Self:
        nuevo_tablero = self._copy()
        nuevo_tablero._mover_pieza(origen, destino)
        return nuevo_tablero

    def es_tablero_terminado(self):
        pieza = self.casillas_actuales[self.CASILLA_OBJETIVO]
        if pieza is None:
            return False
        return pieza.color == "Negro"

    def _copy(self) -> Self:
        casillas_copiadas = defaultdict(lambda: None)
        for posicion, pieza in self.casillas_actuales.items():
            casillas_copiadas[posicion] = (
                pieza.copy() if pieza is not None else None
            )

        return type(self)(casillas_copiadas, set(self.casillas_vacias), self.largo_camino + 1)

    def _es_casilla_valida(self, posicion):
        # Una casilla pertenece al tablero si tiene una pieza o está registrada como vacía
        return (
            self.casillas_actuales.get(posicion) is not None
            or posicion in self.casillas_vacias
        )


    def __repr__(self):
        columnas = "    ".join(chr(ord("a") + i) for i in range(self.COLUMNAS))
        lineas = [f"    {columnas}"]

        for j in range(self.FILAS - 1, -1, -1):
            linea = f"{j + 1} |"
            for i in range(self.COLUMNAS):
                posicion = (i, (self.FILAS - 1) - j)
                pieza = self.casillas_actuales.get(posicion)

                if pieza is not None:
                    linea += f" {repr(pieza)} |"
                elif posicion in self.casillas_vacias:
                    linea += f" {self.VACIA} |"
                else:
                    linea += f" {self.FUERA} |"
            lineas.append(linea)

        return "\n".join(lineas)


class TableroGrande(Tablero):
    FILAS = 5
    COLUMNAS = 6
    CASILLA_OBJETIVO = (5, 4)

    def _iniciar_posicion(self):
        piezas_inicio = [
            # Fila 0
            Caballo(0, 0, "Negro"),
            Arfil(1, 0),
            Arfil(2, 0),
            Arfil(3, 0),
            Arfil(4, 0),
            Torre(5, 0),
            # Fila 1
            Caballo(0, 1),
            Caballo(1, 1),
            Caballo(2, 1),
            Caballo(3, 1),
            Torre(4, 1),
            Torre(5, 1),
            Torre(4, 2),
            # Fila 2
            Caballo(0, 2),
            Arfil(1, 2),
            Arfil(2, 2),
            Caballo(3, 2),
            Arfil(5, 2),
            # Fila 3
            Torre(0, 3),
            Torre(1, 3),
            Caballo(2, 3),
            Arfil(3, 3),
            Caballo(4, 3),
            Arfil(5, 3),
            # Fila 4
            Torre(0, 4),
            Torre(1, 4),
            Arfil(2, 4),
            Arfil(3, 4),
            #Torre(4, 4)
        ]

        self.casillas_actuales = defaultdict(
            lambda: None,
            {pieza.posicion: pieza for pieza in piezas_inicio}
        )

        self.casillas_vacias = {(4,4)}

        self.casillas_actuales[self.CASILLA_OBJETIVO] = None
        self.casillas_vacias.add(self.CASILLA_OBJETIVO)





