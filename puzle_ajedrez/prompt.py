from Piezas import Arfil, Caballo, Torre
from tablero import Tablero

NOMBRES_OBJETIVO = {
    Caballo: "el caballo negro",
    Arfil: "el alfil negro",
    Torre: "la torre negra",
}


def generar_prompt(tablero: Tablero) -> str:
    objetivo = tablero.nombre_casilla(tablero.CASILLA_OBJETIVO)
    pieza = tablero.pieza_objetivo()

    return f"""Lleva {NOMBRES_OBJETIVO[type(pieza)]} ({pieza!r}) a {objetivo} con la menor cantidad de movimientos.

{tablero!r}

CB: caballo blanco. AB: alfil blanco. TB: torre blanca. ".": vacía. "##": no existe.
Cada movimiento lleva una pieza a una casilla vacía. No hay capturas.
Caballo: salta en L. Torre: 1 casilla recta. Alfil: 1 casilla en diagonal.

Usa movimientos_legales para ver las opciones y mover para jugar.
No razones de memoria ni expliques: llama a una herramienta en cada turno."""
