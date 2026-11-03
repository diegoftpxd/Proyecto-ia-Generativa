import json
import random
from dataclasses import dataclass

from busqueda import a_estrella
from generador import Ejercicio, Nivel, generar_ejercicio
from modelo import ModeloLLM, ModeloLMStudio
from prompt import generar_prompt

# Las bandas no se traslapan: sin movimientos < sin resolver < resuelto.
RECOMPENSA_SIN_MOVIMIENTOS = -1.0
RECOMPENSA_SIN_RESOLVER = -0.5  # Sube hasta 0 según el progreso logrado
RECOMPENSA_RESUELTO = 0.5  # Sube hasta 1 según qué tan corta es la solución

# Cada movimiento suele requerir 2 llamadas: ver las opciones y mover.
LLAMADAS_POR_MOVIMIENTO_OPTIMO = 4
LLAMADAS_ADICIONALES = 4


@dataclass(frozen=True)
class Resultado:
    recompensa: float
    resuelto: bool
    movimientos_validos: int
    movimientos_ilegales: int
    movimientos_optimos: int


class EntornoPuzle:
    # La conversación completa del episodio queda en self.mensajes.
    def __init__(self, semilla: int | None = None):
        self._rng = random.Random(semilla)
        self.ejercicio: Ejercicio | None = None
        self.tablero = None
        self.mensajes: list[dict] = []
        self.terminado = True

    def reset(self, nivel: Nivel) -> list[dict]:
        return self.iniciar(generar_ejercicio(nivel, self._rng))

    def iniciar(self, ejercicio: Ejercicio) -> list[dict]:
        self.ejercicio = ejercicio
        self.tablero = ejercicio.tablero
        self.mensajes = [{"role": "user", "content": generar_prompt(ejercicio.tablero)}]
        self.terminado = False
        self._llamadas = 0
        self._ilegales = 0
        return self.mensajes

    def step(self, mensaje: dict) -> bool:
        # Recibe la respuesta del modelo y devuelve True si el episodio terminó.
        if self.terminado:
            raise RuntimeError("El episodio no está activo: hay que llamar a reset")

        self.mensajes.append(mensaje)
        self._llamadas += 1
        llamadas = mensaje.get("tool_calls") or []

        if not llamadas:
            # Un turno sin herramientas gasta una llamada, pero no termina el episodio.
            self.mensajes.append({"role": "user", "content": "Debes llamar a una herramienta."})

        for llamada in llamadas:
            funcion = llamada["function"]
            contenido = self._ejecutar_herramienta(funcion["name"], funcion.get("arguments"))
            self.mensajes.append({"role": "tool", "tool_call_id": llamada["id"], "content": contenido})
            if self.tablero.es_tablero_terminado():
                break

        limite = (
            LLAMADAS_POR_MOVIMIENTO_OPTIMO * self.ejercicio.movimientos_optimos
            + LLAMADAS_ADICIONALES
        )
        self.terminado = self.tablero.es_tablero_terminado() or self._llamadas >= limite
        return self.terminado

    def _opciones(self) -> dict[str, tuple]:
        return {
            f"{self.tablero.casillas_actuales[origen]!r} "
            f"{self.tablero.nombre_casilla(origen)}-{self.tablero.nombre_casilla(destino)}": (origen, destino)
            for origen, destino in self.tablero.movimientos_legales()
        }

    def herramientas(self) -> list[dict]:
        # mover solo acepta los movimientos legales del turno, así que cambia en cada llamada.
        return [
            {
                "type": "function",
                "function": {
                    "name": "movimientos_legales",
                    "description": "Lista los movimientos legales del tablero actual.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mover",
                    "description": "Mueve una pieza y devuelve el tablero actualizado.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "movimiento": {"type": "string", "enum": sorted(self._opciones())},
                        },
                        "required": ["movimiento"],
                    },
                },
            },
        ]

    def _listar_movimientos(self) -> str:
        return "\n".join(sorted(self._opciones()))

    def _error(self, motivo) -> str:
        return f"Error: {motivo} Movimientos legales:\n{self._listar_movimientos()}"

    def _ejecutar_herramienta(self, nombre, argumentos) -> str:
        if nombre == "movimientos_legales":
            return self._listar_movimientos()
        if nombre == "mover":
            return self._mover(argumentos)
        return self._error(f"la herramienta {nombre} no existe.")

    def _mover(self, argumentos) -> str:
        # Los argumentos los escribe el modelo, así que pueden venir mal formados.
        try:
            movimiento = json.loads(argumentos)["movimiento"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return self._error('mover requiere el argumento {"movimiento": "CN a2-c1"}.')

        opciones = self._opciones()
        if not isinstance(movimiento, str) or movimiento not in opciones:
            self._ilegales += 1
            return self._error(f"{movimiento} no es legal. El tablero no cambió.")

        self.tablero = self.tablero.aplicar_movimiento(*opciones[movimiento])
        if self.tablero.es_tablero_terminado():
            return "Puzle resuelto."
        return repr(self.tablero)

    def resultado(self) -> Resultado:
        if self.ejercicio is None or not self.terminado:
            raise RuntimeError("El episodio aún no termina")

        optimo = self.ejercicio.movimientos_optimos
        validos = self.tablero.largo_camino
        resuelto = self.tablero.es_tablero_terminado()

        if resuelto:
            recompensa = RECOMPENSA_RESUELTO + 0.5 * optimo / validos
        elif validos == 0:
            recompensa = RECOMPENSA_SIN_MOVIMIENTOS
        else:
            # El progreso es cuánto se acortó la distancia real al objetivo.
            _, optimo_restante, _ = a_estrella(self.tablero)
            recompensa = RECOMPENSA_SIN_RESOLVER + 0.5 * max(0.0, (optimo - optimo_restante) / optimo)

        return Resultado(recompensa, resuelto, validos, self._ilegales, optimo)


def jugar_episodio(entorno: EntornoPuzle, modelo: ModeloLLM, nivel: Nivel) -> Resultado:
    entorno.reset(nivel)
    while not entorno.step(modelo.conversar(entorno.mensajes, entorno.herramientas())):
        pass
    return entorno.resultado()


if __name__ == "__main__":
    from generador import NIVELES

    entorno = EntornoPuzle()
    print(jugar_episodio(entorno, ModeloLMStudio(), NIVELES[0]))
