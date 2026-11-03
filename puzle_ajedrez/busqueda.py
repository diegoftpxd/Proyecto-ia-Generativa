from collections import deque
from tablero import Tablero, TableroGrande

import heapq





class Cola:
    def __init__(self, iterable=()):
        self._datos = deque(iterable)

    def push(self, x):
        self._datos.append(x)

    def pop(self):
        return self._datos.popleft()

    def __bool__(self):
        return bool(self._datos)


class Stack:
    def __init__(self, iterable=()):
        self._datos = deque(iterable)

    def push(self, x):
        self._datos.append(x)

    def pop(self):
        return self._datos.pop()

    def __bool__(self):
        return bool(self._datos)

class ColaPrioridad:
    def __init__(self, iterable=()):
        self._datos = []
        self._contador = 0
        for x in iterable:
            self.push(x)

    def push(self, tablero:Tablero):
        h = tablero.heuristica()
        f = tablero.largo_camino + h
        #f(x) = g(x) + h(x). Si f empata, se prefiere el tablero con menor h.
        heapq.heappush(self._datos, (f, h, self._contador, tablero))
        self._contador += 1

    def pop(self):
        return heapq.heappop(self._datos)[3]

    def __bool__(self):
        return bool(self._datos)



def reconstruir_camino(estado_final, padre, estados):
    camino = []

    while estado_final is not None:
        camino.append(estados[estado_final])
        estado_final = padre[estado_final]

    camino.reverse()
    return camino


def busqueda(tablero_inicial: Tablero, estructura):

    frontera = estructura([tablero_inicial])
    visitados = {tablero_inicial.estado}
    diccionario_de_padres = {tablero_inicial.estado: None}
    estados = {tablero_inicial.estado: tablero_inicial}

    while frontera:
        tablero = frontera.pop()

        # Si terminamos generamos la respuesta y retornamos
        if tablero.es_tablero_terminado():
            camino = reconstruir_camino(
                tablero.estado,
                diccionario_de_padres,
                estados,
            )
            return camino, len(camino) - 1, len(visitados)
        
        # Analizo todos los nodos vecinos
        for vecino in tablero.obtener_tablero_alcanzables():
            if vecino.estado in visitados:
                continue
            # Añado los vecinos a la frontera y guardo sus datos.
            visitados.add(vecino.estado)
            diccionario_de_padres[vecino.estado] = tablero.estado
            estados[vecino.estado] = vecino
            frontera.push(vecino)

    return [], -1, -1


def bfs(tablero):
    return busqueda(tablero, Cola)

def dfs(tablero):
    return busqueda(tablero, Stack)














def a_estrella(tablero_inicial: Tablero):
    frontera = ColaPrioridad([tablero_inicial])
    visitados = {tablero_inicial.estado}
    diccionario_de_padres = {tablero_inicial.estado: None}
    estados = {tablero_inicial.estado: tablero_inicial}

    while frontera:
        tablero = frontera.pop()

        if tablero.es_tablero_terminado():
            camino = reconstruir_camino(
                tablero.estado,
                diccionario_de_padres,
                estados,
            )
            return camino, len(camino) - 1, len(visitados)

        for vecino in tablero.obtener_tablero_alcanzables():
            
            # La única diferencia es que ahora, si el nodo ya fue visitado. 
            if vecino.estado in visitados and vecino.largo_camino >= estados[vecino.estado].largo_camino:
                continue

            diccionario_de_padres[vecino.estado] = tablero.estado
            estados[vecino.estado] = vecino
            visitados.add(vecino.estado)

            frontera.push(vecino)

    return [], -1, -1


def mostrar_resultados(respuesta):
    camino, largo_optimo, largo_nodos_visitados = respuesta

    print("=" * 40)
    print("Resumen de la búsqueda")
    print("=" * 40)
    print(f"Cantidad de movimientos : {largo_optimo}")
    print(f"Nodos visitados         : {largo_nodos_visitados}")
    print(f"Estados en el camino    : {len(camino)}")
    print("=" * 40)

    for i, tablero in enumerate(camino):
        input(f"\nPaso {i}/{largo_optimo} (Enter para continuar)")
        print(tablero)




if __name__ == "__main__":
    mostrar_resultados(a_estrella(TableroGrande()))


