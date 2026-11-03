def leer_datos():
    n, c = map(int, input().split())
    ganancias = list(map(int, input().split()))
    aristas = [[] for _ in range(n)]
    for _ in range(n - 1):
        a, b = map(int, input().split())
        aristas[a - 1].append(b - 1)
        aristas[b - 1].append(a - 1)
    return n, c, ganancias, aristas

def retornar_respuesta(costo):
    print(costo)


def calcular_optimo(n, c, ganancias, aristas):
    # Recorro el arbol para inventarme un orden cualquiera
    padre = [-10000] * n
    orden = [0]
    for nodo_padre in orden:
        for nodo_hijo in aristas[nodo_padre]:
            if nodo_hijo != padre[nodo_padre]:
                padre[nodo_hijo] = nodo_padre
                orden.append(nodo_hijo)

    no_elegir = [0] * n
    elegir = ganancias[:]
    for nodo_padre in reversed(orden[1:]):
        p = padre[nodo_padre]
        # Si no elijo ahora
        # ¿Me conviene haber elegido antes o no?
        no_elegir[p] += max(no_elegir[nodo_padre], elegir[nodo_padre])

        # Si se me ocurre elegir este
        # 1) No elijo el anterior -> Salgo gratis
        # 2) Si elijo el anterior -> Pago 2c
        elegir[p] += max(no_elegir[nodo_padre], elegir[nodo_padre] - 2 * c)

    return max(no_elegir[0], elegir[0])


def solve():
    n, c, ganancias, aristas = leer_datos()
    print(calcular_optimo(n, c, ganancias, aristas))


if __name__ == '__main__':
    solve()
